"""
Audio capture via soundcard WASAPI/PulseAudio loopback.

Captures the system audio output as float32 numpy chunks in a thread-safe
queue — no .wav files, no disk I/O on the hot path.

A `PlaybackMuteGate` prevents the capture pipeline from re-ingesting audio
that our own TTS playback produces, avoiding feedback loops.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

try:
    import soundcard as sc  # type: ignore[import-untyped]
except (ImportError, ModuleNotFoundError, OSError):  # OSError when libpulse/libcoreAudio missing
    sc = None  # type: ignore[assignment]

# Minimum amplitude to treat as a real signal rather than hardware noise floor.
_SILENCE_GATE: float = 0.001
# Target amplitude after software AGC
_AGC_TARGET: float = 0.4
# Maximum AGC gain multiplier
_AGC_MAX_GAIN: float = 10.0


class PlaybackMuteGate:
    """
    Threading gate that signals active TTS playback.

    Set ``active`` while audio is being played back so that the capture
    pipeline can suppress that window and prevent feedback.
    """

    def __init__(self) -> None:
        self._active = threading.Event()

    def open(self) -> None:
        """Mark TTS playback as active — capture pipeline should mute."""
        self._active.set()

    def close(self) -> None:
        """Mark TTS playback as done — capture pipeline resumes."""
        self._active.clear()

    @property
    def is_active(self) -> bool:
        return self._active.is_set()


class AudioCapture:
    """
    Captures system audio using ``soundcard`` loopback (WASAPI on Windows,
    PulseAudio/ALSA on Linux).  Produces float32 mono numpy chunks and pushes
    them into an in-memory queue.

    Integration notes:
    - Pass a :class:`PlaybackMuteGate` so the pipeline can suppress TTS
      feedback.  While the gate is open chunks are silently discarded.
    - Audio is normalised with a software AGC to a target peak of ~0.4 so
      downstream VAD and TTS models receive a consistent signal level.
    """

    def __init__(
        self,
        device_name: str | None = None,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 512,
        mute_gate: PlaybackMuteGate | None = None,
    ) -> None:
        self.device_name = device_name
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize  # ~32 ms at 16 kHz
        self.mute_gate = mute_gate or PlaybackMuteGate()
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.logger = logging.getLogger("modern_bonzi_buddy.audio")
        self.is_recording = False
        self._thread: threading.Thread | None = None
        self._log_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_audio_devices(self) -> list[tuple[int, str]]:
        """Return ``(index, name)`` pairs for all loopback/input devices."""
        if sc is None:
            return []
        devices = sc.all_microphones(include_loopback=True)
        return [(i, dev.name) for i, dev in enumerate(devices)]

    def start_recording(self) -> None:
        """Start the background capture thread."""
        if self.is_recording:
            return
        # Flush any stale audio from a previous session
        _drain_queue(self.audio_queue)
        self.is_recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop_recording(self) -> None:
        """Gracefully stop the capture thread."""
        self.is_recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.logger.info("Audio capture stopped.")

    def get_chunk(self, timeout: float | None = None) -> np.ndarray | None:
        """Pop the next float32 mono numpy chunk, or return None on timeout."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_loop(self) -> None:
        if sc is None:
            self.logger.error(
                "soundcard is not installed. Install with: pip install soundcard"
            )
            self.is_recording = False
            return

        target_id = self.device_name
        if target_id is None:
            try:
                target_id = sc.default_speaker().name
                self.logger.info(f"Loopback targeting default speaker: {target_id!r}")
            except Exception as exc:
                self.logger.error(f"Could not resolve default speaker: {exc}")
                self.is_recording = False
                return

        try:
            mic = sc.get_microphone(id=target_id, include_loopback=True)
        except Exception as exc:
            self.logger.error(f"Could not attach loopback microphone: {exc}")
            self.is_recording = False
            return

        try:
            with mic.recorder(
                samplerate=self.samplerate, channels=self.channels
            ) as recorder:
                self.logger.info(f"Capture started on loopback: {mic.name!r}")
                while self.is_recording:
                    data: np.ndarray = recorder.record(numframes=self.blocksize)
                    chunk = _to_float32_mono(data)
                    chunk = _apply_agc(chunk)

                    self._log_counter += 1
                    if self._log_counter % 100 == 0:
                        peak = float(np.abs(chunk).max())
                        self.logger.debug(
                            f"Capture peak (post-AGC): {peak:.5f} | "
                            f"gate={'open' if self.mute_gate.is_active else 'closed'}"
                        )

                    # Suppress our own TTS output — discard chunk while playing back
                    if self.mute_gate.is_active:
                        continue

                    self.audio_queue.put(chunk)
        except Exception as exc:
            self.logger.error(f"Exception in capture loop: {exc}")
            self.is_recording = False


# ---------------------------------------------------------------------------
# Helpers — module-level so they are easily unit-tested
# ---------------------------------------------------------------------------


def _to_float32_mono(data: np.ndarray) -> np.ndarray:
    """Convert a recorder output array to a 1-D float32 mono array.

    If the data is already single-channel we use ``ravel()`` (zero-copy view)
    instead of ``mean(axis=1)`` to avoid an unnecessary allocation.
    """
    if data.ndim == 2:
        if data.shape[1] == 1:
            # Bolt: single-channel ravel() is O(1) vs. mean() O(N)
            return data.ravel().astype(np.float32, copy=False)
        return data.mean(axis=1).astype(np.float32)
    return data.astype(np.float32, copy=False)


def _apply_agc(chunk: np.ndarray) -> np.ndarray:
    """Apply software AGC: boost quiet signals toward *_AGC_TARGET* peak.

    Signals below *_SILENCE_GATE* are considered noise floor and passed through
    unmodified so we never blow up hardware silence to a usable level.
    """
    peak = float(np.abs(chunk).max())
    if peak < _SILENCE_GATE:
        return chunk
    gain = min(_AGC_MAX_GAIN, _AGC_TARGET / max(peak, 0.01))
    if gain > 1.0:
        return chunk * gain
    return chunk


def _drain_queue(q: queue.Queue) -> None:  # type: ignore[type-arg]
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
