"""
TTS audio playback using sounddevice.

Plays PCM bytes (from Gemini Flash TTS) as a float32 numpy array
directly via sounddevice — no .wav files written to disk.

While playback is active the :class:`~modern_bonzi_buddy.audio.audio_capture.PlaybackMuteGate`
is held open so the capture pipeline suppresses its own output.
"""
from __future__ import annotations

import io
import logging
import struct
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modern_bonzi_buddy.audio.audio_capture import PlaybackMuteGate

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    sd = None  # type: ignore[assignment]

logger = logging.getLogger("modern_bonzi_buddy.playback")

# Gemini Flash TTS default: 24 kHz, 16-bit linear PCM, mono
_GEMINI_SAMPLE_RATE = 24000


def pcm_bytes_to_float32(pcm: bytes, dtype: str = "int16") -> np.ndarray:
    """Convert raw PCM bytes to a float32 numpy array in [-1.0, 1.0].

    Gemini Flash TTS returns 16-bit signed PCM.  We decode it directly from
    the byte buffer using numpy — no soundfile, no disk access.
    """
    raw = np.frombuffer(pcm, dtype=np.dtype(dtype))
    # Normalise to [-1.0, 1.0] based on dtype range
    info = np.iinfo(raw.dtype)
    return raw.astype(np.float32) / float(info.max)


def play_audio(
    pcm_bytes: bytes,
    *,
    samplerate: int = _GEMINI_SAMPLE_RATE,
    mute_gate: "PlaybackMuteGate | None" = None,
    blocking: bool = True,
) -> None:
    """Play raw PCM bytes as audio.

    Parameters
    ----------
    pcm_bytes:
        Raw PCM audio returned by Gemini Flash TTS (16-bit signed, mono).
    samplerate:
        Sample rate of the PCM data.  Defaults to Gemini's 24 kHz output.
    mute_gate:
        If provided, the gate is opened before playback and closed after so
        the capture pipeline suppresses any loopback of our own audio.
    blocking:
        If True (default) the call blocks until playback finishes.
    """
    if sd is None:
        logger.warning(
            "sounddevice is not installed; audio playback is disabled. "
            "Install with: pip install sounddevice"
        )
        return

    audio_array = pcm_bytes_to_float32(pcm_bytes)

    if mute_gate is not None:
        mute_gate.open()

    try:
        sd.play(audio_array, samplerate=samplerate)
        if blocking:
            sd.wait()
    finally:
        if mute_gate is not None:
            mute_gate.close()


def play_audio_nonblocking(
    pcm_bytes: bytes,
    *,
    samplerate: int = _GEMINI_SAMPLE_RATE,
    mute_gate: "PlaybackMuteGate | None" = None,
) -> threading.Thread:
    """Fire-and-forget version of :func:`play_audio`.

    Returns the daemon thread so callers can join if needed.
    """
    t = threading.Thread(
        target=play_audio,
        kwargs={"pcm_bytes": pcm_bytes, "samplerate": samplerate, "mute_gate": mute_gate, "blocking": True},
        daemon=True,
    )
    t.start()
    return t
