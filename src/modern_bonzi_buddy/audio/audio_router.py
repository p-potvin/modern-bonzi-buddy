"""
High-level audio routing facade.

Wraps :class:`~modern_bonzi_buddy.audio.audio_capture.AudioCapture` and
:class:`~modern_bonzi_buddy.audio.audio_capture.PlaybackMuteGate` to provide
the pipeline with a single start/stop interface.
"""
from __future__ import annotations

from modern_bonzi_buddy.audio.audio_capture import AudioCapture, PlaybackMuteGate


class AudioRouter:
    """Facade over AudioCapture + PlaybackMuteGate for pipeline use."""

    def __init__(
        self,
        device_name: str | None = None,
        samplerate: int = 16000,
        blocksize: int = 512,
    ) -> None:
        self.mute_gate = PlaybackMuteGate()
        self.capture = AudioCapture(
            device_name=device_name,
            samplerate=samplerate,
            blocksize=blocksize,
            mute_gate=self.mute_gate,
        )
        self.running = False

    def start(self) -> None:
        self.capture.start_recording()
        self.running = True

    def stop(self) -> None:
        self.capture.stop_recording()
        self.running = False
