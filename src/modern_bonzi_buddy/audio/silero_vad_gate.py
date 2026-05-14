from __future__ import annotations

from typing import Any

class SileroVadGate:
    """Thin wrapper over Silero VAD for speech/silence gating."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> Any:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model
