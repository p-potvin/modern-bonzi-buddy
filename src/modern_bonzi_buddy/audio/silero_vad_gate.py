"""
Silero VAD wrapper for real-time speech activity detection.

Ported from realtime-stt with the following optimizations retained:
- Absolute silence gate prevents feeding hardware noise floor into the stateful model
- Native ``np.abs(x).max()`` peak calculation is ~2x faster than ``np.max(np.abs(x))``
- Float32 normalisation avoids GPU tensor copies where possible
- CUDA → CPU fallback on OOM
"""
from __future__ import annotations

import logging

import numpy as np

try:
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Minimum peak amplitude to treat audio as a real signal rather than
# hardware noise floor.  Values below this are returned as 0.0 probability
# without ever touching the stateful VAD model.
_SILENCE_GATE: float = 0.005


class SileroVadGate:
    """
    Lightweight, reusable Silero VAD wrapper for real-time speech detection.

    Uses ``torch.hub`` to load the model on first instantiation.  Supports
    CPU and CUDA inference with an automatic CPU fallback on OOM.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        device: str = "cpu",
        logger_name: str = "modern_bonzi_buddy.vad",
        bypass: bool = False,
    ) -> None:
        self.samplerate = samplerate
        self.logger = logging.getLogger(logger_name)
        self.bypass = bypass
        self._model = None

        if torch is None:
            self.logger.warning(
                "torch is not installed; VAD will pass all audio through. "
                "Install with: pip install torch"
            )
            self.device = None
        else:
            self.device = torch.device(device)

        if not self.bypass and torch is not None:
            self._initialize_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Return speech probability (0.0–1.0) for a float32 mono chunk.

        Fast-paths:
        - Returns 0.0 immediately when the signal is below *_SILENCE_GATE*
          (hardware noise floor) without querying the model.
        - When ``bypass=True`` always returns 1.0 (all audio is treated as speech).
        """
        if self.bypass or self._model is None:
            return 1.0

        try:
            chunk = audio_chunk.astype(np.float32) if audio_chunk.dtype != np.float32 else audio_chunk

            # Bolt: np.abs(x).max() is ~2x faster than np.max(np.abs(x))
            peak = float(np.abs(chunk).max())

            if peak < _SILENCE_GATE:
                return 0.0

            audio_tensor = torch.from_numpy(chunk).to(self.device)

            with torch.no_grad():
                # Normalise only when above a useful threshold to avoid
                # amplifying near-silence to 1.0 which confuses the model.
                vad_input = audio_tensor / peak if peak > 0.01 else audio_tensor
                speech_prob: float = self._model(vad_input, self.samplerate).item()

            return speech_prob
        except Exception as exc:
            self.logger.warning(f"VAD error: {exc}")
            return 0.0

    def is_speech(self, audio_chunk: np.ndarray, threshold: float = 0.4) -> bool:
        """Return True when speech probability meets *threshold*."""
        return self.get_speech_prob(audio_chunk) >= threshold

    def reset_states(self) -> None:
        """Reset the model's internal recurrent state between sessions."""
        if self._model is not None:
            self._model.reset_states()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        try:
            self.logger.info(f"Loading Silero VAD (device={self.device})…")
            torch.set_num_threads(1)
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            try:
                self._model.to(self.device)
            except (RuntimeError, Exception) as exc:
                if self.device.type == "cuda":
                    self.logger.warning(f"CUDA move failed ({exc}), falling back to CPU.")
                    self.device = torch.device("cpu")
                    self._model.to(self.device)
                else:
                    raise
            self.logger.info(f"Silero VAD ready on {self.device}.")
        except Exception as exc:
            self.logger.error(f"Silero VAD load failed: {exc}")
            raise RuntimeError(f"Could not load Silero VAD: {exc}") from exc
