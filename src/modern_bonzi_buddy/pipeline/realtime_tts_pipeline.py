"""
Realtime AST+TTS pipeline orchestrator.

Flow
----
AudioCapture → SileroVAD gate → GeminiFlashTTS translate + synthesize → AudioPlayback

The pipeline runs the audio processing loop in a background daemon thread.
Callers interact via :meth:`start` / :meth:`stop` and the ``on_audio_translated``
callback hook.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from modern_bonzi_buddy.audio.audio_capture import PlaybackMuteGate
from modern_bonzi_buddy.audio.audio_router import AudioRouter
from modern_bonzi_buddy.audio.silero_vad_gate import SileroVadGate
from modern_bonzi_buddy.services.gemini_flash_tts import GeminiFlashTTS

SUPPORTED_LANGUAGES = ("english", "french")

logger = logging.getLogger("modern_bonzi_buddy.pipeline")


@dataclass(slots=True)
class PipelineSettings:
    source_language: str
    target_language: str

    # VAD parameters
    vad_threshold: float = 0.15
    max_buffer_chunks: int = 75
    min_speech_chunks: int = 5
    silence_chunks: int = 8

    def validate(self) -> None:
        for value in (self.source_language, self.target_language):
            if value.lower() not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {value}")


class RealtimeTtsPipeline:
    """
    Realtime audio → VAD gate → translate → TTS → playback pipeline.

    Parameters
    ----------
    settings:
        Language and VAD tuning options.
    tts:
        TTS/translation service.  Defaults to :class:`GeminiFlashTTS`.
    audio_router:
        Audio capture facade.  Defaults to a freshly created :class:`AudioRouter`.
    vad:
        VAD gate.  Defaults to :class:`SileroVadGate` in bypass mode (no model
        load required for the scaffold) so CI/tests work without torch.hub.
    on_audio_translated:
        Optional callback ``(translated_text: str, pcm_bytes: bytes) -> None``
        called for each completed TTS chunk.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        tts: GeminiFlashTTS | None = None,
        audio_router: AudioRouter | None = None,
        vad: SileroVadGate | None = None,
        on_audio_translated: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self.settings = settings
        self.settings.validate()
        self.tts = tts or GeminiFlashTTS()
        self.router = audio_router or AudioRouter()
        self.vad = vad or SileroVadGate(bypass=True)
        self.on_audio_translated = on_audio_translated
        self.running = False

        self._speech_buffer: list[np.ndarray] = []
        self._tts_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._capture_thread: threading.Thread | None = None
        self._tts_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.router.start()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="bonzi-capture"
        )
        self._capture_thread.start()

        self._tts_thread = threading.Thread(
            target=self._tts_loop, daemon=True, name="bonzi-tts"
        )
        self._tts_thread.start()

        logger.info("Pipeline started.")

    def stop(self) -> None:
        self.running = False
        self.router.stop()
        self._tts_queue.put(None)  # sentinel to unblock the TTS loop
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._tts_thread:
            self._tts_thread.join(timeout=2.0)
        logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    # Single-shot text processing (no audio capture)
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> bytes:
        """Translate *text* and return synthesised PCM bytes."""
        translated = self.tts.translate_text(
            text=text,
            source_language=self.settings.source_language,
            target_language=self.settings.target_language,
        )
        return self.tts.synthesize(translated, language=self.settings.target_language)

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Pull audio chunks from the router, run VAD, and queue speech segments."""
        silence_counter = 0

        while self.running:
            chunk = self.router.capture.get_chunk(timeout=0.5)
            if chunk is None:
                continue

            speech_prob = self.vad.get_speech_prob(chunk)
            in_speech = bool(self._speech_buffer)

            if speech_prob >= self.settings.vad_threshold:
                self._speech_buffer.append(chunk)
                silence_counter = 0

                if len(self._speech_buffer) >= self.settings.max_buffer_chunks:
                    self._flush_buffer()
            else:
                if in_speech:
                    self._speech_buffer.append(chunk)
                silence_counter += 1

                if silence_counter >= self.settings.silence_chunks:
                    if len(self._speech_buffer) >= self.settings.min_speech_chunks:
                        self._flush_buffer()
                    else:
                        logger.debug(
                            f"Discarding short buffer ({len(self._speech_buffer)} chunks)"
                        )
                        self._speech_buffer.clear()
                    silence_counter = 0

    def _flush_buffer(self) -> None:
        if not self._speech_buffer:
            return
        audio = np.concatenate(self._speech_buffer)
        # Keep 1-chunk overlap for context continuity at segment boundaries
        self._speech_buffer = [self._speech_buffer[-1]]
        self._tts_queue.put(audio)

    def _tts_loop(self) -> None:
        """Pull audio segments, translate+synthesise, and fire the callback."""
        while self.running:
            try:
                audio_payload = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio_payload is None:
                break

            try:
                # Use the audio numpy array for a future STT → translate stage.
                # For the scaffold we translate a placeholder; the intent is
                # that an STT model (Whisper/Parakeet) sits between capture and TTS.
                placeholder_text = f"[audio:{len(audio_payload)} samples]"
                pcm = self.process_text(placeholder_text)
                if self.on_audio_translated:
                    self.on_audio_translated(placeholder_text, pcm)
            except Exception as exc:
                logger.error(f"TTS loop error: {exc}")
