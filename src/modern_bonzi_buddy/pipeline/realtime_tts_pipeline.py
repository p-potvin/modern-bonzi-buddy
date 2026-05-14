from __future__ import annotations

from dataclasses import dataclass

from modern_bonzi_buddy.services.gemini_flash_tts import GeminiFlashTTS

SUPPORTED_LANGUAGES = ("english", "french")


@dataclass(slots=True)
class PipelineSettings:
    source_language: str
    target_language: str

    def validate(self) -> None:
        for value in (self.source_language, self.target_language):
            if value.lower() not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {value}")


class RealtimeTtsPipeline:
    def __init__(self, settings: PipelineSettings, tts: GeminiFlashTTS | None = None) -> None:
        self.settings = settings
        self.settings.validate()
        self.tts = tts or GeminiFlashTTS()
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def process_text(self, text: str) -> bytes:
        translated = self.tts.translate_text(
            text=text,
            source_language=self.settings.source_language,
            target_language=self.settings.target_language,
        )
        return self.tts.synthesize(translated, language=self.settings.target_language)
