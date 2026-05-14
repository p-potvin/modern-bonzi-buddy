from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import handled at runtime
    genai = None
    types = None

VOICE_BY_LANGUAGE = {
    "english": "Kore",
    "french": "Aoede",
}


def save_binary_file(file_name: str | os.PathLike[str], data: bytes) -> None:
    Path(file_name).write_bytes(data)


@dataclass(slots=True)
class GeminiFlashTTS:
    model: str = "gemini-2.0-flash-exp"
    api_key: str | None = None

    def _client(self):
        if genai is None:
            msg = "google-genai is required. Install with: pip install google-genai"
            raise RuntimeError(msg)

        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        return genai.Client(api_key=key)

    def synthesize(self, text: str, language: str) -> bytes:
        language_key = language.lower()
        if language_key not in VOICE_BY_LANGUAGE:
            raise ValueError(f"Unsupported language: {language}")

        client = self._client()

        if types is None:
            raise RuntimeError("google-genai types import failed")

        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=VOICE_BY_LANGUAGE[language_key]
                        )
                    )
                ),
            ),
        )

        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    return inline_data.data

        raise RuntimeError("No audio payload returned by Gemini Flash TTS")

    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        if source_language.lower() == target_language.lower():
            return text

        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=(
                f"Translate from {source_language} to {target_language}. "
                f"Return only translated text:\n\n{text}"
            ),
        )

        translated = getattr(response, "text", "") or ""
        translated = translated.strip()
        if not translated:
            raise RuntimeError("Gemini translation returned empty text")
        return translated
