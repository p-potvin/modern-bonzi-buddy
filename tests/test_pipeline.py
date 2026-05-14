import pytest

from modern_bonzi_buddy.pipeline.realtime_tts_pipeline import PipelineSettings, RealtimeTtsPipeline


def test_pipeline_settings_validate_language_support():
    settings = PipelineSettings(source_language="english", target_language="french")

    settings.validate()


def test_pipeline_settings_reject_unknown_language():
    settings = PipelineSettings(source_language="spanish", target_language="french")

    with pytest.raises(ValueError, match="Unsupported language"):
        settings.validate()


def test_pipeline_start_stop_changes_state():
    pipeline = RealtimeTtsPipeline(PipelineSettings(source_language="english", target_language="french"))

    pipeline.start()
    assert pipeline.running is True

    pipeline.stop()
    assert pipeline.running is False


def test_pipeline_process_text_uses_translation_then_tts():
    class FakeTTS:
        def __init__(self):
            self.calls = []

        def translate_text(self, text: str, source_language: str, target_language: str) -> str:
            self.calls.append(("translate", text, source_language, target_language))
            return "bonjour"

        def synthesize(self, text: str, language: str) -> bytes:
            self.calls.append(("synthesize", text, language))
            return b"audio"

    fake = FakeTTS()
    pipeline = RealtimeTtsPipeline(
        PipelineSettings(source_language="english", target_language="french"),
        tts=fake,
    )

    output = pipeline.process_text("hello")

    assert output == b"audio"
    assert fake.calls == [
        ("translate", "hello", "english", "french"),
        ("synthesize", "bonjour", "french"),
    ]
