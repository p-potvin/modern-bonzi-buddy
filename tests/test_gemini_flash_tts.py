from modern_bonzi_buddy.services.gemini_flash_tts import VOICE_BY_LANGUAGE, save_binary_file


def test_save_binary_file(tmp_path):
    target = tmp_path / "audio.raw"
    data = b"abc123"

    save_binary_file(target, data)

    assert target.read_bytes() == data


def test_voice_mapping_supports_target_languages():
    assert "english" in VOICE_BY_LANGUAGE
    assert "french" in VOICE_BY_LANGUAGE
