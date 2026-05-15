"""Tests for audio playback helpers."""
from __future__ import annotations

import numpy as np
import pytest

from modern_bonzi_buddy.audio.playback import pcm_bytes_to_float32


def test_pcm_bytes_to_float32_converts_correctly():
    samples = np.array([0, 16383, -16384, 32767], dtype=np.int16)
    pcm = samples.tobytes()
    result = pcm_bytes_to_float32(pcm)
    assert result.dtype == np.float32
    assert len(result) == 4
    # 32767 / 32767 == 1.0, -16384 / 32767 ≈ -0.5
    assert result[2] < 0
    assert abs(result[3] - 1.0) < 0.001


def test_pcm_bytes_to_float32_zeros():
    samples = np.zeros(512, dtype=np.int16)
    result = pcm_bytes_to_float32(samples.tobytes())
    np.testing.assert_allclose(result, 0.0)


def test_pcm_bytes_to_float32_range():
    rng = np.random.default_rng(42)
    samples = rng.integers(-32768, 32767, size=1024, dtype=np.int16)
    result = pcm_bytes_to_float32(samples.tobytes())
    assert result.min() >= -1.0 - 1e-5
    assert result.max() <= 1.0 + 1e-5
