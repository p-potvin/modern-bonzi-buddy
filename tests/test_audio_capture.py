"""Tests for audio capture helpers, mute gate, and AGC."""
from __future__ import annotations

import queue

import numpy as np
import pytest

from modern_bonzi_buddy.audio.audio_capture import (
    PlaybackMuteGate,
    _AGC_TARGET,
    _SILENCE_GATE,
    _apply_agc,
    _drain_queue,
    _to_float32_mono,
)


# ──────────────────────────────────────────────────────────────────────────────
# PlaybackMuteGate
# ──────────────────────────────────────────────────────────────────────────────


def test_mute_gate_starts_closed():
    gate = PlaybackMuteGate()
    assert gate.is_active is False


def test_mute_gate_open_close():
    gate = PlaybackMuteGate()
    gate.open()
    assert gate.is_active is True
    gate.close()
    assert gate.is_active is False


# ──────────────────────────────────────────────────────────────────────────────
# _to_float32_mono
# ──────────────────────────────────────────────────────────────────────────────


def test_to_float32_mono_single_channel():
    stereo_like = np.ones((512, 1), dtype=np.float32) * 0.5
    result = _to_float32_mono(stereo_like)
    assert result.ndim == 1
    assert result.dtype == np.float32
    assert len(result) == 512
    np.testing.assert_allclose(result, 0.5)


def test_to_float32_mono_stereo_averages():
    data = np.zeros((512, 2), dtype=np.float32)
    data[:, 0] = 0.2
    data[:, 1] = 0.6
    result = _to_float32_mono(data)
    assert result.ndim == 1
    np.testing.assert_allclose(result, 0.4, atol=1e-6)


def test_to_float32_mono_1d_passthrough():
    data = np.ones(512, dtype=np.float32) * 0.3
    result = _to_float32_mono(data)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.3)


def test_to_float32_mono_converts_dtype():
    data = np.ones((512, 1), dtype=np.float64) * 0.7
    result = _to_float32_mono(data)
    assert result.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# _apply_agc
# ──────────────────────────────────────────────────────────────────────────────


def test_apply_agc_boosts_quiet_signal():
    quiet = np.ones(512, dtype=np.float32) * 0.01  # peak = 0.01
    result = _apply_agc(quiet)
    peak = float(np.abs(result).max())
    # Should be boosted toward _AGC_TARGET
    assert peak > 0.01


def test_apply_agc_does_not_boost_silence():
    silence = np.zeros(512, dtype=np.float32)
    result = _apply_agc(silence)
    assert float(np.abs(result).max()) == 0.0


def test_apply_agc_does_not_attenuate_loud_signal():
    loud = np.ones(512, dtype=np.float32) * _AGC_TARGET
    result = _apply_agc(loud)
    # At exactly target peak, gain==1 so array unchanged
    np.testing.assert_allclose(result, loud, atol=1e-5)


def test_apply_agc_below_silence_gate_passthrough():
    near_silence = np.ones(512, dtype=np.float32) * (_SILENCE_GATE * 0.5)
    result = _apply_agc(near_silence)
    # Below the gate — no modification
    np.testing.assert_allclose(result, near_silence)


# ──────────────────────────────────────────────────────────────────────────────
# _drain_queue
# ──────────────────────────────────────────────────────────────────────────────


def test_drain_queue_empties_queue():
    q: queue.Queue = queue.Queue()
    for i in range(5):
        q.put(i)
    _drain_queue(q)
    assert q.empty()


def test_drain_queue_noop_on_empty():
    q: queue.Queue = queue.Queue()
    _drain_queue(q)  # should not raise
    assert q.empty()
