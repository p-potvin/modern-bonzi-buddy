"""Tests for the pipeline benchmark module."""
from __future__ import annotations

import pytest

from modern_bonzi_buddy.benchmarks.pipeline_benchmark import (
    BenchmarkReport,
    PipelineBenchmark,
    StageResult,
    _time_iterations,
)


def test_time_iterations_returns_positive_ms():
    def noop() -> None:
        pass

    result = _time_iterations(10, noop)
    assert result >= 0.0


def test_stage_result_str_ok():
    sr = StageResult(name="audio_buffer", elapsed_ms=1.234)
    out = str(sr)
    assert "audio_buffer" in out
    assert "OK" in out
    assert "1.234" in out


def test_stage_result_str_error():
    sr = StageResult(name="translate", elapsed_ms=0.0, ok=False, error="timeout")
    out = str(sr)
    assert "ERROR" in out
    assert "timeout" in out


def test_benchmark_report_total_ms_only_ok_stages():
    report = BenchmarkReport(
        stages=[
            StageResult("a", 10.0, ok=True),
            StageResult("b", 5.0, ok=False),
            StageResult("c", 3.0, ok=True),
        ]
    )
    assert report.total_ms == 13.0


def test_benchmark_report_summary_contains_stage_names():
    report = BenchmarkReport(
        stages=[
            StageResult("audio_buffer", 0.5),
            StageResult("vad_gate", 0.3),
        ]
    )
    summary = report.summary()
    assert "audio_buffer" in summary
    assert "vad_gate" in summary
    assert "Total" in summary


def test_pipeline_benchmark_run_mock_completes():
    bench = PipelineBenchmark(iterations=2, real_tts=False)
    report = bench.run()

    assert len(report.stages) == 4
    stage_names = [s.name for s in report.stages]
    assert "audio_buffer" in stage_names
    assert "translate" in stage_names
    assert "tts_synth" in stage_names


def test_pipeline_benchmark_mock_stages_all_ok():
    bench = PipelineBenchmark(iterations=1, real_tts=False)
    report = bench.run()
    failed = [s for s in report.stages if not s.ok]
    assert failed == [], f"Unexpected failures: {failed}"


def test_pipeline_benchmark_total_ms_positive():
    bench = PipelineBenchmark(iterations=2, real_tts=False)
    report = bench.run()
    assert report.total_ms > 0.0
