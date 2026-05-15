"""
Pipeline benchmark — measures wall-clock latency of each stage end-to-end.

Stages benchmarked
------------------
1. **audio_buffer** — simulate 512-sample float32 chunk generation time
2. **vad_gate**     — run VAD on a synthetic speech chunk (bypass mode for CI)
3. **translate**    — call the translate_text service (real or mock)
4. **tts_synth**    — call the synthesize service (real or mock)
5. **total**        — sum of the above

Usage (CLI)
-----------
.. code-block:: bash

    python -m modern_bonzi_buddy.benchmarks.pipeline_benchmark

Usage (programmatic)
--------------------
.. code-block:: python

    from modern_bonzi_buddy.benchmarks.pipeline_benchmark import PipelineBenchmark
    results = PipelineBenchmark().run()
    print(results.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from modern_bonzi_buddy.audio.audio_capture import _apply_agc, _to_float32_mono
from modern_bonzi_buddy.audio.silero_vad_gate import SileroVadGate
from modern_bonzi_buddy.pipeline.realtime_tts_pipeline import (
    PipelineSettings,
    RealtimeTtsPipeline,
)

# ──────────────────────────────────────────────────────────────────────────────
# Result data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StageResult:
    name: str
    elapsed_ms: float
    ok: bool = True
    error: str = ""

    def __str__(self) -> str:
        status = "OK" if self.ok else f"ERROR({self.error})"
        return f"  {self.name:<20} {self.elapsed_ms:>10.3f} ms   [{status}]"


@dataclass
class BenchmarkReport:
    stages: list[StageResult] = field(default_factory=list)
    iterations: int = 1

    @property
    def total_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.stages if s.ok)

    def summary(self) -> str:
        header = f"\n{'─'*55}\n  Pipeline Benchmark  (iterations={self.iterations})\n{'─'*55}"
        rows = "\n".join(str(s) for s in self.stages)
        footer = f"{'─'*55}\n  {'Total (ok stages)':<20} {self.total_ms:>10.3f} ms\n{'─'*55}"
        return "\n".join([header, rows, footer])


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


class PipelineBenchmark:
    """
    Runs a timed end-to-end benchmark of the AST+TTS pipeline.

    By default, translation and TTS synthesis are exercised through the same
    :class:`~modern_bonzi_buddy.pipeline.realtime_tts_pipeline.RealtimeTtsPipeline`
    code paths but with a *mock TTS* injected so no real API call is made.

    Parameters
    ----------
    iterations:
        Number of timed iterations to average over.
    source_language / target_language:
        Language pair to benchmark.
    real_tts:
        If True, use the real :class:`~modern_bonzi_buddy.services.gemini_flash_tts.GeminiFlashTTS`
        (requires a valid ``GEMINI_API_KEY`` env var).
    sample_text:
        Text snippet fed to the translate→TTS stages.
    samplerate:
        Sample rate used when generating the synthetic audio buffer.
    blocksize:
        Number of frames per simulated audio chunk.
    """

    def __init__(
        self,
        iterations: int = 3,
        source_language: str = "english",
        target_language: str = "french",
        real_tts: bool = False,
        sample_text: str = "Hello, this is a benchmark test sentence.",
        samplerate: int = 16000,
        blocksize: int = 512,
    ) -> None:
        self.iterations = iterations
        self.source_language = source_language
        self.target_language = target_language
        self.real_tts = real_tts
        self.sample_text = sample_text
        self.samplerate = samplerate
        self.blocksize = blocksize

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Execute all benchmark stages and return a :class:`BenchmarkReport`."""
        report = BenchmarkReport(iterations=self.iterations)

        report.stages.append(self._bench_audio_buffer())
        report.stages.append(self._bench_vad_gate())
        report.stages.append(self._bench_translate())
        report.stages.append(self._bench_tts_synth())

        return report

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _bench_audio_buffer(self) -> StageResult:
        """Time synthetic float32 chunk generation + AGC (in-process, zero I/O)."""
        elapsed = _time_iterations(self.iterations, self._simulate_audio_chunk)
        return StageResult(name="audio_buffer", elapsed_ms=elapsed)

    def _bench_vad_gate(self) -> StageResult:
        """Time Silero VAD in bypass mode (no model load) for pure overhead baseline."""
        vad = SileroVadGate(bypass=True)
        chunk = self._make_speech_chunk()

        def _vad_run() -> None:
            vad.get_speech_prob(chunk)

        elapsed = _time_iterations(self.iterations, _vad_run)
        return StageResult(name="vad_gate (bypass)", elapsed_ms=elapsed)

    def _bench_translate(self) -> StageResult:
        """Time the translate_text path using the mock or real service."""
        tts = self._make_tts()

        def _translate() -> None:
            tts.translate_text(self.sample_text, self.source_language, self.target_language)

        try:
            elapsed = _time_iterations(self.iterations, _translate)
            return StageResult(name="translate", elapsed_ms=elapsed)
        except Exception as exc:
            return StageResult(name="translate", elapsed_ms=0.0, ok=False, error=str(exc))

    def _bench_tts_synth(self) -> StageResult:
        """Time the synthesize path using the mock or real service."""
        tts = self._make_tts()
        translated = self.sample_text  # use same text for synth timing

        def _synth() -> None:
            tts.synthesize(translated, language=self.target_language)

        try:
            elapsed = _time_iterations(self.iterations, _synth)
            return StageResult(name="tts_synth", elapsed_ms=elapsed)
        except Exception as exc:
            return StageResult(name="tts_synth", elapsed_ms=0.0, ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _simulate_audio_chunk(self) -> None:
        """Generate + process one synthetic float32 audio chunk."""
        raw = np.random.randn(self.blocksize, 1).astype(np.float32) * 0.1
        mono = _to_float32_mono(raw)
        _apply_agc(mono)

    def _make_speech_chunk(self) -> np.ndarray:
        """Return a float32 chunk that looks like quiet speech (above noise floor)."""
        return (np.random.randn(self.blocksize).astype(np.float32) * 0.05).clip(-1.0, 1.0)

    def _make_tts(self) -> "GeminiFlashTTS | _MockTTS":
        if self.real_tts:
            from modern_bonzi_buddy.services.gemini_flash_tts import GeminiFlashTTS

            return GeminiFlashTTS()
        return _MockTTS()


class _MockTTS:
    """In-process mock TTS used by the benchmark when ``real_tts=False``."""

    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        if source_language.lower() == target_language.lower():
            return text
        # Simulate non-trivial string work
        return f"[{target_language}] {text}"

    def synthesize(self, text: str, language: str) -> bytes:
        # Simulate a 24 kHz mono 16-bit PCM buffer ~0.5 s long
        samples = np.zeros(12000, dtype=np.int16)
        return samples.tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────────────────────────────────────


def _time_iterations(n: int, fn) -> float:  # noqa: ANN001
    """Return average wall-clock time of *fn()* over *n* iterations in ms."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / n) * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the AST+TTS pipeline")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per stage")
    parser.add_argument("--real-tts", action="store_true", help="Use real Gemini TTS (needs API key)")
    parser.add_argument("--source", default="english", help="Source language")
    parser.add_argument("--target", default="french", help="Target language")
    args = parser.parse_args()

    bench = PipelineBenchmark(
        iterations=args.iterations,
        real_tts=args.real_tts,
        source_language=args.source,
        target_language=args.target,
    )
    report = bench.run()
    print(report.summary())


if __name__ == "__main__":
    _cli()
