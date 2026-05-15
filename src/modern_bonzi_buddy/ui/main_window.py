from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modern_bonzi_buddy.pipeline.realtime_tts_pipeline import PipelineSettings, RealtimeTtsPipeline


class _BenchSignals(QObject):
    result_ready = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modern Bonzi Buddy  ·  Realtime AST + TTS")
        self.resize(1040, 680)

        self.pipeline: RealtimeTtsPipeline | None = None
        self._bench_signals = _BenchSignals()
        self._bench_signals.result_ready.connect(self._on_bench_result)

        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # ── Header ────────────────────────────────────────────────────
        title = QLabel("Realtime AST + TTS")
        title.setObjectName("title")
        subtitle = QLabel(
            "System audio  →  Silero VAD  →  Translation  →  Gemini Flash TTS"
        )
        subtitle.setObjectName("subtitle")

        # ── Settings card ─────────────────────────────────────────────
        settings_card = QFrame()
        settings_card.setObjectName("card")
        settings_grid = QGridLayout(settings_card)
        settings_grid.setHorizontalSpacing(16)
        settings_grid.setVerticalSpacing(10)
        settings_grid.setContentsMargins(16, 14, 16, 14)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["English", "French"])

        self.target_combo = QComboBox()
        self.target_combo.addItems(["French", "English"])

        settings_grid.addWidget(_label("Source language"), 0, 0)
        settings_grid.addWidget(self.source_combo, 0, 1)
        settings_grid.addWidget(_label("Target language"), 1, 0)
        settings_grid.addWidget(self.target_combo, 1, 1)
        settings_grid.setColumnStretch(1, 1)

        # ── Controls row ──────────────────────────────────────────────
        self.start_btn = QPushButton("▶  Start Realtime")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.bench_btn = QPushButton("⏱  Run Benchmark")

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.bench_btn)

        # ── Splitter: log | benchmark ─────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline logs will appear here…")
        self.log.setObjectName("logPanel")

        bench_card = QFrame()
        bench_card.setObjectName("card")
        bench_v = QVBoxLayout(bench_card)
        bench_v.setContentsMargins(14, 12, 14, 12)
        bench_v.setSpacing(6)
        bench_header = _label("Benchmark Results")
        bench_header.setObjectName("sectionHeader")
        self.bench_out = QTextEdit()
        self.bench_out.setReadOnly(True)
        self.bench_out.setPlaceholderText("Press ⏱ Run Benchmark to profile each pipeline stage…")
        self.bench_out.setObjectName("benchPanel")
        bench_v.addWidget(bench_header)
        bench_v.addWidget(self.bench_out)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.log)
        splitter.addWidget(bench_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ── Assemble ──────────────────────────────────────────────────
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(settings_card)
        layout.addLayout(ctrl_row)
        layout.addWidget(splitter, stretch=1)

        # ── Signals ───────────────────────────────────────────────────
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.bench_btn.clicked.connect(self._run_benchmark)

        # ── Stylesheet ────────────────────────────────────────────────
        self.setStyleSheet(
            """
            QWidget { background: #0f1220; color: #e8ecff; font-size: 14px; font-family: Inter, Segoe UI, sans-serif; }
            QLabel#title { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
            QLabel#subtitle { color: #6b80c9; font-size: 13px; margin-bottom: 2px; }
            QLabel#sectionHeader { font-weight: 600; color: #94a1d3; font-size: 13px; }
            QFrame#card { background: #151929; border: 1px solid #202847; border-radius: 12px; }
            QComboBox {
                background: #0d1120;
                border: 1px solid #2a3460;
                border-radius: 7px;
                padding: 5px 10px;
                min-height: 28px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #151929; border: 1px solid #2a3460; }
            QTextEdit#logPanel, QTextEdit#benchPanel {
                background: #080d1a;
                border: 1px solid #1e2748;
                border-radius: 8px;
                padding: 8px;
                font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
                font-size: 12px;
                color: #9db4d8;
            }
            QPushButton {
                background: #2c4fd4;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: 600;
                min-height: 32px;
            }
            QPushButton:hover  { background: #3c6dff; }
            QPushButton:pressed { background: #1f3dae; }
            QPushButton:disabled { background: #1e2748; color: #4a5880; }
            QSplitter::handle { background: #1a2035; width: 1px; }
            """
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _start(self) -> None:
        settings = PipelineSettings(
            source_language=self.source_combo.currentText().lower(),
            target_language=self.target_combo.currentText().lower(),
        )
        self.pipeline = RealtimeTtsPipeline(settings=settings)
        self.pipeline.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log.append(
            "Pipeline started  [Silero VAD bypass=True + Gemini Flash TTS scaffold]"
        )

    def _stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.append("Pipeline stopped.")

    def _run_benchmark(self) -> None:
        self.bench_btn.setEnabled(False)
        self.bench_out.setPlainText("Running benchmark…")

        def _worker() -> None:
            try:
                from modern_bonzi_buddy.benchmarks.pipeline_benchmark import PipelineBenchmark

                report = PipelineBenchmark(iterations=5).run()
                self._bench_signals.result_ready.emit(report.summary())
            except Exception as exc:
                self._bench_signals.result_ready.emit(f"Benchmark error: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_bench_result(self, text: str) -> None:
        self.bench_out.setPlainText(text)
        self.bench_btn.setEnabled(True)
        self.log.append("Benchmark complete — see results panel →")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
    return lbl
