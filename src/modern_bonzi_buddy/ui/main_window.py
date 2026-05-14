from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modern_bonzi_buddy.pipeline.realtime_tts_pipeline import PipelineSettings, RealtimeTtsPipeline


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modern Bonzi Buddy - Realtime AST + TTS")
        self.resize(980, 620)

        self.pipeline: RealtimeTtsPipeline | None = None

        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Realtime AST + TTS")
        title.setObjectName("title")
        subtitle = QLabel("Machine Audio Feed -> Silero VAD -> Translation -> Gemini Flash TTS")
        subtitle.setObjectName("subtitle")

        card = QFrame()
        card.setObjectName("card")
        card_layout = QGridLayout(card)
        card_layout.setHorizontalSpacing(12)
        card_layout.setVerticalSpacing(12)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["English", "French"])

        self.target_combo = QComboBox()
        self.target_combo.addItems(["French", "English"])

        self.start_btn = QPushButton("Start Realtime")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline logs will appear here...")

        card_layout.addWidget(QLabel("Source language"), 0, 0)
        card_layout.addWidget(self.source_combo, 0, 1)
        card_layout.addWidget(QLabel("Target language"), 1, 0)
        card_layout.addWidget(self.target_combo, 1, 1)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(card)
        layout.addLayout(button_row)
        layout.addWidget(self.log, stretch=1)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        self.setStyleSheet(
            """
            QWidget { background: #0f1220; color: #e8ecff; font-size: 14px; }
            QLabel#title { font-size: 28px; font-weight: 700; }
            QLabel#subtitle { color: #94a1d3; margin-bottom: 4px; }
            QFrame#card { background: #171c30; border: 1px solid #252d4e; border-radius: 12px; }
            QComboBox, QTextEdit {
                background: #0d1120;
                border: 1px solid #2d3760;
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton {
                background: #3c6dff;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #2a3767;
                color: #8fa1dd;
            }
            """
        )

    def _start(self) -> None:
        settings = PipelineSettings(
            source_language=self.source_combo.currentText().lower(),
            target_language=self.target_combo.currentText().lower(),
        )
        self.pipeline = RealtimeTtsPipeline(settings=settings)
        self.pipeline.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log.append("Pipeline started with Silero VAD + Gemini Flash TTS scaffold.")

    def _stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.append("Pipeline stopped.")
