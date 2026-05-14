# modern-bonzi-buddy

Realtime AST + TTS desktop scaffold using **PySide6** and **Gemini Flash TTS**.

## Features in this scaffold
- Desktop GUI built from scratch with PySide6 (English/French source + target language flow)
- Gemini Flash TTS service wrapper with binary audio saving helper
- Realtime pipeline skeleton for machine audio capture -> VAD gate -> translation/TTS output
- Silero VAD integration point (lazy-loaded)
- CUDA-ready dependency stack (`torch` + `torchaudio`) for GPU acceleration workflows

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For NVIDIA CUDA wheels (example for RTX/GTX cards), install PyTorch CUDA builds after base install:
```bash
pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-key"
```

## Run
```bash
python -m modern_bonzi_buddy.main
```
