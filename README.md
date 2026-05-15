# modern-bonzi-buddy

Realtime AST + TTS desktop app using **PySide6** and **Gemini Flash TTS**.

## Features
- Desktop GUI built from scratch with PySide6 (English ↔ French)
- Gemini Flash TTS service: binary audio save helper + translation + synthesis
- `soundcard` WASAPI/PulseAudio loopback capture — grabs system audio without virtual cables
- **TTS playback mute gate** — prevents the capture pipeline from re-ingesting our own output
- All audio uses **float32 numpy buffers** in-memory; no `.wav` files written on the hot path
- Silero VAD gate with silence floor, software AGC, and CUDA/CPU fallback
- `sounddevice.play(float32_array)` for zero-copy TTS playback (no file encoding overhead)
- Pipeline benchmark CLI that profiles each stage (buffer → VAD → translate → TTS) in ms
- CUDA-ready stack (`torch + torchaudio`) for GTX 3060 / RTX GPU workflows

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For NVIDIA CUDA wheels (GTX 3060 / RTX series):
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

## Benchmark
```bash
# Dry-run (mock TTS — no API key needed)
modern-bonzi-buddy-bench --iterations 10

# Real Gemini API calls
modern-bonzi-buddy-bench --iterations 5 --real-tts

# Example output:
# ───────────────────────────────────────────────────────
#   Pipeline Benchmark  (iterations=10)
# ───────────────────────────────────────────────────────
#   audio_buffer              0.002 ms   [OK]
#   vad_gate (bypass)         0.018 ms   [OK]
#   translate               450.123 ms   [OK]
#   tts_synth               312.456 ms   [OK]
# ───────────────────────────────────────────────────────
#   Total (ok stages)       762.599 ms
# ───────────────────────────────────────────────────────
```

## Audio capture design

```
System speakers (loopback)
      │
      ▼
 AudioCapture (soundcard, float32, 512-frame blocks)
      │
      │◄── PlaybackMuteGate (open while OUR TTS is playing → chunks discarded)
      │
      ▼
 SileroVadGate (~1 ms overhead, CUDA or CPU)
      │ speech detected
      ▼
 GeminiFlashTTS.translate_text()
      │
      ▼
 GeminiFlashTTS.synthesize()  →  PCM bytes
      │
      ▼
 sounddevice.play(float32_array)  ← no .wav file, direct numpy playback
      │
      └──► PlaybackMuteGate.open() … close()  (suppress feedback window)
```

**Why no `.wav` files?**
Encoding/decoding `.wav` adds ~5–15 ms of latency and O(N) disk I/O on every
chunk. We pass `float32` numpy arrays through in-memory `queue.Queue` buffers,
matching the pattern in `realtime-stt`'s `AudioRecorder`.

**Why `soundcard` over `sounddevice` for capture?**
`soundcard` wraps WASAPI (Windows) and PulseAudio/ALSA (Linux) with first-class
loopback support. `sounddevice` does not expose loopback devices on Windows
without a virtual cable. We still use `sounddevice` for *playback* because its
`sd.play(float32_array)` API is simpler and more portable than `soundcard`'s
playback path for our use case.

