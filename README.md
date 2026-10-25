# 🛡️ SonicShield

**Zero-Friction Acoustic Liveness Detection**  
*Hackathon Prototype — Built for October 31, 2026*

---

## 📌 Overview

AI voice cloning allows fraudsters to impersonate individuals during telephone conversations with only seconds of reference audio. Traditional defenses introduce customer friction (e.g., security questions, biometrics, phrase repetition). 

**SonicShield** explores zero-friction acoustic liveness detection. By analyzing telephone-quality audio through two independent engines, SonicShield determines whether a caller's voice is a live human speaking in a physical room or an AI clone:

1. **🏛️ Physics Engine (Room & Echo):** Measures room reverberation decay tails ($RT_{60}$ proxy), ambient pause room tone floor, digital silence/gating, and short-delay cepstral micro-reflections.
2. **🫁 Biology Engine (Breath & Voice):** Analyzes conversational breath pause events, natural pitch intonation variation, involuntary vocal tremor (4–12 Hz), jitter, shimmer, and harmonics-to-noise ratio (HNR).
3. **🎯 Confidence Scorer:** Fuses both engines with Exponential Moving Average (EMA) smoothing across 6-second rolling windows (1-second hops) to output an intuitive **Liveness Score (0–100)** with human-readable explanations and an **Agent-Assist** prompt for uncertain calls.
4. **📻 Codec Crusher:** Simulates authentic telephone network conditions (Clean 8 kHz, ITU-T G.711 $\mu$-law, AMR-NB, packet loss, and line noise).

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+ (tested and verified on Python 3.14 on Windows 11).
- No external paid APIs or cloud keys required. 100% locally runnable offline.

### 2. Environment Setup

#### Windows (PowerShell)
```powershell
# Clone or open repository directory
cd "c:\Users\Akshat\Desktop\Sonic Sheild"

# Create virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install -r requirements.txt
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Launch the Streamlit Agent Dashboard
The primary dashboard contains audio upload, built-in benchmark demos, live gauge, sub-engine meters, timeline chart, and side-by-side comparison:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```
Open your browser to: **`http://localhost:8501`**

---

### Option B: Launch the FastAPI Backend
For REST integration and streaming:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation: **`http://127.0.0.1:8000/docs`**

#### Available Endpoints
- `GET /health` — Application status, version, and codec capabilities.
- `POST /analyze` — Upload WAV audio, choose codec and ablation mode, receive per-window JSON analysis.
- `WS /stream` — WebSocket streaming for real-time 1-second audio chunk processing.

---

## 🧪 Running Automated Tests

Run the full automated test suite (36 unit and integration tests):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

All tests execute in ~1.5 seconds covering audio ingestion, G.711 companding, rolling windows, physics/biology engines, scorer banding, and API endpoints.

---

## 📊 Generating Benchmark Signals & Running Evaluation

### Generate Benchmark Smoke Audio
```powershell
.\.venv\Scripts\python.exe scripts/generate_smoke_audio.py
```

### Run Evaluation Script
```powershell
.\.venv\Scripts\python.exe eval/evaluate.py --labels data/labels_template.csv
```

---

## 📁 Repository Structure

```
SonicShield/
├── README.md                   # Setup guide and documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── data/
│   ├── labels_template.csv     # Dataset metadata template
│   ├── smoke/                  # Deterministic synthetic test signals
│   └── raw/                    # Real and fake audio recordings
├── src/
│   ├── audio_utils.py          # WAV ingestion, validation, resampling
│   ├── windows.py              # 6s window / 1s hop segmenter
│   ├── codec_crusher.py        # Clean, G.711 mu-law, AMR-NB simulation
│   ├── physics_engine.py       # Room reverb, room tone, cepstral echo
│   ├── biology_engine.py       # Breaths, pitch dynamics, jitter/shimmer
│   ├── scorer.py               # EMA smoothing, risk bands, agent tips
│   └── pipeline.py             # Unified end-to-end processing pipeline
├── api/
│   └── main.py                 # FastAPI backend (/health, /analyze, /stream)
├── dashboard/
│   └── app.py                  # High-end Streamlit cybersecurity dashboard
├── eval/
│   ├── metrics.py              # ROC-AUC, EER, FAR, FRR calculation
│   └── evaluate.py             # Evaluation runner across codecs and modes
├── tests/                      # 36 automated unit & integration tests
├── scripts/
│   ├── check_environment.py    # Environment diagnostic check
│   └── generate_smoke_audio.py # Deterministic benchmark signal generator
└── docs/
    ├── ARCHITECTURE.md         # Detailed technical design
    ├── DATASET_GUIDE.md        # Data collection, consent, and splitting
    ├── DEMO_SCRIPT.md          # 3-minute timed hackathon presentation
    └── EVALUATION.md           # Metric definitions and benchmark matrix
```

---

## ⚠️ Honest Limitations & Ethical Disclosure

1. **Acoustic Overlap:** Acoustic indicators are imperfect heuristics. Genuine callers speaking through headsets, in heavily treated rooms, or with aggressive microphone noise suppression can exhibit low room reverberation.
2. **Replayed Clones:** Synthetic audio played through a loudspeaker into a physical room will capture genuine environmental reflections. SonicShield pairs the Physics Engine with the Biology Engine to mitigate this vulnerability.
3. **Prototype Calibration:** Displayed scores represent experimental heuristic confidence indicators (0–100), not formal posterior probabilities of authenticity.
4. **No Automated Decisions:** SonicShield is an **agent-assist tool**. It never terminates calls or blocks accounts. For calls in the amber or red bands, it provides neutral conversational prompts (e.g., asking the caller to clarify or describe recent activity) to stimulate spontaneous un-scripted speech.
