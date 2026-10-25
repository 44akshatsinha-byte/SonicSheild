# 🛡️ SonicShield

**Zero-Friction Acoustic Liveness Detection**  
*Hackathon Prototype — Upgraded with Supervised Machine Learning*

---

## 📌 Overview

AI voice cloning allows fraudsters to impersonate individuals during telephone conversations with only seconds of reference audio. Traditional defenses introduce customer friction (e.g., security questions, biometrics, phrase repetition). 

**SonicShield** explores zero-friction acoustic liveness detection. By analyzing telephone-quality audio through two independent engines and a supervised machine learning classifier, SonicShield determines whether a caller's voice is a live human speaking in a physical room or an AI clone:

1. **🏛️ Physics Engine (Room & Echo):** Measures room reverberation decay tails ($RT_{60}$ proxy), ambient pause room tone floor, digital silence/gating, and short-delay cepstral micro-reflections.
2. **🫁 Biology Engine (Breath & Voice):** Analyzes conversational breath pause events, natural pitch intonation variation, involuntary vocal tremor (4–12 Hz), jitter, shimmer, and harmonics-to-noise ratio (HNR).
3. **🤖 Supervised Machine Learning Classifier:** Consumes a fixed 18-dimensional feature vector combining room reverberation, pause noise floor, and vocal perturbation. Implements scikit-learn Logistic Regression and Gradient Boosting pipelines with median imputation and feature standardization, trained strictly with zero-leakage speaker-separated splits.
4. **🎯 Confidence Scorer & Modes:** 
   - **`baseline`**: Transparent rule-based heuristic scoring (0–100) with Exponential Moving Average (EMA) smoothing across rolling windows.
   - **`ml`**: Direct model-derived probability scaled to an experimental 0–100 liveness score.
   - **`compare`**: Displays both rule-based and ML outputs side-by-side on the dashboard and API.
5. **📻 Codec Crusher:** Simulates authentic telephone network conditions (Clean 8 kHz, ITU-T G.711 $\mu$-law, AMR-NB, packet loss, and line noise).

---

## 📐 Machine Learning Architecture & Feature Schema

The ML classifier relies on a strictly ordered, 18-dimensional numerical feature vector extracted per 6-second rolling window (`src/ml_features.py`):

| # | Feature Name | Source | Description |
| :--- | :--- | :--- | :--- |
| 1 | `phys_score` | Physics Engine | Heuristic room & echo score [0.0 - 1.0] |
| 2 | `phys_rms_db` | Physics Engine | Active window RMS energy level in dBFS |
| 3 | `phys_decay_slope` | Physics Engine | Speech offset energy decay slope in dB/ms ($RT_{60}$ proxy) |
| 4 | `phys_reverb_score` | Physics Engine | Natural room decay tail quality score [0.0 - 1.0] |
| 5 | `phys_pause_rms_db` | Physics Engine | Background room tone energy floor in pauses (dBFS) |
| 6 | `phys_spectral_flatness` | Physics Engine | Wiener entropy of ambient pause spectrum |
| 7 | `phys_is_digital_silence` | Physics Engine | Binary flag for artificial digital silence / noise gate |
| 8 | `phys_cepstral_peak_ratio` | Physics Engine | Early room reflection micro-echo peak ratio |
| 9 | `bio_score` | Biology Engine | Heuristic breath & voice biology score [0.0 - 1.0] |
| 10 | `bio_breath_count` | Biology Engine | Count of detected sub-speech breath bursts in window |
| 11 | `bio_breaths_per_min` | Biology Engine | Extrapolated breathing cadence (breaths/minute) |
| 12 | `bio_pitch_mean_hz` | Biology Engine | Mean fundamental frequency ($F_0$) across voiced frames |
| 13 | `bio_pitch_std_hz` | Biology Engine | Fundamental frequency standard deviation (prosody intonation) |
| 14 | `bio_tremor_ratio` | Biology Engine | Involuntary physiological vocal tremor (4–12 Hz) |
| 15 | `bio_jitter_local_percent` | Biology Engine | Local cycle-to-cycle frequency perturbation (%) via Parselmouth |
| 16 | `bio_shimmer_local_percent` | Biology Engine | Local cycle-to-cycle amplitude perturbation (%) via Parselmouth |
| 17 | `bio_hnr_db` | Biology Engine | Harmonics-to-Noise Ratio (glottal turbulence) in dB |
| 18 | `bio_voiced_fraction` | Biology Engine | Fraction of frames containing voiced speech phonation |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+ (verified on Python 3.14 on Windows 11).
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
The dashboard features an engine selector (`Baseline`, `ML Classifier`, `Compare Both`), live gauge, sub-engine meters, timeline chart, and side-by-side comparison:

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
- `GET /health` — Application status, codec capabilities, and ML model status (`loaded`, `model_type`).
- `POST /analyze` — Upload WAV audio, choose codec, engine ablation (`both`, `physics`, `biology`), and detection mode (`baseline`, `ml`, `compare`).
- `WS /stream` — WebSocket streaming for real-time 1-second audio chunk processing with baseline and ML updates.

---

## 🧠 Training the Supervised ML Model

To train the classifier on your labeled audio recordings:

1. Place real and cloned audio files in `data/raw/real/` and `data/raw/fake/`.
2. Index them in `data/labels.csv` following the schema:
   `file,label,speaker,source,codec,room,attack_type`
3. Run the training script:
```powershell
.\.venv\Scripts\python.exe scripts/train_ml_model.py --manifest data/labels.csv
```

### Zero-Leakage Splitting Guarantee
The training script groups recordings strictly by **Speaker ID** (70% train, 15% validation, 15% test). No speaker or overlapping audio window is ever shared between training and test sets.

The fitted pipeline is saved to `models/sonicshield_ml.joblib` and metadata to `models/sonicshield_ml_meta.json`.

---

## 🧪 Running Automated Tests

Run the full automated test suite (47 unit and integration tests):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

All 47 tests execute in ~1.8 seconds covering:
- Audio ingestion, validation, and resampling.
- G.711 $\mu$-law 8-bit companding round trip.
- 6s rolling windows with 1s hops.
- Physics and Biology engines.
- Fixed 18-feature schema consistency and NaN/Inf sanitization.
- ML classifier inference, model loading, and missing model fallback.
- Speaker-separated split integrity (zero speaker leakage).
- FastAPI `/health` and `/analyze` endpoints with detection modes.

---

## 📊 Generating Benchmark Signals & Running Evaluation

### Generate Benchmark Smoke Audio
```powershell
.\.venv\Scripts\python.exe scripts/generate_smoke_audio.py
```

### Run Model & Baseline Evaluation
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
├── models/
│   ├── sonicshield_ml.joblib   # Saved fitted scikit-learn pipeline (when trained)
│   └── sonicshield_ml_meta.json # Model metadata, schema, and metrics
├── src/
│   ├── audio_utils.py          # WAV ingestion, validation, resampling
│   ├── windows.py              # 6s window / 1s hop segmenter
│   ├── codec_crusher.py        # Clean, G.711 mu-law, AMR-NB simulation
│   ├── physics_engine.py       # Room reverb, room tone, cepstral echo
│   ├── biology_engine.py       # Breaths, pitch dynamics, jitter/shimmer
│   ├── ml_features.py          # 18-dimensional ML feature extractor
│   ├── ml_classifier.py        # ML model loader and inference engine
│   ├── scorer.py               # EMA smoothing, risk bands, agent tips
│   └── pipeline.py             # Unified end-to-end processing pipeline
├── api/
│   └── main.py                 # FastAPI backend (/health, /analyze, /stream)
├── dashboard/
│   └── app.py                  # High-end Streamlit cybersecurity dashboard
├── eval/
│   ├── metrics.py              # ROC-AUC, EER, FAR, FRR calculation
│   └── evaluate.py             # Evaluation runner across codecs and models
├── tests/                      # 47 automated unit & integration tests
├── scripts/
│   ├── check_environment.py    # Environment diagnostic check
│   ├── generate_smoke_audio.py # Deterministic benchmark signal generator
│   └── train_ml_model.py       # Supervised ML model training script
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
3. **Prototype Calibration:** Displayed scores represent experimental confidence indicators (0–100), not formal posterior probabilities of authenticity.
4. **No Automated Decisions:** SonicShield is an **agent-assist tool**. It never terminates calls or blocks accounts. For calls in the amber or red bands, it provides neutral conversational prompts to stimulate spontaneous un-scripted speech.
