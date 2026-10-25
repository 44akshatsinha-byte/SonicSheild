# SonicShield 3-Minute Hackathon Demo Script

**Theme:** Zero-Friction Acoustic Liveness Detection  
**Target Duration:** 3 Minutes (180 Seconds)  
**Hackathon Date:** October 31, 2026

---

## ⏱️ Timed Presentation Flow

### 0:00 – 0:20 | The Problem & Value Proposition
- **Speaker:** *"Judges, voice cloning has broken the telephone authentication barrier. Today, anyone can clone a family member's or executive's voice in 3 seconds. Call-center fraud costs billions, but asking customers to repeat passphrases creates massive friction.*  
*Meet **SonicShield**: zero-friction acoustic liveness detection. We don't ask the caller to do anything. Instead, our dual-engine architecture analyzes room acoustics and vocal biology directly over telephone lines."*

---

### 0:20 – 1:20 | Live Demo: Real Human vs. AI Clone
- **Action:** Open the SonicShield Dashboard at `http://localhost:8501`.
- **Action:** Select **Recording A (Human Reference)** from the Quick Demo menu or upload a consented teammate recording. Click **Analyze Caller Audio**.
- **Speaker points:**
  - Notice the **Liveness Gauge** reads **Green (80+)**.
  - Show the **Sub-Engine Meters**:
    - *Room & Echo (Physics):* Detects natural room reverberation decay tails and authentic ambient room tone in pauses.
    - *Breath & Voice (Biology):* Detects organic breath intervals, pitch intonation dynamics, and physiological jitter/shimmer.
  - Show the **Timeline Chart**: Demonstrates stability across 6-second rolling windows.
- **Action:** Now switch to **Recording B (Synthetic Clone)**.
- **Speaker points:**
  - The gauge drops into the **Red Zone (< 40)**.
  - Notice the explanations: *"Abrupt speech cutoff with minimal room decay tail"* and *"Unusually flat pitch track detected."*

---

### 1:20 – 1:50 | The Telephone Reality Check (G.711 μ-law)
- **Speaker:** *"Many audio models work in clean studio conditions but collapse over telephone lines. SonicShield includes a built-in **Codec Crusher**."*
- **Action:** In the sidebar, select **G.711 μ-law (ITU-T Standard Landline)**.
- **Action:** Click **Analyze** again.
- **Speaker points:**
  - Play back the audio widget to let the judges hear the genuine 8-bit companding distortion and bandpass filtering.
  - Show that SonicShield's features survive 8 kHz telephone compression because room reverberation and pitch dynamics persist across codecs.

---

### 1:50 – 2:20 | Agent Assist Workflow (Zero Confrontation)
- **Speaker:** *"SonicShield is built for human agents, not automated rejections. When a suspicious or uncertain call arrives..."*
- **Action:** Highlight the **Agent Assist Verification Prompt** box on the dashboard.
- **Speaker points:**
  - *"Instead of accusing the caller, SonicShield prompts the agent with a neutral question:*  
    **'Sorry, the connection crackled for a second. Could you repeat that and tell me the last thing you did today?'**
  - *This neutral friction causes spontaneous speech that synthetic engines struggle to generate in real-time, without alienating legitimate callers."*

---

### 2:20 – 3:00 | Honest Evaluation & Limitations
- **Speaker:** *"We believe in honest engineering. Acoustic features are not magic biometrics:*
  - *A real caller on a noise-canceling headset or in an anechoic closet will exhibit low room reverberation.*
  - *A replayed clone played through a loudspeaker into a bedroom picks up physical room reverb.*
  - *That is why we use two independent engines: Physics and Biology.*
  - *Our local test suite runs 36 automated unit tests in under 2 seconds, verifying the full end-to-end pipeline.*
  *SonicShield provides an immediate layer of defense for voice calls without sacrificing customer experience. Thank you!"*

---

## 📋 Backup Demo Checklist

In case of live venue hardware, network, or display issues:

1. **Pre-Cached Demo Files:** Smoke test WAV files are stored in `data/smoke/` and can be loaded instantly with zero upload latency.
2. **Offline Localhost:** Ensure `.venv` has all dependencies installed; the application requires zero external cloud connections.
3. **Backup Video Recording:** Keep a 1080p screen recording of the workflow saved locally on the laptop desktop.
