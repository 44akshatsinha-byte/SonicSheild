# SonicShield Dataset Guide & Protocol

This guide outlines the data ingestion, labeling schema, recommended public corpora, and consent guidelines for evaluating SonicShield.

---

## 1. Directory Structure

```
data/
├── labels_template.csv     # Schema template for dataset metadata
├── labels.csv              # Active dataset index (when populated)
├── smoke/                  # Deterministic synthetic test signals (for CI/CD)
├── raw/
│   ├── real/               # Human speaker recordings
│   └── fake/               # Cloned / synthetic speech recordings
└── phone/
    ├── g711/               # Codec-crushed audio (G.711 mu-law)
    └── amr/                # Codec-crushed audio (AMR-NB)
```

---

## 2. Metadata Schema (`labels.csv`)

Every evaluated recording must be indexed in `data/labels.csv` with the following columns:

| Column | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `file` | string | e.g. `raw/real/spk01_01.wav` | Path relative to `data/` or absolute path |
| `label` | string | `real`, `fake` | Ground truth classification |
| `speaker` | string | e.g. `spk01`, `spk02` | Unique speaker identifier (used for splitting) |
| `source` | string | `consented_local`, `vctk`, `librispeech`, `f5_tts`, `wavefake` | Origin or generator used |
| `codec` | string | `clean`, `g711`, `amr` | Telephony compression applied |
| `room` | string | `office`, `living_room`, `bedroom`, `studio`, `none` | Environmental acoustic setting |
| `attack_type` | string | `none`, `dry_synthetic`, `replayed_speaker` | Type of spoofing attempt |

---

## 3. Recommended Data Sources

### A. Real Speech (Check Individual Licenses)
- **Consented Local Recordings:** 10+ volunteer teammates in domestic rooms (living room, office, hallway) recorded using mobile phones or headsets.
- **VCTK Corpus:** High-quality diverse speaker speech corpus.
- **LibriSpeech:** Multi-speaker clean reading dataset.

### B. AI-Generated & Cloned Speech
- **F5-TTS / ChatterBox / XTTS / Bark:** Generate matched voice clones from consented teammates' reference audio.
- **ASVspoof 2019 / 2021:** Standard spoofing benchmarks (the 2021 edition includes phone-quality codec conditions).
- **WaveFake:** Comprehensive multi-vocoder synthetic audio collection.

### C. Hard Case: Replay Attacks
- **Dry Clones:** Synthetic audio played directly from a file or virtual audio cable.
- **Replayed Clones:** Synthetic audio played through physical speakers into a room and re-recorded by a microphone (acquires natural room reverb, requiring the Biology Engine to detect).

---

## 4. Train / Val / Test Partitioning Rules

> [!IMPORTANT]
> **Zero Speaker Leakage:** Always partition data **by speaker ID**, never randomly by audio clip. If `spk01` is in the training set, `spk01` must never appear in the validation or test sets.

Recommended split:
- **70% Training**
- **15% Validation**
- **15% Test**
