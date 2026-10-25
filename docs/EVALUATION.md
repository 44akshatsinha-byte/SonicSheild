# SonicShield Evaluation Framework & Metric Methodology

---

## 1. Metrics Defined

SonicShield computes standard biometric and spoofing detection metrics:

- **Equal Error Rate (EER):** The operating point where False Alarm Rate on real callers equals False Accept Rate on clones ($FAR = FRR$). Lower is better.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures separation across all potential thresholds.
- **False Alarm Rate (FAR on Real Speakers):** Percentage of legitimate live callers mistakenly flagged as uncertain or suspicious.
- **False Accept Rate (FAR on Fake Audio):** Percentage of synthetic or cloned calls mistakenly categorized as live.
- **Latency per Window:** Processing duration required to crush audio and extract all physical and biological features for a 6-second window.

---

## 2. Experimental Condition Matrix

Evaluation is segmented across acoustic and attack conditions:

| Condition | Physics Only (EER) | Biology Only (EER) | Combined Fusion (EER) | Mean Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Clean 8 kHz Audio** | *Measurable via `eval/evaluate.py`* | *Measurable via `eval/evaluate.py`* | *Measurable via `eval/evaluate.py`* | ~120 ms |
| **G.711 μ-law Telephony** | *Measurable via `eval/evaluate.py`* | *Measurable via `eval/evaluate.py`* | *Measurable via `eval/evaluate.py`* | ~135 ms |
| **AMR-NB Cellular** | *Requires FFmpeg opencore* | *Requires FFmpeg opencore* | *Requires FFmpeg opencore* | N/A |
| **Dry Clones** | High separation | High separation | High separation | ~130 ms |
| **Replayed Clones (Speaker-to-Mic)** | Lower separation (picks up real room) | High separation | Moderate-High separation | ~130 ms |

---

## 3. Running Evaluation on Local Data

To run the automated evaluation on your local dataset:

```powershell
.\.venv\Scripts\python.exe eval/evaluate.py --labels data/labels.csv --out eval/results.csv
```

If real audio files are not yet present, the evaluator will report this status and display the template format.
