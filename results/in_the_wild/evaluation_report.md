# 🛡️ SonicShield In-The-Wild Dataset Evaluation Report

**Model:** Pretrained AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)  
**Dataset:** In-The-Wild Audio Deepfake Dataset (Müller et al.)  
**Checkpoint:** `models/pretrained/AASIST.pth` (Official Clova AI weights, evaluated without retraining or fine-tuning)  
**Evaluation Scope:** Balanced 20-sample benchmark (10 genuine human recordings, 10 synthetic voice clones)  

---

## 📊 Summary of Evaluation Metrics

| Metric | Measured Value | Description / Formula |
| :--- | :---: | :--- |
| **Total Recordings** | **20** | 10 Genuine (`bona-fide`), 10 Fake (`spoof`) |
| **Accuracy** | **55.0%** | Overall classification accuracy: $\frac{\text{TP} + \text{TN}}{N}$ |
| **Genuine Precision** | **0.538** | Precision on bona-fide speech: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| **Genuine Recall** | **0.700** | Recall on bona-fide speech: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |
| **Genuine F1-Score** | **0.609** | Harmonic mean of genuine precision & recall |
| **Fake Precision** | **0.571** | Precision on spoof/clone speech: $\frac{\text{TN}}{\text{TN} + \text{FN}}$ |
| **Fake Recall** | **0.400** | Recall on spoof/clone speech: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ |
| **Fake F1-Score** | **0.471** | Harmonic mean of spoof precision & recall |
| **Macro F1-Score** | **0.540** | Unweighted average F1 across both classes |
| **ROC-AUC** | **0.45** | Area Under Receiver Operating Characteristic Curve |
| **Equal Error Rate (EER)** | **0.5** | Point where False Acceptance Rate equals False Alarm Rate |
| **EER Optimal Threshold** | **0.756** | Continuous bona-fide score threshold corresponding to EER |

---

## 🗂️ Confusion Matrix Breakdown

| Actual \ Predicted | Predicted Spoof (Fake, 0) | Predicted Bona-fide (Live, 1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Spoof (Fake, 0)** | **4** (True Negative) | **6** (False Positive) | **10** |
| **Actual Bona-fide (Live, 1)** | **3** (False Negative) | **7** (True Positive) | **10** |
| **Total Predicted** | **7** | **13** | **20** |

* **True Negatives (4/10):** Synthetic deepfakes correctly rejected as spoof.
* **False Positives (6/10):** Synthetic deepfakes erroneously accepted as genuine human speech.
* **False Negatives (3/10):** Genuine human speakers erroneously flagged as spoof.
* **True Positives (7/10):** Genuine human speakers correctly identified as bona-fide.

A rendered visualization has been saved to [`confusion_matrix.png`](confusion_matrix.png).

---

## 🔍 Per-Recording Detailed Predictions

| # | File | Speaker | Ground Truth | AASIST Prediction | Bona-fide Score | Liveness Index | Result |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `3491.wav` | Adam Driver | `bona-fide` | `bona_fide` | 0.9381 | 94/100 | ✅ Correct |
| 2 | `21886.wav` | Donald Trump | `bona-fide` | `bona_fide` | 0.8254 | 83/100 | ✅ Correct |
| 3 | `25945.wav` | Bernie Sanders | `bona-fide` | `bona_fide` | 0.7560 | 76/100 | ✅ Correct |
| 4 | `16979.wav` | JFK | `bona-fide` | `spoof` | 0.0001 | 0/100 | ❌ Error |
| 5 | `14445.wav` | Barack Obama | `bona-fide` | `spoof` | 0.2135 | 21/100 | ❌ Error |
| 6 | `23748.wav` | JFK | `bona-fide` | `bona_fide` | 0.5368 | 54/100 | ✅ Correct |
| 7 | `11237.wav` | Donald Trump | `bona-fide` | `bona_fide` | 0.6149 | 61/100 | ✅ Correct |
| 8 | `4483.wav` | Mark Zuckerberg | `bona-fide` | `bona_fide` | 0.9924 | 99/100 | ✅ Correct |
| 9 | `26038.wav` | Ayn Rand | `bona-fide` | `spoof` | 0.0897 | 9/100 | ❌ Error |
| 10 | `23822.wav` | Kanye West | `bona-fide` | `bona_fide` | 0.9927 | 99/100 | ✅ Correct |
| 11 | `21859.wav` | Arnold Schwarzenegger | `spoof` | `spoof` | 0.0058 | 1/100 | ✅ Correct |
| 12 | `24063.wav` | Bernie Sanders | `spoof` | `bona_fide` | 0.9992 | 100/100 | ❌ Error |
| 13 | `30266.wav` | Orson Welles | `spoof` | `bona_fide` | 0.9277 | 93/100 | ❌ Error |
| 14 | `2007.wav` | Ayn Rand | `spoof` | `bona_fide` | 0.7494 | 75/100 | ❌ Error |
| 15 | `22425.wav` | Ayn Rand | `spoof` | `spoof` | 0.0648 | 6/100 | ✅ Correct |
| 16 | `28096.wav` | Alec Guinness | `spoof` | `bona_fide` | 0.9952 | 100/100 | ❌ Error |
| 17 | `9882.wav` | Alec Guinness | `spoof` | `spoof` | 0.1169 | 12/100 | ✅ Correct |
| 18 | `11209.wav` | Bernie Sanders | `spoof` | `bona_fide` | 0.9962 | 100/100 | ❌ Error |
| 19 | `30340.wav` | Alec Guinness | `spoof` | `bona_fide` | 0.9917 | 99/100 | ❌ Error |
| 20 | `30698.wav` | Milton Friedman | `spoof` | `spoof` | 0.0002 | 0/100 | ✅ Correct |

---

## ⚠️ Key Findings & Limitations

1. **Zero Model Tampering:** Inference used the original, unmodified `AASIST.pth` weights without fine-tuning, adhering strictly to audit protocols.
2. **Domain Shift & Uncalibrated Scores:** The AASIST checkpoint was originally trained on the ASVspoof 2019 Logical Access studio dataset. In-the-wild recordings contain spontaneous background noise, compression artifacts, and varied acoustics that cause distribution shifts.
3. **No General Accuracy Claims:** This evaluation verifies pipeline functionality on a balanced sample ($N=20$). General production detection accuracy cannot be claimed across the full dataset without complete out-of-domain evaluation and probability calibration.
