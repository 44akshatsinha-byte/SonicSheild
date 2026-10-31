# SonicShield: Technical Audit of Multi-Engine Scoring & In-The-Wild Evaluation

**Date:** October 31, 2026  
**Audit Scope:** 200-recording balanced benchmark (`data/external/in_the_wild/release_in_the_wild/`, Seed 42)  
**Sample Composition:** 100 Bona-Fide (genuine human speech) and 100 Spoof (deepfake audio)  
**Evaluated Systems:**
1. Physics Engine (spectral flatness, harmonic ratio, noise envelope, room acoustics)
2. Biology Engine (pitch continuity, vocal tract stability, jitter, shimmer)
3. Pretrained AASIST Engine (AASIST graph neural network on raw 16 kHz waveforms)
4. Combined Baseline (`0.50 * Physics + 0.50 * Biology`)
5. Combined Hybrid (`0.50 * Baseline + 0.50 * AASIST`)

---

## Executive Summary & Core Verdict

| Metric | Physics Engine | Biology Engine | AASIST Engine | Combined Baseline | Combined Hybrid | Hybrid vs AASIST Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy (th=0.5)** | 46.0% | 53.5% | **56.5%** | 46.0% | **57.5%** | **+1.0%** |
| **ROC-AUC** | 0.4736 | 0.5545 | **0.6225** | 0.4852 | **0.5712** | **-0.0513 (Worse)** |
| **PR-AUC (Avg Prec)** | 0.5021 | 0.5619 | **0.6677** | 0.5145 | **0.5624** | **-0.1053 (Worse)** |
| **Equal Error Rate (EER)**| 0.5250 | 0.4950 | **0.4400** | 0.5450 | **0.4650** | **+0.0250 (Worse)** |
| **True Positives (TP)** | 13 / 100 | 99 / 100 | 77 / 100 | 38 / 100 | 79 / 100 | +2 genuine caught |
| **True Negatives (TN)** | 79 / 100 | 8 / 100 | 36 / 100 | 54 / 100 | 36 / 100 | **0 (Identical)** |
| **False Acceptance Rate** | 21.0% | 92.0% | 64.0% | 46.0% | 64.0% | **0.0% (Identical)** |
| **False Rejection Rate** | 87.0% | 1.0% | 23.0% | 62.0% | 21.0% | -2.0% |

### Key Scientific Conclusions:
1. **The hybrid accuracy is 57.5%, compared with 56.5% for AASIST (+1.0% marginal difference).**
2. **Hybrid ROC-AUC (0.5712) and EER (0.4650) are distinctly worse than AASIST alone (ROC-AUC 0.6225, EER 0.4400).**
3. **The hybrid catches the exact same number of fake recordings as AASIST in this sample (True Negatives = 36 / 100, FAR = 64.0%).**
4. **The empirical results do NOT establish that the current hybrid formula is superior to standalone AASIST.** In reality, taking a naive 50/50 linear average of an uncalibrated heuristic baseline with a deep neural network degrades the model's discriminative ranking power across all threshold ranges.

---

## 1. Verification of Score Directions & Empirical Distributions

### 1.1 Architectural Design Intent
Across all components of SonicShield, the score convention is architected so that:
$$\text{Score} \in [0.0, 1.0]$$
- $\mathbf{1.0} \implies \textbf{Bona-Fide}$ (Genuine human speech / live speaker)
- $\mathbf{0.0} \implies \textbf{Spoof}$ (Synthesized speech / voice conversion / deepfake)
- **Decision Rule at Default Operating Point:** If $\text{Score} \ge 0.5 \implies \text{Classify as Bona-Fide}$; if $\text{Score} < 0.5 \implies \text{Classify as Spoof}$.

### 1.2 Empirical Score Distributions (200 In-The-Wild Audio Files)
Calculating the sample mean, median, standard deviation, and range for genuine ($n=100$) and fake ($n=100$) recordings independently reveals the actual behavior:

| Engine | Label | Mean Score | Median Score | Std Dev | Min Score | Max Score | Mean Delta ($\mu_{\text{bona}} - \mu_{\text{spoof}}$) | Empirical Direction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Physics Engine** | Bona-Fide<br>Spoof | 0.3871<br>0.4107 | 0.3480<br>0.3480 | 0.1060<br>0.1182 | 0.3450<br>0.3450 | 0.8193<br>0.7031 | **-0.0236** | **Inverted / Collapsed** (Fake scores higher than Genuine) |
| **Biology Engine** | Bona-Fide<br>Spoof | 0.6466<br>0.6264 | 0.6100<br>0.6100 | 0.0890<br>0.0951 | 0.4720<br>0.4270 | 0.8600<br>0.7801 | **+0.0202** | Near-Identical (Median separation = 0.0000) |
| **AASIST Engine** | Bona-Fide<br>Spoof | 0.7596<br>0.6252 | 0.9794<br>0.8918 | 0.3567<br>0.4225 | 0.0000<br>0.0002 | 1.0000<br>0.9997 | **+0.1344** | **Genuine > Fake** (Median delta = +0.0877) |
| **Combined Baseline** | Bona-Fide<br>Spoof | 0.5184<br>0.5198 | 0.4800<br>0.4800 | 0.0782<br>0.0815 | 0.4100<br>0.3900 | 0.8100<br>0.7340 | **-0.0014** | **Null Separation** (Zero discriminative margin) |
| **Combined Hybrid** | Bona-Fide<br>Spoof | 0.6390<br>0.5725 | 0.7264<br>0.7060 | 0.1873<br>0.2221 | 0.2301<br>0.2004 | 0.9050<br>0.8644 | **+0.0665** | **Genuine > Fake** (Carried entirely by AASIST) |

### 1.3 Root Cause Analysis of Score Drift & Inversion
1. **Physics Engine Inversion ($\Delta = -0.0236$, AUC 0.4736):**
   The Physics Engine evaluates room acoustic reverberation, spectral rolloff, and high-frequency noise floor. In controlled studio environments, these heuristics function well. However, in the In-The-Wild dataset:
   - Genuine recordings are often professionally captured podcast or interview microphones equipped with noise gates, pop filters, and dry acoustics (yielding low reverberation and low background noise floor).
   - Deepfake recordings harvested from social media often contain compression artifacts, background room audio, or secondary acoustic re-recording that the heuristic misinterprets as "natural acoustic environment."
   - Consequently, fake recordings achieve a slightly higher mean physics score than genuine studio recordings.
2. **Biology Engine Compression ($\Delta = +0.0202$, Medians identical at 0.6100):**
   The Biology Engine inspects pitch jitter, vocal tract stability, and shimmer. Modern deepfake vocoders (e.g., HiFi-GAN, VITS, BigVGAN) now synthesize continuous pitch tracks with micro-prosody that easily bypass simple acoustic jitter/shimmer thresholds. As a result, 92 out of 100 fake recordings scored $\ge 0.5$, producing an extreme False Acceptance Rate of 92.0%.
3. **No Post-Hoc Inversion:**
   Per scientific integrity guidelines, scores must **never** be inverted post-hoc simply to elevate AUC numbers. The heuristic engines must be recognized as exhibiting domain sensitivity on in-the-wild social media audio.

---

## 2. Audit of Score Comparability & Mathematical Fusion

### 2.1 The Current Hybrid Formula
The production fusion logic implemented in `src/pipeline.py` computes:
$$\text{Score}_{\text{Baseline}} = 0.50 \cdot \text{Score}_{\text{Physics}} + 0.50 \cdot \text{Score}_{\text{Biology}}$$
$$\text{Score}_{\text{Hybrid}} = 0.50 \cdot \text{Score}_{\text{Baseline}} + 0.50 \cdot \text{Score}_{\text{AASIST}}$$

Expanding this into full linear form:
$$\text{Score}_{\text{Hybrid}} = 0.25 \cdot \text{Score}_{\text{Physics}} + 0.25 \cdot \text{Score}_{\text{Biology}} + 0.50 \cdot \text{Score}_{\text{AASIST}}$$

### 2.2 Calibration Audit: Are These Scores Probabilities?
**No. None of the five engine scores represent calibrated posterior probabilities.**
- **Physics and Biology Engines:** These values are heuristic bounding scores constructed via min-max linear clamping:
  $$\text{Score}_{\text{heuristic}} = \text{clip}\left(\sum w_i f_i, 0.0, 1.0\right)$$
  They have never been fitted using logistic loss, Brier score optimization, or isotonic regression against empirical ground truth. They do **not** reflect $P(Y = \text{bona-fide} \mid X)$. Describing them as "probabilities" or "percentages of genuineness" in user-facing documentation or UI is mathematically inaccurate.
- **AASIST Engine:** AASIST outputs raw class logits $(z_0, z_1)$ and applies standard softmax:
  $$\sigma(z_1) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$
  While the softmax output is bounded in $[0, 1]$, deep neural networks are well-documented to suffer from severe overconfidence under domain shift (Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017). AASIST was pretrained on ASVspoof 2019 logical access (controlled telephonic audio). When presented with in-the-wild YouTube audio, its softmax distribution polarizes toward 0.0 or 1.0, failing reliability diagrams.

### 2.3 Why Linear Averaging is Statistically Flawed
Taking a weighted arithmetic mean of uncalibrated scores across radically disparate distributions is statistically unsound for three fundamental reasons:

1. **Incommensurate Variances:**
   - $\text{Var}(\text{AASIST}) \approx 0.155$ (broad bimodal distribution spanning $[0.000, 1.000]$).
   - $\text{Var}(\text{Baseline}) \approx 0.006$ (narrow distribution clustered tightly within $[0.40, 0.65]$).
   Linear combination assumes equal scale variance. Here, the low-variance heuristic score acts as an arbitrary DC bias shift that compresses the AASIST decision boundary.
2. **Noise Dilution of Strong Signal:**
   AASIST achieves a standalone ROC-AUC of **0.6225** and Average Precision of **0.6677**. The Combined Baseline achieves an AUC of **0.4852** (worse than random guessing). Averaging a signal with AUC 0.6225 and a noise signal with AUC 0.4852 mathematically pulls the resulting hybrid AUC down to **0.5712**, destroying $5.13\%$ of AASIST's discriminative ranking capacity.
3. **Threshold Distortion:**
   Because the baseline engine rarely drops below $0.40$ (even on egregious deepfakes), adding $0.50 \times \text{Baseline}$ prevents the hybrid score from ever dropping close to $0.0$. An AASIST spoof logit producing score $0.05$ gets lifted to $0.05 \times 0.5 + 0.48 \times 0.5 = 0.265$.

---

## 3. Threshold Analysis & Operating Characteristics

### 3.1 Complete Operating Sweep Across Thresholds $[0.1 \text{ to } 0.9]$

The table below documents False Acceptance Rate (FAR, spoof classified as genuine), False Rejection Rate (FRR, genuine classified as spoof), Accuracy, Precision, Recall, F1-score, and confusion matrix counts across the entire operating spectrum for all five engines:

#### AASIST Engine (ROC-AUC: 0.6225, PR-AUC: 0.6677, EER: 0.4400)
| Threshold | FAR | FRR | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | 0.7300 | 0.1500 | 56.0% | 0.5380 | 0.8500 | 0.6589 | 85 | 73 | 27 | 15 |
| **0.2** | 0.7000 | 0.1600 | 57.0% | 0.5455 | 0.8400 | 0.6614 | 84 | 70 | 30 | 16 |
| **0.3** | 0.6800 | 0.1700 | 57.5% | 0.5497 | 0.8300 | 0.6614 | 83 | 68 | 32 | 17 |
| **0.4** | 0.6500 | 0.1700 | 59.0% | 0.5608 | 0.8300 | 0.6694 | 83 | 65 | 35 | 17 |
| **0.5 (Default)**| **0.6400** | **0.2300** | **56.5%** | **0.5461** | **0.7700** | **0.6390** | **77** | **64** | **36** | **23** |
| **0.6** | 0.6400 | 0.2400 | 56.0% | 0.5429 | 0.7600 | 0.6333 | 76 | 64 | 36 | 24 |
| **0.7** | 0.6000 | 0.2700 | 56.5% | 0.5489 | 0.7300 | 0.6266 | 73 | 60 | 40 | 27 |
| **0.8** | 0.5700 | 0.3200 | 55.5% | 0.5440 | 0.6800 | 0.6044 | 68 | 57 | 43 | 32 |
| **0.9** | 0.4900 | 0.3700 | 57.0% | 0.5625 | 0.6300 | 0.5943 | 63 | 49 | 51 | 37 |

#### Combined Hybrid Engine (ROC-AUC: 0.5712, PR-AUC: 0.5624, EER: 0.4650)
| Threshold | FAR | FRR | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | 1.0000 | 0.0000 | 50.0% | 0.5000 | 1.0000 | 0.6667 | 100 | 100 | 0 | 0 |
| **0.2** | 1.0000 | 0.0000 | 50.0% | 0.5000 | 1.0000 | 0.6667 | 100 | 100 | 0 | 0 |
| **0.3** | 0.7700 | 0.1300 | 55.0% | 0.5305 | 0.8700 | 0.6591 | 87 | 77 | 23 | 13 |
| **0.4** | 0.6900 | 0.1600 | 57.5% | 0.5490 | 0.8400 | 0.6640 | 84 | 69 | 31 | 16 |
| **0.5 (Default)**| **0.6400** | **0.2100** | **57.5%** | **0.5524** | **0.7900** | **0.6502** | **79** | **64** | **36** | **21** |
| **0.6** | 0.6100 | 0.2600 | 56.5% | 0.5481 | 0.7400 | 0.6298 | 74 | 61 | 39 | 26 |
| **0.7** | 0.5200 | 0.3800 | 55.0% | 0.5439 | 0.6200 | 0.5794 | 62 | 52 | 48 | 38 |
| **0.8** | 0.0700 | 0.9200 | 50.5% | 0.5333 | 0.0800 | 0.1391 | 8 | 7 | 93 | 92 |
| **0.9** | 0.0000 | 0.9900 | 50.5% | 1.0000 | 0.0100 | 0.0198 | 1 | 0 | 100 | 99 |

#### Combined Baseline Engine (ROC-AUC: 0.4852, PR-AUC: 0.5145, EER: 0.5450)
| Threshold | FAR | FRR | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.4** | 0.9600 | 0.0000 | 52.0% | 0.5102 | 1.0000 | 0.6757 | 100 | 96 | 4 | 0 |
| **0.5 (Default)**| **0.4600** | **0.6200** | **46.0%** | **0.4524** | **0.3800** | **0.4130** | **38** | **46** | **54** | **62** |
| **0.6** | 0.1800 | 0.8300 | 49.5% | 0.4857 | 0.1700 | 0.2519 | 17 | 18 | 82 | 83 |

### 3.2 Visual Comparison of ROC and Precision-Recall Curves
- The multi-engine ROC Curves comparison is saved at [`results/in_the_wild/roc_curves_comparison.png`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/roc_curves_comparison.png).
- The multi-engine Precision-Recall Curves comparison is saved at [`results/in_the_wild/pr_curves_comparison.png`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/pr_curves_comparison.png).

### 3.3 The Fallacy of Post-Hoc Threshold Optimization
Notice that at threshold 0.4, AASIST achieves 59.0% accuracy (F1: 0.6694), and at threshold 0.9, AASIST drops its FAR to 49.0% with 57.0% accuracy.
However, **selecting an optimal threshold based on this 200-recording evaluation and reporting it as general system accuracy is a methodological violation (data snooping / target leakage).**

To legitimately optimize thresholds, the project must adhere to a strict three-way data split protocol:
1. **Fit / Train Split (60%):** Estimate calibration parameters (e.g. Platt logistic weights $\alpha, \beta$).
2. **Validation Split (20%):** Determine operating threshold $\tau$ corresponding to a target business metric (e.g. $\text{FAR} \le 5\%$ or empirical EER).
3. **Untouched Test Split (20%):** Report final evaluation once without further threshold adjustments.

---

## 4. Speaker-Level Generalization & Overlap Analysis

### 4.1 Sample Speaker Demographics
- **Total Unique Speakers:** 38 public figures across 200 recordings.
- **Overlapping Speakers (both bona-fide and spoof present):** 21 speakers accounting for **171 out of 200 recordings (85.5% of the evaluation sample)**.
- **Disjoint Speakers:**
  - 10 speakers with bona-fide recordings only (16 recordings).
  - 7 speakers with spoof recordings only (13 recordings).

### 4.2 Performance Breakdown Across Overlapping Speakers (Threshold = 0.5)

| Speaker | Bona Count | Spoof Count | Total Files | AASIST Accuracy | Hybrid Accuracy | Baseline Accuracy | Physics Accuracy | Biology Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alec Guinness** | 9 | 23 | 32 | 56.2% | 56.2% | 40.6% | 37.5% | 43.8% |
| **Bernie Sanders** | 8 | 11 | 19 | 42.1% | 42.1% | 84.2% | 57.9% | 42.1% |
| **Ayn Rand** | 10 | 7 | 17 | 29.4% | 29.4% | 17.6% | 52.9% | 58.8% |
| **Donald Trump** | 15 | 2 | 17 | 94.1% | 94.1% | 35.3% | 17.6% | 88.2% |
| **Barack Obama** | 9 | 1 | 10 | 50.0% | 50.0% | 10.0% | 10.0% | 80.0% |
| **Ronald Reagan** | 2 | 8 | 10 | 40.0% | 40.0% | 70.0% | 80.0% | 30.0% |
| **Bill Clinton** | 5 | 3 | 8 | 75.0% | 75.0% | 25.0% | 25.0% | 62.5% |
| **Winston Churchill** | 5 | 2 | 7 | 100.0% | 100.0% | 57.1% | 28.6% | 85.7% |
| **Christopher Hitchens**| 2 | 4 | 6 | 33.3% | 50.0% | 66.7% | 66.7% | 50.0% |
| **JFK** | 3 | 3 | 6 | 50.0% | 50.0% | 0.0% | 33.3% | 50.0% |
| **Queen Elizabeth II** | 1 | 5 | 6 | 83.3% | 83.3% | 83.3% | 83.3% | 16.7% |
| **Mark Zuckerberg** | 1 | 4 | 5 | 80.0% | 80.0% | 80.0% | 80.0% | 20.0% |
| **Richard Nixon** | 4 | 1 | 5 | 60.0% | 60.0% | 40.0% | 0.0% | 80.0% |
| **Milton Friedman** | 2 | 2 | 4 | 100.0% | 100.0% | 25.0% | 50.0% | 50.0% |
| **Boris Johnson** | 2 | 1 | 3 | 100.0% | 100.0% | 66.7% | 66.7% | 66.7% |
| **George W. Bush** | 1 | 2 | 3 | 66.7% | 66.7% | 66.7% | 66.7% | 33.3% |
| **Frank Sinatra** | 1 | 2 | 3 | 66.7% | 66.7% | 33.3% | 66.7% | 33.3% |
| **Orson Welles** | 1 | 2 | 3 | 33.3% | 33.3% | 33.3% | 66.7% | 33.3% |
| **Martin Luther King**| 1 | 2 | 3 | 0.0% | 0.0% | 0.0% | 66.7% | 33.3% |
| **Kanye West** | 1 | 1 | 2 | 50.0% | 50.0% | 0.0% | 50.0% | 50.0% |
| **George Carlin** | 1 | 1 | 2 | 50.0% | 50.0% | 50.0% | 100.0% | 50.0% |

### 4.3 Generalization Gap: Overlapping vs Disjoint Performance
Evaluating the model separately on speakers that appear in both classes versus speakers that appear only in a single class exposes a critical insight:

| Evaluation Subset | Sample Size | AASIST Accuracy | AASIST ROC-AUC | Hybrid Accuracy | Hybrid ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overlapping Speakers** (Both bona & spoof) | $n = 171$ | **58.5%** | **0.6449** | **59.1%** | **0.5889** |
| **Disjoint Speakers** (Single class only) | $n = 29$ | **44.8%** | **0.4543** | **48.3%** | **0.4399** |

**Crucial Scientific Takeaway:**
When evaluating speakers whose vocal profiles were not balanced across classes, model accuracy plummets below random guessing (44.8% for AASIST, 48.3% for Hybrid). This indicates that the classifiers are heavily influenced by **speaker timbral idiosyncratic cues and background acoustic environments**, rather than detecting purely acoustic artifacts of voice cloning.

### 4.4 Feasibility of Speaker-Balanced Evaluation
The full In-The-Wild dataset contains 31,779 recordings across 58 unique speakers. A speaker-balanced evaluation protocol is highly feasible:
- We can construct a **Speaker-Disjoint Cross-Validation (SD-CV)** split where the voices present in the test partition never appear in the training or calibration sets.
- This prevents acoustic memorization of speaker-specific pitch registers and provides an unbiased estimate of real-world zero-shot performance on unknown voices.

---

## 5. Architectural Recommendation

Based on the empirical evidence gathered during this technical audit, we present a definitive recommendation on the multi-engine scoring architecture:

### Verdict: **REDESIGN THE HYBRID SCORING SYSTEM**
The current formula:
$$\text{Score}_{\text{Hybrid}} = 0.50 \cdot (0.50 \cdot \text{Physics} + 0.50 \cdot \text{Biology}) + 0.50 \cdot \text{AASIST}$$
**must NOT be retained in production.**

### Rationale:
1. **Performance Degradation:** The hybrid formula reduces ROC-AUC from $0.6225 \to 0.5712$ and PR-AUC from $0.6677 \to 0.5624$.
2. **Zero Gain in Deepfake Detection:** The hybrid catches 36 fakes out of 100, exactly identical to AASIST alone. It provides zero additional defensive coverage against voice clones.
3. **Uncalibrated Averaging:** Linearly adding uncalibrated bounded heuristics to deep neural logits creates statistical distortion without probabilistic meaning.

---

### Proposed Redesign: Two Viable Engineering Paths

#### Option A: Learned Stacking with Regularized Platt Scaling (Recommended)
Rather than arbitrary 50/50 linear weights, treat the multi-engine outputs as feature representations into a calibrated stacking layer:
$$P(\text{bona-fide} \mid x) = \sigma\left(w_0 + w_1 \cdot z_{\text{AASIST}} + w_2 \cdot f_{\text{Physics}} + w_3 \cdot f_{\text{Biology}} + w_4 \cdot \text{Confidence}\right)$$
- Fit weights $w_i$ using L2-regularized logistic regression on a dedicated 60% calibration set.
- If the heuristic features provide zero or negative discriminative value on in-the-wild audio, the regularizer naturally shrinks $w_2$ and $w_3$ toward zero, preventing degradation of AASIST's signal.

#### Option B: Asymmetric Cascaded / Gated Pipeline
Instead of blending scores into a single scalar:
1. Use **AASIST** as the primary detection engine.
2. Use the **Physics Engine** specifically as a sanity filter for synthetic silence / zero-crossing anomalies.
3. Use the **Biology Engine** strictly when vocal tract stability indicates overt robotic discontinuity.
4. Output separate, un-averaged diagnostic channels to the user dashboard rather than a misleading blended "probability."

---

## Summary of Generated Audit Artifacts

- **Detailed Comparative CSV (200 files):** [`results/in_the_wild/comparative_results_200.csv`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/comparative_results_200.csv)
- **Metrics Summary Table:** [`results/in_the_wild/comparative_metrics_table.csv`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/comparative_metrics_table.csv)
- **Confusion Matrices Quad-Plot:** [`results/in_the_wild/confusion_matrices_comparison.png`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/confusion_matrices_comparison.png)
- **ROC Curves Quad-Plot:** [`results/in_the_wild/roc_curves_comparison.png`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/roc_curves_comparison.png)
- **Precision-Recall Curves Plot:** [`results/in_the_wild/pr_curves_comparison.png`](file:///C:/Users/Akshat/Desktop/Sonic%20Sheild/results/in_the_wild/pr_curves_comparison.png)
- **Full Test Suite Status:** **60 / 60 tests passing** in `pytest`.

