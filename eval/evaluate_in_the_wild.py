"""SonicShield In-The-Wild Audio Deepfake Dataset Evaluation.

Evaluates the pretrained AASIST model on the In-The-Wild dataset
(Müller et al.) without retraining or fine-tuning.
Produces reproducible CSV predictions, confusion matrix image,
detailed Markdown report, and skipped-files log.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import time
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from eval.metrics import compute_eer
from src.pretrained_aasist import get_aasist_classifier, AASIST_NB_SAMPLES, AASIST_SAMPLE_RATE

DEFAULT_DATASET_DIR = os.path.join(
    "data", "external", "in_the_wild", "release_in_the_wild", "release_in_the_wild"
)
DEFAULT_META_PATH = os.path.join(DEFAULT_DATASET_DIR, "meta.csv")
DEFAULT_OUTPUT_DIR = os.path.join("results", "in_the_wild")


def render_confusion_matrix_image(
    cm: np.ndarray,
    labels: list[str],
    metrics: dict,
    output_path: str,
) -> None:
    """Renders a clean, high-resolution PNG image of the confusion matrix using Pillow."""
    width, height = 760, 560
    img = Image.new("RGB", (width, height), color=(11, 19, 38))
    draw = ImageDraw.Draw(img)

    # Fallback to default bitmap font if TTF not found
    font_large = ImageFont.load_default()
    font_med = ImageFont.load_default()
    font_small = ImageFont.load_default()

    # Title header
    draw.text((30, 25), "SONICSHIELD — IN-THE-WILD EVALUATION", fill=(0, 242, 254), font=font_large)
    draw.text((30, 50), "Pretrained AASIST Anti-Spoofing Model (ASVspoof Checkpoint)", fill=(148, 163, 184), font=font_med)
    draw.line([(30, 75), (width - 30, 75)], fill=(30, 41, 59), width=2)

    # Grid dimensions
    grid_x = 180
    grid_y = 120
    cell_w = 230
    cell_h = 140

    # Labels
    # Rows: Actual (0 = Fake / Spoof, 1 = Genuine / Bona-fide)
    # Cols: Predicted (0 = Fake / Spoof, 1 = Genuine / Bona-fide)
    draw.text((grid_x + 60, grid_y - 30), "Pred: Spoof (Fake)", fill=(244, 114, 182), font=font_med)
    draw.text((grid_x + cell_w + 50, grid_y - 30), "Pred: Bona-fide (Live)", fill=(0, 230, 118), font=font_med)

    draw.text((35, grid_y + 55), "True: Spoof", fill=(244, 114, 182), font=font_med)
    draw.text((35, grid_y + cell_h + 55), "True: Bona-fide", fill=(0, 230, 118), font=font_med)

    # Cell coordinates & colors:
    # cm[0,0] = TN (True Spoof, Pred Spoof)
    # cm[0,1] = FP (True Spoof, Pred Bona-fide)
    # cm[1,0] = FN (True Bona-fide, Pred Spoof)
    # cm[1,1] = TP (True Bona-fide, Pred Bona-fide)
    cells = [
        ((0, 0), grid_x, grid_y, cm[0, 0], "True Spoof (TN)", (34, 197, 94, 50)),
        ((0, 1), grid_x + cell_w, grid_y, cm[0, 1], "False Live (FP)", (239, 68, 68, 60)),
        ((1, 0), grid_x, grid_y + cell_h, cm[1, 0], "False Spoof (FN)", (239, 68, 68, 60)),
        ((1, 1), grid_x + cell_w, grid_y + cell_h, cm[1, 1], "True Live (TP)", (34, 197, 94, 50)),
    ]

    total_samples = int(np.sum(cm))

    for (r, c), x, y, count, label_name, fill_color in cells:
        # Background rect
        pct = (count / total_samples) * 100.0 if total_samples > 0 else 0.0
        # Draw cell background
        draw.rectangle([x, y, x + cell_w - 10, y + cell_h - 10], fill=(22, 33, 62), outline=(51, 65, 85), width=2)
        
        # Center count
        count_str = f"{count}"
        draw.text((x + (cell_w - 10) // 2 - 10, y + 35), count_str, fill=(255, 255, 255), font=font_large)
        draw.text((x + (cell_w - 10) // 2 - 25, y + 65), f"({pct:.1f}%)", fill=(148, 163, 184), font=font_small)
        draw.text((x + 20, y + 95), label_name, fill=(203, 213, 225), font=font_small)

    # Metrics Summary Bar at Bottom
    summary_y = grid_y + 2 * cell_h + 30
    draw.line([(30, summary_y), (width - 30, summary_y)], fill=(30, 41, 59), width=2)

    acc = metrics.get("accuracy", 0.0)
    prec = metrics.get("precision_bonafide", 0.0)
    rec = metrics.get("recall_bonafide", 0.0)
    f1 = metrics.get("f1_bonafide", 0.0)
    auc = metrics.get("roc_auc")
    eer = metrics.get("eer")

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    eer_str = f"{eer:.4f}" if eer is not None else "N/A"

    line1 = f"Sample: N={total_samples} (Balanced: 10 Genuine / 10 Fake)   |   Accuracy: {acc * 100:.1f}%"
    line2 = f"Genuine Precision: {prec:.3f}   |   Recall: {rec:.3f}   |   F1: {f1:.3f}   |   ROC-AUC: {auc_str}   |   EER: {eer_str}"
    
    draw.text((35, summary_y + 15), line1, fill=(255, 255, 255), font=font_med)
    draw.text((35, summary_y + 40), line2, fill=(0, 242, 254), font=font_med)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def run_in_the_wild_evaluation(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    meta_path: str = DEFAULT_META_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_size_per_class: int = 10,
    random_seed: int = 42,
) -> dict:
    """Executes evaluation on a balanced sample from the In-The-Wild dataset."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("SONICSHIELD — IN-THE-WILD AUDIO DEEPFAKE EVALUATION")
    print("=" * 72)
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Metadata File:     {meta_path}")
    print(f"Output Directory:  {output_dir}")

    # Check files exist
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # 1. Read metadata
    df = pd.read_csv(meta_path)
    total_in_meta = len(df)
    label_counts = df["label"].value_counts().to_dict()
    print(f"\nTotal records in metadata: {total_in_meta}")
    print(f"Label distribution: {label_counts}")

    # 2. Get AASIST model classifier
    classifier = get_aasist_classifier()
    if not classifier.is_loaded:
        raise RuntimeError(f"Pretrained AASIST model is not loaded: {classifier.load_error}")
    print(f"AASIST Model Status: {classifier.load_status.upper()} (Checkpoint: {classifier.checkpoint_path})")

    # 3. Balanced sampling (10 bona-fide, 10 spoof)
    bona_pool = df[df["label"] == "bona-fide"]
    spoof_pool = df[df["label"] == "spoof"]

    if len(bona_pool) < sample_size_per_class or len(spoof_pool) < sample_size_per_class:
        raise ValueError(
            f"Not enough samples in dataset. Requested {sample_size_per_class} per class, "
            f"found {len(bona_pool)} bona-fide and {len(spoof_pool)} spoof."
        )

    bona_sample = bona_pool.sample(n=sample_size_per_class, random_state=random_seed)
    spoof_sample = spoof_pool.sample(n=sample_size_per_class, random_state=random_seed)
    eval_df = pd.concat([bona_sample, spoof_sample]).reset_index(drop=True)

    print(f"\nSelected balanced evaluation sample: {len(eval_df)} recordings "
          f"({sample_size_per_class} bona-fide, {sample_size_per_class} spoof)")

    # 4. Run inference per recording
    evaluated_records = []
    skipped_records = []

    for idx, row in eval_df.iterrows():
        fname = row["file"]
        speaker = row["speaker"]
        true_label = row["label"]  # 'bona-fide' or 'spoof'
        true_bin = 1 if true_label == "bona-fide" else 0
        audio_path = os.path.join(dataset_dir, fname)

        if not os.path.exists(audio_path):
            skipped_records.append({
                "file": fname,
                "speaker": speaker,
                "label": true_label,
                "reason": f"File not found on disk at {audio_path}",
            })
            continue

        try:
            # Inspect audio info
            audio_info = sf.info(audio_path)
            duration_sec = round(audio_info.duration, 3)
            sr = audio_info.samplerate

            # Read audio as float32 mono
            audio_data, file_sr = sf.read(audio_path, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            if len(audio_data) == 0:
                skipped_records.append({
                    "file": fname,
                    "speaker": speaker,
                    "label": true_label,
                    "reason": "Audio file contains 0 samples (empty).",
                })
                continue

            # Run model prediction (one prediction per original recording)
            pred_res = classifier.predict(audio_data, sr=file_sr, codec="clean")

            pred_class = pred_res.get("predicted_class")  # 'bona_fide', 'spoof', or error
            pred_bin = 1 if pred_class == "bona_fide" else 0
            bf_score = pred_res.get("bona_fide_score")
            live_score = pred_res.get("liveness_score")
            raw_logits = pred_res.get("raw_score", [None, None])

            record = {
                "file": fname,
                "speaker": speaker,
                "duration_sec": duration_sec,
                "sample_rate": sr,
                "true_label": true_label,
                "true_binary": true_bin,
                "predicted_label": pred_class,
                "predicted_binary": pred_bin,
                "is_correct": bool(true_bin == pred_bin),
                "bona_fide_score": bf_score,
                "spoof_score": round(1.0 - bf_score, 4) if bf_score is not None else None,
                "liveness_score": live_score,
                "spoof_logit": raw_logits[0] if raw_logits else None,
                "bonafide_logit": raw_logits[1] if raw_logits else None,
                "warnings": "; ".join(pred_res.get("warnings", [])),
                "status": pred_res.get("status", "success"),
            }
            evaluated_records.append(record)

        except Exception as e:
            skipped_records.append({
                "file": fname,
                "speaker": speaker,
                "label": true_label,
                "reason": f"Processing exception: {e}",
            })

    # 5. Calculate Metrics
    y_true = np.array([r["true_binary"] for r in evaluated_records])
    y_pred = np.array([r["predicted_binary"] for r in evaluated_records])
    y_scores = np.array([r["bona_fide_score"] for r in evaluated_records])

    total_eval = len(evaluated_records)
    total_skipped = len(skipped_records)
    n_bona = int(np.sum(y_true == 1))
    n_spoof = int(np.sum(y_true == 0))

    acc = float(accuracy_score(y_true, y_pred))
    prec_bona = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_bona = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_bona = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    prec_spoof = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_spoof = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_spoof = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    # Confusion matrix with explicit labels [0: spoof, 1: bona-fide]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = None

    # Equal Error Rate (EER)
    eer, eer_thresh = compute_eer(y_true, y_scores)
    eer_val = round(eer, 4) if not np.isnan(eer) else None
    eer_thresh_val = round(eer_thresh, 4) if not np.isnan(eer_thresh) else None

    metrics = {
        "total_evaluated": total_eval,
        "total_skipped": total_skipped,
        "genuine_evaluated": n_bona,
        "fake_evaluated": n_spoof,
        "accuracy": round(acc, 4),
        "precision_bonafide": round(prec_bona, 4),
        "recall_bonafide": round(rec_bona, 4),
        "f1_bonafide": round(f1_bona, 4),
        "precision_spoof": round(prec_spoof, 4),
        "recall_spoof": round(rec_spoof, 4),
        "f1_spoof": round(f1_spoof, 4),
        "macro_f1": round(macro_f1, 4),
        "true_positives_bona": int(tp),
        "true_negatives_spoof": int(tn),
        "false_positives_fake_as_live": int(fp),
        "false_negatives_live_as_fake": int(fn),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "eer": eer_val,
        "eer_optimal_threshold": eer_thresh_val,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    # 6. Save Outputs
    # A. CSV
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "speaker", "duration_sec", "sample_rate",
            "true_label", "true_binary", "predicted_label", "predicted_binary",
            "is_correct", "bona_fide_score", "spoof_score", "liveness_score",
            "spoof_logit", "bonafide_logit", "status", "warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in evaluated_records:
            writer.writerow(r)

    # B. Confusion Matrix Image
    img_path = os.path.join(output_dir, "confusion_matrix.png")
    render_confusion_matrix_image(cm, labels=["spoof", "bona_fide"], metrics=metrics, output_path=img_path)

    # C. Skipped Files Log
    log_path = os.path.join(output_dir, "skipped_files.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"SonicShield In-The-Wild Evaluation — Skipped Files Log\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Skipped: {total_skipped}\n\n")
        if total_skipped == 0:
            f.write("No audio files were skipped. All 20 recordings were successfully evaluated.\n")
        else:
            for s in skipped_records:
                f.write(f"- File: {s['file']} | Speaker: {s['speaker']} | Label: {s['label']} | Reason: {s['reason']}\n")

    # D. Markdown Evaluation Report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    write_markdown_report(report_path, metrics, evaluated_records, cm)

    print("\n" + "=" * 72)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"Total evaluated:      {metrics['total_evaluated']} ({n_bona} Genuine / {n_spoof} Fake)")
    print(f"Accuracy:             {metrics['accuracy'] * 100:.2f}%")
    print(f"Genuine (Bona-fide):  Precision: {metrics['precision_bonafide']:.3f} | Recall: {metrics['recall_bonafide']:.3f} | F1: {metrics['f1_bonafide']:.3f}")
    print(f"Fake (Spoof):         Precision: {metrics['precision_spoof']:.3f} | Recall: {metrics['recall_spoof']:.3f} | F1: {metrics['f1_spoof']:.3f}")
    print(f"Macro F1-Score:       {metrics['macro_f1']:.3f}")
    print(f"Confusion Matrix:     TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"ROC-AUC:              {metrics['roc_auc']}")
    print(f"EER (Equal Error Rate): {metrics['eer']} (threshold: {metrics['eer_optimal_threshold']})")
    print("-" * 72)
    print(f"Artifacts saved:")
    print(f"  • CSV:    {csv_path}")
    print(f"  • Image:  {img_path}")
    print(f"  • Report: {report_path}")
    print(f"  • Log:    {log_path}")
    print("=" * 72)

    return metrics


def write_markdown_report(
    report_path: str,
    metrics: dict,
    evaluated_records: list[dict],
    cm: np.ndarray,
) -> None:
    """Generates a comprehensive Markdown evaluation report."""
    tn, fp, fn, tp = cm.ravel()

    content = f"""# 🛡️ SonicShield In-The-Wild Dataset Evaluation Report

**Model:** Pretrained AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)  
**Dataset:** In-The-Wild Audio Deepfake Dataset (Müller et al.)  
**Checkpoint:** `models/pretrained/AASIST.pth` (Official Clova AI weights, evaluated without retraining or fine-tuning)  
**Evaluation Scope:** Balanced 20-sample benchmark (10 genuine human recordings, 10 synthetic voice clones)  

---

## 📊 Summary of Evaluation Metrics

| Metric | Measured Value | Description / Formula |
| :--- | :---: | :--- |
| **Total Recordings** | **{metrics['total_evaluated']}** | 10 Genuine (`bona-fide`), 10 Fake (`spoof`) |
| **Accuracy** | **{metrics['accuracy'] * 100:.1f}%** | Overall classification accuracy: $\\frac{{\\text{{TP}} + \\text{{TN}}}}{{N}}$ |
| **Genuine Precision** | **{metrics['precision_bonafide']:.3f}** | Precision on bona-fide speech: $\\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FP}}}}$ |
| **Genuine Recall** | **{metrics['recall_bonafide']:.3f}** | Recall on bona-fide speech: $\\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}}$ |
| **Genuine F1-Score** | **{metrics['f1_bonafide']:.3f}** | Harmonic mean of genuine precision & recall |
| **Fake Precision** | **{metrics['precision_spoof']:.3f}** | Precision on spoof/clone speech: $\\frac{{\\text{{TN}}}}{{\\text{{TN}} + \\text{{FN}}}}$ |
| **Fake Recall** | **{metrics['recall_spoof']:.3f}** | Recall on spoof/clone speech: $\\frac{{\\text{{TN}}}}{{\\text{{TN}} + \\text{{FP}}}}$ |
| **Fake F1-Score** | **{metrics['f1_spoof']:.3f}** | Harmonic mean of spoof precision & recall |
| **Macro F1-Score** | **{metrics['macro_f1']:.3f}** | Unweighted average F1 across both classes |
| **ROC-AUC** | **{metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}** | Area Under Receiver Operating Characteristic Curve |
| **Equal Error Rate (EER)** | **{metrics['eer'] if metrics['eer'] is not None else 'N/A'}** | Point where False Acceptance Rate equals False Alarm Rate |
| **EER Optimal Threshold** | **{metrics['eer_optimal_threshold'] if metrics['eer_optimal_threshold'] is not None else 'N/A'}** | Continuous bona-fide score threshold corresponding to EER |

---

## 🗂️ Confusion Matrix Breakdown

| Actual \\ Predicted | Predicted Spoof (Fake, 0) | Predicted Bona-fide (Live, 1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Spoof (Fake, 0)** | **{tn}** (True Negative) | **{fp}** (False Positive) | **10** |
| **Actual Bona-fide (Live, 1)** | **{fn}** (False Negative) | **{tp}** (True Positive) | **10** |
| **Total Predicted** | **{tn + fn}** | **{fp + tp}** | **20** |

* **True Negatives ({tn}/10):** Synthetic deepfakes correctly rejected as spoof.
* **False Positives ({fp}/10):** Synthetic deepfakes erroneously accepted as genuine human speech.
* **False Negatives ({fn}/10):** Genuine human speakers erroneously flagged as spoof.
* **True Positives ({tp}/10):** Genuine human speakers correctly identified as bona-fide.

A rendered visualization has been saved to [`confusion_matrix.png`](confusion_matrix.png).

---

## 🔍 Per-Recording Detailed Predictions

| # | File | Speaker | Ground Truth | AASIST Prediction | Bona-fide Score | Liveness Index | Result |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""

    for i, r in enumerate(evaluated_records, start=1):
        status_symbol = "✅ Correct" if r["is_correct"] else "❌ Error"
        content += (
            f"| {i} | `{r['file']}` | {r['speaker']} | `{r['true_label']}` | "
            f"`{r['predicted_label']}` | {r['bona_fide_score']:.4f} | {r['liveness_score']}/100 | {status_symbol} |\n"
        )

    content += """
---

## ⚠️ Key Findings & Limitations

1. **Zero Model Tampering:** Inference used the original, unmodified `AASIST.pth` weights without fine-tuning, adhering strictly to audit protocols.
2. **Domain Shift & Uncalibrated Scores:** The AASIST checkpoint was originally trained on the ASVspoof 2019 Logical Access studio dataset. In-the-wild recordings contain spontaneous background noise, compression artifacts, and varied acoustics that cause distribution shifts.
3. **No General Accuracy Claims:** This evaluation verifies pipeline functionality on a balanced sample ($N=20$). General production detection accuracy cannot be claimed across the full dataset without complete out-of-domain evaluation and probability calibration.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_in_the_wild_evaluation()

