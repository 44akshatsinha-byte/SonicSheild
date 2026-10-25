"""Evaluation metrics calculation for SonicShield."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Calculates Equal Error Rate (EER) and the optimal threshold.
    
    Args:
        y_true: Ground truth binary labels (1 = real/live, 0 = fake/clone).
        y_scores: Continuous decision scores (e.g. 0-100 or 0-1).
        
    Returns:
        eer: Equal Error Rate in [0.0, 1.0].
        threshold: Decision threshold where False Positive Rate == False Negative Rate.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    # In our definition: score higher means real (class 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where |FPR - FNR| is minimal
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx])

    return eer, thresh

def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    threshold: float = 50.0,
) -> dict:
    """Computes comprehensive binary detection metrics at a specified decision threshold.
    
    Args:
        y_true: Ground truth labels (1 = real, 0 = fake).
        y_scores: Continuous scores (0 to 100).
        threshold: Score threshold above which prediction is labeled real (1).
        
    Returns:
        Dictionary of measured metrics.
    """
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    if len(y_true_arr) == 0:
        return {"error": "Empty dataset provided."}

    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) < 2:
        return {
            "error": "Dataset must contain both positive (real) and negative (fake) samples to calculate metrics.",
            "total_samples": len(y_true_arr),
            "present_labels": unique_labels.tolist(),
        }

    y_pred = (y_scores_arr >= threshold).astype(int)

    # Confusion matrix
    # Format: tn, fp, fn, tp (where 0 is fake, 1 is real)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    # Rates
    # False Alarm Rate (FAR on real callers = real rejected as fake): FN / (TP + FN)
    false_alarm_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    # False Accept Rate (FAR on fake callers = fake accepted as real): FP / (TN + FP)
    false_accept_rate = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Accuracy
    accuracy = float((tp + tn) / len(y_true_arr))

    # ROC-AUC
    try:
        auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except Exception:
        auc = float("nan")

    # EER
    eer, eer_thresh = compute_eer(y_true_arr, y_scores_arr)

    return {
        "threshold": threshold,
        "total_samples": int(len(y_true_arr)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "accuracy": round(accuracy, 4),
        "false_alarm_rate_on_real": round(false_alarm_rate, 4),
        "false_accept_rate_on_fakes": round(false_accept_rate, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "eer": round(eer, 4) if not np.isnan(eer) else None,
        "eer_optimal_threshold": round(eer_thresh, 2) if not np.isnan(eer_thresh) else None,
    }
