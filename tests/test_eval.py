"""Unit tests for evaluation metrics and evaluator."""
import numpy as np
from eval.metrics import compute_eer, compute_classification_metrics
from eval.evaluate import run_evaluation

def test_compute_eer_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([10.0, 20.0, 30.0, 70.0, 80.0, 90.0])
    eer, thresh = compute_eer(y_true, y_scores)
    assert eer == 0.0

def test_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_scores = [20.0, 35.0, 75.0, 85.0]
    metrics = compute_classification_metrics(y_true, y_scores, threshold=50.0)
    assert metrics["accuracy"] == 1.0
    assert metrics["false_alarm_rate_on_real"] == 0.0
    assert metrics["false_accept_rate_on_fakes"] == 0.0
    assert metrics["roc_auc"] == 1.0

def test_evaluate_script_nonexistent_labels():
    # Should run gracefully without crashing and display instructions
    run_evaluation("data/nonexistent_labels.csv")
