"""SonicShield Dataset Evaluation Runner."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import numpy as np

from eval.metrics import compute_classification_metrics
from src.pipeline import run_pipeline

def run_evaluation(labels_csv: str, output_csv: str = "eval/results.csv") -> None:
    print("=" * 70)
    print("SONICSHIELD DATASET EVALUATION")
    print("=" * 70)
    print(f"Labels CSV: {labels_csv}")

    if not os.path.exists(labels_csv):
        print(f"\n[INFO] Labels file '{labels_csv}' does not exist.")
        print("To run evaluation on genuine speech datasets:")
        print("  1. Populate 'data/labels.csv' with paths to real and clone recordings.")
        print("  2. See 'docs/DATASET_GUIDE.md' for guidelines and recommended data sources.")
        print("=" * 70)
        return

    # Read records
    records = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} entries from {labels_csv}.")

    # Verify if files exist on disk
    valid_records = []
    base_dir = os.path.dirname(labels_csv)
    for r in records:
        f_rel = r.get("file", "").strip()
        # Check relative to labels file dir or current working dir
        candidates = [
            os.path.join(base_dir, f_rel),
            os.path.join("data", f_rel),
            f_rel,
        ]
        found_path = None
        for c in candidates:
            if os.path.exists(c):
                found_path = c
                break

        if found_path:
            valid_records.append({**r, "resolved_path": found_path})

    print(f"Resolved {len(valid_records)} accessible audio files on disk.")

    if len(valid_records) == 0:
        print("\n[NOTE] No accessible audio recordings found for the paths in the labels file.")
        print("An evaluation template table is displayed below for when real dataset clips are added:")
        print("\n" + "-" * 60)
        print(f"{'Condition':<20} | {'Physics EER':<12} | {'Biology EER':<12} | {'Both EER':<12}")
        print("-" * 60)
        print(f"{'Clean audio':<20} | {'[Pending data]':<12} | {'[Pending data]':<12} | {'[Pending data]':<12}")
        print(f"{'G.711 mu-law':<20} | {'[Pending data]':<12} | {'[Pending data]':<12} | {'[Pending data]':<12}")
        print(f"{'AMR-NB':<20} | {'[Pending data]':<12} | {'[Pending data]':<12} | {'[Pending data]':<12}")
        print("-" * 60)
        print("Honest Reporting: SonicShield does not report synthetic numbers as measured performance.")
        return

    # Execute evaluation on accessible records
    conditions = ["clean", "g711"]
    modes = ["both", "physics", "biology"]

    results_table = []
    start_eval = time.time()

    for cond in conditions:
        for mode in modes:
            y_true = []
            y_scores = []
            latencies = []

            for r in valid_records:
                path = r["resolved_path"]
                lbl = r.get("label", "").lower().strip()
                target_val = 1 if lbl in ("real", "human", "1") else 0

                t0 = time.perf_counter()
                try:
                    res = run_pipeline(path, codec=cond, mode=mode)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    y_true.append(target_val)
                    y_scores.append(res["overall_liveness"])
                except Exception as e:
                    print(f"  [WARN] Skipping {path}: {e}")

            if len(y_true) > 0:
                metrics = compute_classification_metrics(y_true, y_scores)
                avg_lat = round(float(np.mean(latencies)), 1) if latencies else 0.0
                results_table.append({
                    "condition": cond,
                    "mode": mode,
                    "samples": len(y_true),
                    "roc_auc": metrics.get("roc_auc"),
                    "eer": metrics.get("eer"),
                    "accuracy": metrics.get("accuracy"),
                    "mean_latency_ms": avg_lat,
                })

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if results_table:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
            writer.writeheader()
            writer.writerows(results_table)
        print(f"\nSaved evaluation metrics to {output_csv}")

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE ({time.time() - start_eval:.2f}s)")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SonicShield Evaluation Tool")
    parser.add_argument("--labels", default="data/labels_template.csv", help="Path to labels CSV")
    parser.add_argument("--out", default="eval/results.csv", help="Output results CSV")
    args = parser.parse_args()

    run_evaluation(args.labels, args.out)
