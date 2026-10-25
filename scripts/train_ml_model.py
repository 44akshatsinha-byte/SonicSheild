"""SonicShield Supervised Machine Learning Model Training Script.

Trains a genuine acoustic classifier on speaker-separated real/fake speech datasets.
Prevents data leakage by strictly partitioning speakers across train, validation, and test sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
import joblib
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_utils import load_audio
from src.windows import generate_windows
from src.physics_engine import analyze as analyze_physics
from src.biology_engine import analyze as analyze_biology
from src.ml_features import FEATURE_NAMES, extract_features

# Label normalization mapping
LABEL_MAP = {
    "real": 1,
    "live": 1,
    "human": 1,
    "genuine": 1,
    "1": 1,
    "fake": 0,
    "clone": 0,
    "synthetic": 0,
    "spoof": 0,
    "0": 0,
}

def load_and_validate_manifest(manifest_path: str) -> tuple[list[dict], list[str]]:
    """Loads and validates dataset records from manifest CSV.
    
    Returns:
        valid_records: List of validated record dicts with resolved audio file paths.
        issues: List of validation warnings and errors.
    """
    issues = []
    if not os.path.exists(manifest_path):
        return [], [f"Manifest file '{manifest_path}' does not exist."]

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            records.append((idx, row))

    if not records:
        return [], [f"Manifest file '{manifest_path}' contains no data rows."]

    valid_records = []
    seen_files = set()
    base_dir = os.path.dirname(manifest_path)

    for line_num, row in records:
        raw_file = row.get("file", "").strip()
        raw_label = row.get("label", "").lower().strip()
        raw_speaker = row.get("speaker", "").strip()

        if not raw_file:
            issues.append(f"Line {line_num}: Empty 'file' field.")
            continue

        if raw_label not in LABEL_MAP:
            issues.append(f"Line {line_num}: Unrecognized label '{raw_label}'. Must be real/live or fake/clone.")
            continue

        if not raw_speaker:
            issues.append(f"Line {line_num}: Missing 'speaker' ID. Speaker ID is mandatory to prevent data leakage.")
            continue

        # Resolve file path
        candidates = [
            os.path.join(base_dir, raw_file),
            os.path.join("data", raw_file),
            raw_file,
        ]
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break

        if not resolved:
            issues.append(f"Line {line_num}: Audio file '{raw_file}' does not exist on disk.")
            continue

        if resolved in seen_files:
            issues.append(f"Line {line_num}: Duplicate file entry '{resolved}'.")
            continue
        seen_files.add(resolved)

        valid_records.append({
            "line": line_num,
            "path": resolved,
            "label": LABEL_MAP[raw_label],
            "speaker": raw_speaker,
            "source": row.get("source", "unknown"),
            "codec": row.get("codec", "clean"),
            "room": row.get("room", "unknown"),
            "attack_type": row.get("attack_type", "none"),
        })

    return valid_records, issues

def create_speaker_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Partitions records by unique speaker ID to prevent speaker/window leakage.
    
    Guarantees that no speaker appears in both train and test.
    """
    rng = np.random.RandomState(random_seed)

    # Separate speakers by dominant class
    speaker_labels: dict[str, list[int]] = {}
    for r in records:
        spk = r["speaker"]
        speaker_labels.setdefault(spk, []).append(r["label"])

    real_speakers = [s for s, lbls in speaker_labels.items() if np.mean(lbls) >= 0.5]
    fake_speakers = [s for s, lbls in speaker_labels.items() if np.mean(lbls) < 0.5]

    rng.shuffle(real_speakers)
    rng.shuffle(fake_speakers)

    def split_list(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        if n == 0:
            return [], [], []
        if n == 1:
            # Single speaker cannot be split into 3 sets
            return items, [], []
        if n == 2:
            return [items[0]], [], [items[1]]
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]
        if not test_items and len(val_items) > 1:
            test_items = [val_items.pop()]
        return train_items, val_items, test_items

    r_train, r_val, r_test = split_list(real_speakers)
    f_train, f_val, f_test = split_list(fake_speakers)

    train_spks = set(r_train + f_train)
    val_spks = set(r_val + f_val)
    test_spks = set(r_test + f_test)

    train_records = [r for r in records if r["speaker"] in train_spks]
    val_records = [r for r in records if r["speaker"] in val_spks]
    test_records = [r for r in records if r["speaker"] in test_spks]

    split_meta = {
        "train_speakers": sorted(list(train_spks)),
        "val_speakers": sorted(list(val_spks)),
        "test_speakers": sorted(list(test_spks)),
        "num_real_speakers": len(real_speakers),
        "num_fake_speakers": len(fake_speakers),
    }

    return train_records, val_records, test_records, split_meta

def extract_features_from_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Processes audio files into 6s rolling windows and extracts feature vectors."""
    X_list = []
    y_list = []
    speakers_list = []

    for r in records:
        path = r["path"]
        lbl = r["label"]
        spk = r["speaker"]

        try:
            audio, sr, _ = load_audio(path, target_sr=8000)
            windows, _ = generate_windows(audio, sr=sr)
            for w in windows:
                p_res = analyze_physics(w.audio, sr=sr)
                b_res = analyze_biology(w.audio, sr=sr)
                feat_vec, _, quality = extract_features(p_res, b_res, window_audio=w.audio, sr=sr)
                X_list.append(feat_vec)
                y_list.append(lbl)
                speakers_list.append(spk)
        except Exception as e:
            print(f"  [WARN] Failed extracting features from {path}: {e}")

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=int), []

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), speakers_list

def train_and_evaluate(
    manifest_path: str,
    output_model_path: str = "models/sonicshield_ml.joblib",
    output_meta_path: str = "models/sonicshield_ml_meta.json",
) -> bool:
    print("=" * 70)
    print("SONICSHIELD MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)
    print(f"Dataset Manifest: {manifest_path}")

    # 1. Dataset Validation
    records, issues = load_and_validate_manifest(manifest_path)
    if issues:
        print("\n[Dataset Validation Report]")
        for iss in issues[:10]:
            print(f"  - {iss}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues.")

    if not records:
        print("\n[ERROR] No valid, readable audio records found in the manifest.")
        print("To train a genuine ML model:")
        print("  1. Add real and clone audio WAV files to data/raw/real/ and data/raw/fake/.")
        print("  2. Index them in data/labels.csv using columns: file,label,speaker,source,codec,room,attack_type.")
        print("  3. Run: python scripts/train_ml_model.py --manifest data/labels.csv")
        print("\nPreserving rule-based baseline operation.")
        return False

    labels_present = set(r["label"] for r in records)
    if len(labels_present) < 2:
        print(f"\n[ERROR] Dataset contains only 1 class ({labels_present}). Both real and fake classes are required.")
        return False

    # Check minimum speaker diversity
    speakers = set(r["speaker"] for r in records)
    if len(speakers) < 3:
        print(f"\n[ERROR] Dataset contains only {len(speakers)} distinct speaker(s).")
        print("A minimum of 3 independent speakers is required to create non-leaking train/val/test splits.")
        return False

    print(f"\nValidated {len(records)} audio files across {len(speakers)} distinct speakers.")

    # 2. Speaker-Separated Splitting
    train_recs, val_recs, test_recs, split_meta = create_speaker_split(records)
    print(f"Split distribution:")
    print(f"  Train: {len(train_recs)} files ({len(split_meta['train_speakers'])} speakers)")
    print(f"  Val:   {len(val_recs)} files ({len(split_meta['val_speakers'])} speakers)")
    print(f"  Test:  {len(test_recs)} files ({len(split_meta['test_speakers'])} speakers)")

    if not train_recs or not val_recs:
        print("\n[ERROR] Insufficient data to form separate train and validation splits.")
        return False

    # 3. Extract Features using exact same inference extractor
    print("\nExtracting acoustic features from rolling windows...")
    X_train, y_train, _ = extract_features_from_records(train_recs)
    X_val, y_val, _ = extract_features_from_records(val_recs)
    X_test, y_test, _ = extract_features_from_records(test_recs)

    print(f"Extracted feature windows: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    if len(np.unique(y_train)) < 2:
        print("\n[ERROR] Training split does not contain both classes.")
        return False

    # 4. Model Candidates
    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=42, n_estimators=60, max_depth=3)),
        ]),
    }

    best_name = None
    best_model = None
    best_val_score = -1.0

    print("\n[Model Training & Validation Tuning]")
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        try:
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = val_acc

        print(f"  {name:20s} -> Val Accuracy: {val_acc:.4f} | Val ROC-AUC: {val_auc:.4f}")

        # Choose best based on validation performance
        if val_auc > best_val_score:
            best_val_score = val_auc
            best_name = name
            best_model = pipeline

    print(f"\nSelected Model: {best_name} (Val Score: {best_val_score:.4f})")

    # 5. Final Evaluation on Held-out Test Set
    test_acc = 0.0
    test_auc = 0.0
    if len(X_test) > 0 and len(np.unique(y_test)) >= 2:
        test_preds = best_model.predict(X_test)
        test_acc = float(accuracy_score(y_test, test_preds))
        try:
            test_probs = best_model.predict_proba(X_test)[:, 1]
            test_auc = float(roc_auc_score(y_test, test_probs))
        except Exception:
            test_auc = test_acc
        print(f"Held-out Test Evaluation -> Accuracy: {test_acc:.4f} | ROC-AUC: {test_auc:.4f}")
    else:
        print("Held-out Test Set has insufficient class diversity for standalone metric reporting.")

    # 6. Save Model and Metadata
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(best_model, output_model_path)

    metadata = {
        "model_type": best_name,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes": [int(c) for c in best_model.classes_],
        "class_mapping": {"0": "fake", "1": "real"},
        "split_summary": split_meta,
        "metrics": {
            "val_roc_auc": round(best_val_score, 4),
            "test_accuracy": round(test_acc, 4),
            "test_roc_auc": round(test_auc, 4),
        },
        "training_samples_windows": len(X_train),
        "validation_samples_windows": len(X_val),
        "test_samples_windows": len(X_test),
    }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to:    {output_model_path}")
    print(f"Saved metadata to: {output_meta_path}")
    print("=" * 70)
    print("TRAINING SUCCESSFUL")
    print("=" * 70)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SonicShield ML Classifier")
    parser.add_argument("--manifest", default="data/labels.csv", help="Path to dataset manifest CSV")
    parser.add_argument("--out-model", default="models/sonicshield_ml.joblib", help="Target model path")
    parser.add_argument("--out-meta", default="models/sonicshield_ml_meta.json", help="Target metadata JSON path")
    args = parser.parse_args()

    success = train_and_evaluate(args.manifest, args.out_model, args.out_meta)
    if not success:
        sys.exit(1)
