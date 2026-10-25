"""Unit tests for ML classifier inference engine."""
import os
import tempfile
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.ml_features import FEATURE_NAMES
from src.ml_classifier import SonicShieldMLClassifier

def test_ml_classifier_missing_model():
    clf = SonicShieldMLClassifier(model_path="models/nonexistent_model.joblib")
    assert not clf.is_loaded
    assert clf.load_status == "model_not_found"

    res = clf.predict_window(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
    assert res["is_available"] is False
    assert res["ml_score"] is None
    assert "not found" in res["warning"].lower()

def test_ml_classifier_with_mock_trained_pipeline():
    # Create temporary fitted pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        meta_path = os.path.join(tmpdir, "meta.json")

        # Create dummy separable 2-class data
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.normal(loc=0.0, scale=1.0, size=(20, len(FEATURE_NAMES))),
            rng.normal(loc=2.0, scale=1.0, size=(20, len(FEATURE_NAMES))),
        ])
        y = np.array([0] * 20 + [1] * 20)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, model_path)

        metadata = {
            "model_type": "LogisticRegression",
            "feature_names": FEATURE_NAMES,
            "classes": [0, 1],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Instantiate classifier with this test model
        clf = SonicShieldMLClassifier(model_path=model_path, meta_path=meta_path)
        assert clf.is_loaded
        assert clf.load_status == "active"
        assert clf.real_class_idx == 1

        # Test prediction on positive-like sample
        test_vec = np.ones(len(FEATURE_NAMES), dtype=np.float32) * 2.5
        pred = clf.predict_window(test_vec)

        assert pred["is_available"] is True
        assert 0 <= pred["ml_score"] <= 100
        assert pred["predicted_label"] in (0, 1)
        assert pred["band"] in ("green", "amber", "red")
        assert pred["status"] == "active"

def test_ml_classifier_schema_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        meta_path = os.path.join(tmpdir, "meta.json")

        pipe = Pipeline([("clf", LogisticRegression())])
        pipe.fit([[0, 0], [1, 1]], [0, 1])
        joblib.dump(pipe, model_path)

        # Write metadata with incompatible feature list
        metadata = {
            "model_type": "LogisticRegression",
            "feature_names": ["feature_a", "feature_b"],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        clf = SonicShieldMLClassifier(model_path=model_path, meta_path=meta_path)
        assert not clf.is_loaded
        assert clf.load_status == "schema_mismatch"
