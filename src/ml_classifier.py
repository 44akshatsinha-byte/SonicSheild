"""SonicShield Machine Learning Classifier Inference Engine."""
from __future__ import annotations

import json
import os
import joblib
import numpy as np

from src.ml_features import FEATURE_NAMES
from src.scorer import get_band, get_band_label

DEFAULT_MODEL_PATH = os.path.join("models", "sonicshield_ml.joblib")
DEFAULT_META_PATH = os.path.join("models", "sonicshield_ml_meta.json")

class SonicShieldMLClassifier:
    """Manages loading, feature schema validation, and inference for the trained ML model."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        meta_path: str = DEFAULT_META_PATH,
    ):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.metadata: dict = {}
        self.is_loaded = False
        self.load_status = "unloaded"
        self.load_error: str | None = None
        self.real_class_idx: int = 1

        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load the trained model pipeline and metadata from disk."""
        if not os.path.exists(self.model_path):
            self.load_status = "model_not_found"
            self.load_error = f"Model file not found at {self.model_path}."
            return

        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            self.load_status = "load_error"
            self.load_error = f"Failed loading joblib model: {e}"
            return

        # Load metadata if present
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

        # Validate feature schema if metadata recorded it
        expected_features = self.metadata.get("feature_names")
        if expected_features and expected_features != FEATURE_NAMES:
            self.load_status = "schema_mismatch"
            self.load_error = (
                f"Model was trained with different feature schema. "
                f"Expected {len(expected_features)} features, current has {len(FEATURE_NAMES)}."
            )
            self.model = None
            return

        # Determine index of the 'real' (positive, 1) class in model.classes_
        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            if 1 in classes:
                self.real_class_idx = classes.index(1)
            elif "real" in classes:
                self.real_class_idx = classes.index("real")
            elif len(classes) == 2:
                # Default assume second column is positive
                self.real_class_idx = 1
            else:
                self.real_class_idx = 0

        self.is_loaded = True
        self.load_status = "active"
        self.load_error = None

    def reload(self) -> None:
        """Reloads the model from disk (useful after retraining)."""
        self._load_model()

    def predict_window(self, feature_vector: np.ndarray) -> dict:
        """Performs ML inference on a single rolling window feature vector.
        
        Args:
            feature_vector: 1D numpy array of length len(FEATURE_NAMES).
            
        Returns:
            Dictionary containing ml_score (0-100), probability, predicted_label, and model status.
        """
        if not self.is_loaded or self.model is None:
            return {
                "ml_score": None,
                "raw_prob": None,
                "predicted_label": None,
                "band": None,
                "band_label": None,
                "is_available": False,
                "status": self.load_status,
                "model_type": None,
                "warning": self.load_error or "ML model is not loaded. Run 'python scripts/train_ml_model.py' to train.",
            }

        # Check dimension
        feat_1d = np.asarray(feature_vector, dtype=np.float32).ravel()
        if len(feat_1d) != len(FEATURE_NAMES):
            return {
                "ml_score": None,
                "raw_prob": None,
                "predicted_label": None,
                "band": None,
                "band_label": None,
                "is_available": False,
                "status": "invalid_dimensions",
                "model_type": self.metadata.get("model_type"),
                "warning": f"Expected {len(FEATURE_NAMES)} features, got {len(feat_1d)}.",
            }

        feat_2d = feat_1d.reshape(1, -1)

        try:
            # Predict probabilities
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(feat_2d)[0]
                prob_real = float(probas[self.real_class_idx])
            else:
                # Decision function / linear prediction
                pred = int(self.model.predict(feat_2d)[0])
                prob_real = 1.0 if pred == 1 else 0.0

            predicted_label = int(self.model.predict(feat_2d)[0])
            ml_score = int(np.clip(round(prob_real * 100.0), 0, 100))
            band = get_band(ml_score)
            band_label = get_band_label(band)

            return {
                "ml_score": ml_score,
                "raw_prob": round(prob_real, 4),
                "predicted_label": predicted_label,
                "band": band,
                "band_label": band_label,
                "is_available": True,
                "status": "active",
                "model_type": self.metadata.get("model_type", "LogisticRegression"),
                "warning": None,
            }

        except Exception as e:
            return {
                "ml_score": None,
                "raw_prob": None,
                "predicted_label": None,
                "band": None,
                "band_label": None,
                "is_available": False,
                "status": "inference_error",
                "model_type": self.metadata.get("model_type"),
                "warning": f"Prediction failed: {e}",
            }

# Module-level cached instance
_CACHED_CLASSIFIER: SonicShieldMLClassifier | None = None

def get_ml_classifier(reload: bool = False) -> SonicShieldMLClassifier:
    """Returns the cached SonicShieldMLClassifier instance."""
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None or reload:
        _CACHED_CLASSIFIER = SonicShieldMLClassifier()
    return _CACHED_CLASSIFIER
