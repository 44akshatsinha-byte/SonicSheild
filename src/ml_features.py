"""SonicShield Machine Learning Feature Extractor.

Extracts a fixed, ordered numerical feature vector from Physics and Biology engine outputs.
This module guarantees identical feature ordering and transformation during both training and inference.
"""
from __future__ import annotations

import math
import numpy as np

# Fixed, documented feature schema (18 acoustic features)
FEATURE_NAMES: list[str] = [
    # Physics Engine features (8)
    "phys_score",
    "phys_rms_db",
    "phys_decay_slope",
    "phys_reverb_score",
    "phys_pause_rms_db",
    "phys_spectral_flatness",
    "phys_is_digital_silence",
    "phys_cepstral_peak_ratio",
    # Biology Engine features (10)
    "bio_score",
    "bio_breath_count",
    "bio_breaths_per_min",
    "bio_pitch_mean_hz",
    "bio_pitch_std_hz",
    "bio_tremor_ratio",
    "bio_jitter_local_percent",
    "bio_shimmer_local_percent",
    "bio_hnr_db",
    "bio_voiced_fraction",
]

def _sanitize_val(val: float | int | bool | None) -> float:
    """Converts a scalar to a safe float, returning np.nan if missing, non-finite, or None."""
    if val is None:
        return np.nan
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return np.nan
        return f
    except (ValueError, TypeError):
        return np.nan

def extract_features(
    physics_result: dict,
    biology_result: dict,
    window_audio: np.ndarray | None = None,
    sr: int = 8000,
) -> tuple[np.ndarray, dict[str, float], dict[str, bool]]:
    """Extracts a fixed 1D numerical feature vector from engine results.
    
    Args:
        physics_result: Output dictionary from physics_engine.analyze().
        biology_result: Output dictionary from biology_engine.analyze().
        window_audio: Optional raw audio window for quality checks.
        sr: Sample rate in Hz.
        
    Returns:
        feature_vector: 1D numpy array of float32 of length len(FEATURE_NAMES).
        feature_dict: Dictionary mapping each feature name to its float value.
        quality: Dictionary of quality indicators (usable, is_silent, has_voiced).
    """
    phys_feat = physics_result.get("features", {})
    bio_feat = biology_result.get("features", {})

    raw_dict: dict[str, float | int | bool | None] = {
        # Physics
        "phys_score": physics_result.get("score"),
        "phys_rms_db": phys_feat.get("rms_db"),
        "phys_decay_slope": phys_feat.get("decay_slope_db_ms"),
        "phys_reverb_score": phys_feat.get("reverb_score"),
        "phys_pause_rms_db": phys_feat.get("pause_rms_db"),
        "phys_spectral_flatness": phys_feat.get("spectral_flatness"),
        "phys_is_digital_silence": phys_feat.get("is_digital_silence"),
        "phys_cepstral_peak_ratio": phys_feat.get("cepstral_peak_ratio"),
        # Biology
        "bio_score": biology_result.get("score"),
        "bio_breath_count": bio_feat.get("breath_count"),
        "bio_breaths_per_min": bio_feat.get("breaths_per_min"),
        "bio_pitch_mean_hz": bio_feat.get("pitch_mean_hz"),
        "bio_pitch_std_hz": bio_feat.get("pitch_std_hz"),
        "bio_tremor_ratio": bio_feat.get("tremor_ratio"),
        "bio_jitter_local_percent": bio_feat.get("jitter_local_percent"),
        "bio_shimmer_local_percent": bio_feat.get("shimmer_local_percent"),
        "bio_hnr_db": bio_feat.get("hnr_db"),
        "bio_voiced_fraction": bio_feat.get("voiced_fraction"),
    }

    feature_dict: dict[str, float] = {}
    vec = []
    missing_count = 0

    for name in FEATURE_NAMES:
        cleaned = _sanitize_val(raw_dict.get(name))
        feature_dict[name] = cleaned
        vec.append(cleaned)
        if np.isnan(cleaned):
            missing_count += 1

    feature_vector = np.array(vec, dtype=np.float32)

    # Audio quality indicators
    is_silent = bool(phys_feat.get("is_digital_silence", False))
    if window_audio is not None and len(window_audio) > 0:
        rms = float(np.sqrt(np.mean(window_audio**2)))
        if rms < 1e-4:
            is_silent = True

    has_voiced = (
        bio_feat.get("voiced_fraction") is not None
        and not np.isnan(bio_feat.get("voiced_fraction", 0.0))
        and bio_feat.get("voiced_fraction", 0.0) > 0.05
    )

    phys_usable = bool(phys_feat.get("usable", True))
    bio_usable = bool(bio_feat.get("usable", True))
    is_usable = phys_usable and bio_usable and not is_silent

    quality = {
        "usable": is_usable,
        "is_silent": is_silent,
        "has_voiced": bool(has_voiced),
        "missing_features_count": missing_count,
    }

    return feature_vector, feature_dict, quality
