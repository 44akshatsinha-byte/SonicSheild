"""Unit tests for ML feature extraction module."""
import numpy as np
from src.ml_features import FEATURE_NAMES, extract_features

def test_feature_schema_consistency():
    assert len(FEATURE_NAMES) == 18
    assert "phys_score" in FEATURE_NAMES
    assert "phys_decay_slope" in FEATURE_NAMES
    assert "bio_score" in FEATURE_NAMES
    assert "bio_breath_count" in FEATURE_NAMES
    assert "bio_pitch_mean_hz" in FEATURE_NAMES
    assert "bio_jitter_local_percent" in FEATURE_NAMES
    assert "bio_shimmer_local_percent" in FEATURE_NAMES

def test_extract_features_valid():
    phys = {
        "score": 0.85,
        "features": {
            "usable": True,
            "rms_db": -24.5,
            "decay_slope_db_ms": -0.12,
            "reverb_score": 0.88,
            "pause_rms_db": -42.0,
            "spectral_flatness": 0.35,
            "is_digital_silence": False,
            "cepstral_peak_ratio": 2.8,
        },
    }
    bio = {
        "score": 0.80,
        "features": {
            "usable": True,
            "breath_count": 2,
            "breaths_per_min": 20.0,
            "pitch_mean_hz": 145.2,
            "pitch_std_hz": 22.4,
            "tremor_ratio": 0.025,
            "jitter_local_percent": 1.15,
            "shimmer_local_percent": 3.8,
            "hnr_db": 15.6,
            "voiced_fraction": 0.65,
        },
    }

    vec, feat_dict, quality = extract_features(phys, bio)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (18,)
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec))  # All features present

    assert feat_dict["phys_score"] == 0.85
    assert feat_dict["phys_is_digital_silence"] == 0.0
    assert feat_dict["bio_breath_count"] == 2.0
    assert quality["usable"] is True
    assert quality["is_silent"] is False
    assert quality["has_voiced"] is True

def test_extract_features_missing_and_nonfinite():
    # Simulate unvoiced / whisper where pitch is None, and some inf/nan values
    phys = {
        "score": float("nan"),
        "features": {
            "rms_db": float("-inf"),
            "decay_slope_db_ms": None,
            "reverb_score": 0.5,
            "pause_rms_db": -80.0,
            "spectral_flatness": None,
            "is_digital_silence": True,
            "cepstral_peak_ratio": float("inf"),
        },
    }
    bio = {
        "score": 0.3,
        "features": {
            "breath_count": 0,
            "breaths_per_min": 0.0,
            "pitch_mean_hz": None,
            "pitch_std_hz": None,
            "tremor_ratio": None,
            "jitter_local_percent": None,
            "shimmer_local_percent": None,
            "hnr_db": None,
            "voiced_fraction": 0.0,
        },
    }

    vec, feat_dict, quality = extract_features(phys, bio)

    assert len(vec) == 18
    # Must not contain inf or crash
    for val in vec:
        assert not np.isinf(val)
    # Check that missing features are np.nan
    assert np.isnan(feat_dict["bio_pitch_mean_hz"])
    assert np.isnan(feat_dict["phys_decay_slope"])
    assert feat_dict["phys_is_digital_silence"] == 1.0
    assert quality["is_silent"] is True
    assert quality["has_voiced"] is False
