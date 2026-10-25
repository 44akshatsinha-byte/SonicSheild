"""Unit tests for the end-to-end SonicShield pipeline."""
import os
import pytest
from src.pipeline import run_pipeline, PipelineError

SMOKE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke")

def test_pipeline_smoke_human_clean():
    filepath = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(filepath), "Smoke human file must exist"

    result = run_pipeline(filepath, codec="clean", mode="both", detection_mode="baseline")
    assert "overall_liveness" in result
    assert 0 <= result["overall_liveness"] <= 100
    assert result["overall_band"] in ("green", "amber", "red")
    assert result["detection_mode"] == "baseline"
    assert result["codec"] == "clean"
    assert result["duration_sec"] == 7.0
    assert result["num_windows"] >= 2
    assert len(result["windows"]) == result["num_windows"]
    assert "features" in result["windows"][0]
    assert "ml_vector" in result["windows"][0]["features"]

def test_pipeline_smoke_g711():
    filepath = os.path.join(SMOKE_DIR, "smoke_synthetic_like.wav")
    assert os.path.exists(filepath)

    result = run_pipeline(filepath, codec="g711", mode="both")
    assert result["codec"] == "g711"
    assert len(result["windows"]) >= 2
    # Synthetic smoke has flat pitch, digital silence, sharp cutoffs -> should be lower score
    assert result["overall_liveness"] <= 60

def test_pipeline_detection_modes():
    filepath = os.path.join(SMOKE_DIR, "smoke_human_like.wav")

    # Compare mode
    res_compare = run_pipeline(filepath, detection_mode="compare")
    assert res_compare["detection_mode"] == "compare"
    assert "overall_baseline_liveness" in res_compare
    assert "overall_ml_liveness" in res_compare

    # ML mode with no model file (should gracefully fall back to baseline with warning)
    res_ml = run_pipeline(filepath, detection_mode="ml")
    assert res_ml["detection_mode"] == "ml"
    assert res_ml["overall_liveness"] is not None

def test_pipeline_smoke_short():
    filepath = os.path.join(SMOKE_DIR, "smoke_short.wav")
    assert os.path.exists(filepath)

    result = run_pipeline(filepath, codec="clean")
    assert result["duration_sec"] == 2.0
    assert result["num_windows"] == 1
    assert any("shorter" in w for w in result["warnings"])

def test_pipeline_smoke_silence():
    filepath = os.path.join(SMOKE_DIR, "smoke_silence.wav")
    assert os.path.exists(filepath)

    result = run_pipeline(filepath, codec="clean")
    assert result["overall_band"] in ("amber", "red")
    assert any("silent" in w.lower() for w in result["warnings"])
