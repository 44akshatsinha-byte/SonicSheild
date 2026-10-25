"""Unit tests for the end-to-end SonicShield pipeline."""
import os
import pytest
from src.pipeline import run_pipeline, PipelineError

SMOKE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke")

def test_pipeline_smoke_human_clean():
    filepath = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(filepath), "Smoke human file must exist"

    result = run_pipeline(filepath, codec="clean", mode="both")
    assert "overall_liveness" in result
    assert 0 <= result["overall_liveness"] <= 100
    assert result["overall_band"] in ("green", "amber", "red")
    assert result["codec"] == "clean"
    assert result["duration_sec"] == 7.0
    assert result["num_windows"] >= 2
    assert len(result["windows"]) == result["num_windows"]
    assert "features" in result["windows"][0]

def test_pipeline_smoke_g711():
    filepath = os.path.join(SMOKE_DIR, "smoke_synthetic_like.wav")
    assert os.path.exists(filepath)

    result = run_pipeline(filepath, codec="g711", mode="both")
    assert result["codec"] == "g711"
    assert len(result["windows"]) >= 2
    # Synthetic smoke has flat pitch, digital silence, sharp cutoffs -> should be lower score
    assert result["overall_liveness"] <= 60

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
