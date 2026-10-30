"""Unit tests for FastAPI endpoints."""
import os
import io
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
SMOKE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke")

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "supported_codecs" in data
    assert data["supported_codecs"]["clean"] is True
    assert data["supported_codecs"]["g711"] is True
    assert "ml_model" in data
    assert "aasist_model" in data
    assert data["aasist_model"]["architecture"] == "AASIST"
    assert "supported_detection_modes" in data
    assert "pretrained" in data["supported_detection_modes"]
    assert "compare" in data["supported_detection_modes"]

def test_analyze_endpoint_success():
    filepath = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(filepath)

    with open(filepath, "rb") as f:
        response = client.post(
            "/analyze?codec=g711&mode=both&detection_mode=compare",
            files={"file": ("test.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "overall_liveness" in data
    assert 0 <= data["overall_liveness"] <= 100
    assert data["codec"] == "g711"
    assert data["detection_mode"] == "compare"
    assert "overall_baseline_liveness" in data
    assert "overall_aasist_liveness" in data
    assert "aasist" in data
    assert "windows" in data
    assert len(data["windows"]) >= 2
    assert "baseline_score" in data["windows"][0]
    assert "aasist_score" in data["windows"][0]

def test_analyze_endpoint_pretrained_mode():
    filepath = os.path.join(SMOKE_DIR, "smoke_synthetic_like.wav")
    assert os.path.exists(filepath)

    with open(filepath, "rb") as f:
        response = client.post(
            "/analyze?codec=clean&mode=both&detection_mode=pretrained",
            files={"file": ("synth.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["detection_mode"] == "pretrained"
    assert "overall_aasist_class" in data
    assert "overall_aasist_bona_fide_score" in data
    assert data["overall_aasist_class"] in ("bona_fide", "spoof")

def test_analyze_endpoint_empty_file():
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/analyze",
        files={"file": ("empty.wav", empty_file, "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_analyze_endpoint_invalid_extension():
    fake_txt = io.BytesIO(b"hello world")
    response = client.post(
        "/analyze",
        files={"file": ("doc.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"].lower()
