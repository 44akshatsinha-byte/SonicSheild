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

def test_analyze_endpoint_success():
    filepath = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(filepath)

    with open(filepath, "rb") as f:
        response = client.post(
            "/analyze?codec=g711&mode=both",
            files={"file": ("test.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "overall_liveness" in data
    assert 0 <= data["overall_liveness"] <= 100
    assert data["codec"] == "g711"
    assert "windows" in data
    assert len(data["windows"]) >= 2

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
