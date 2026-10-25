"""Unit tests for physics_engine module."""
import numpy as np
from src.physics_engine import analyze

def test_physics_engine_empty_window():
    empty = np.array([], dtype=np.float32)
    res = analyze(empty)
    assert 0.0 <= res["score"] <= 1.0
    assert len(res["reasons"]) > 0
    assert "usable" in res["features"]

def test_physics_engine_digital_silence():
    silence = np.zeros(48000, dtype=np.float32)
    res = analyze(silence)
    assert res["score"] <= 0.40
    assert any("silence" in r.lower() for r in res["reasons"])
    assert res["features"]["is_digital_silence"] is True

def test_physics_engine_reverberant_signal():
    sr = 8000
    t = np.linspace(0, 6.0, 48000, endpoint=False)
    # Speech tone with smooth exponential decay tail and room noise
    tone = np.sin(2 * np.pi * 300 * t)
    env = np.zeros_like(t)
    env[:24000] = 0.8
    # 300ms smooth decay
    env[24000:26400] = 0.8 * np.exp(-np.linspace(0, 3, 2400))
    ambient_noise = np.random.normal(0, 0.01, len(t))
    signal = (tone * env + ambient_noise).astype(np.float32)

    res = analyze(signal, sr=sr)
    assert 0.0 <= res["score"] <= 1.0
    assert res["score"] > 0.50
    assert res["features"]["usable"] is True
    assert len(res["reasons"]) > 0
