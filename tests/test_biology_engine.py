"""Unit tests for biology_engine module."""
import numpy as np
from src.biology_engine import analyze

def test_biology_engine_empty_window():
    empty = np.array([], dtype=np.float32)
    res = analyze(empty)
    assert 0.0 <= res["score"] <= 1.0
    assert len(res["reasons"]) > 0

def test_biology_engine_silence():
    silence = np.zeros(48000, dtype=np.float32)
    res = analyze(silence)
    assert res["score"] <= 0.40
    assert any("silence" in r.lower() for r in res["reasons"])

def test_biology_engine_vocal_signal():
    sr = 8000
    t = np.linspace(0, 6.0, 48000, endpoint=False)
    # Natural intonation with 130 Hz + 10 Hz vibrato
    f0 = 130.0 + 8.0 * np.sin(2 * np.pi * 3.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    vocal = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    # Pause in middle with simulated breath burst
    env = np.ones_like(t) * 0.8
    env[20000:28000] = 0.0  # Pause 2.5s to 3.5s
    noise = np.random.normal(0, 0.06, len(t))
    breath_mask = (t >= 2.7) & (t <= 3.2)
    signal = (vocal * env + noise * breath_mask).astype(np.float32)

    res = analyze(signal, sr=sr)
    assert 0.0 <= res["score"] <= 1.0
    assert len(res["reasons"]) > 0
    assert res["features"]["usable"] is True
