"""Unit tests for the Pretrained AASIST Anti-Spoofing Model integration."""
from __future__ import annotations

import os
import numpy as np
import pytest

from src.pretrained_aasist import (
    pad_waveform,
    PretrainedAASIST,
    get_aasist_classifier,
    AASIST_NB_SAMPLES,
    AASIST_SAMPLE_RATE,
)
from src.pipeline import run_pipeline

def test_pad_waveform_short():
    """Verify that waveforms shorter than max_len are tile-padded to exact length."""
    short_audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    padded = pad_waveform(short_audio, max_len=10)
    assert len(padded) == 10
    # Expected cyclic tiling: [0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1]
    assert np.isclose(padded[0], 0.1)
    assert np.isclose(padded[3], 0.1)
    assert np.isclose(padded[9], 0.1)

def test_pad_waveform_exact():
    """Verify that waveforms of exact max_len remain unchanged."""
    audio = np.random.randn(AASIST_NB_SAMPLES).astype(np.float32)
    padded = pad_waveform(audio, max_len=AASIST_NB_SAMPLES)
    assert len(padded) == AASIST_NB_SAMPLES
    assert np.array_equal(audio, padded)

def test_pad_waveform_long():
    """Verify that waveforms longer than max_len are truncated to exactly max_len."""
    audio = np.random.randn(AASIST_NB_SAMPLES + 5000).astype(np.float32)
    padded = pad_waveform(audio, max_len=AASIST_NB_SAMPLES)
    assert len(padded) == AASIST_NB_SAMPLES
    assert np.array_equal(padded, audio[:AASIST_NB_SAMPLES])

def test_prepare_audio_for_aasist():
    """Verify audio preprocessing from 8 kHz to 16 kHz with 64,600 samples."""
    clf = PretrainedAASIST(checkpoint_path="nonexistent.pth")
    
    # 2 seconds of 8000 Hz sine wave
    t = np.linspace(0, 2.0, 16000, endpoint=False)
    audio_8k = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    prepared, warnings = clf.prepare_audio_for_aasist(audio_8k, sr=8000)
    assert prepared.shape == (AASIST_NB_SAMPLES,)
    assert prepared.dtype == np.float32
    assert any("tile-padding" in w for w in warnings)

def test_prepare_audio_silent():
    """Verify that digital silence produces a warning."""
    clf = PretrainedAASIST(checkpoint_path="nonexistent.pth")
    silent_audio = np.zeros(16000, dtype=np.float32)
    prepared, warnings = clf.prepare_audio_for_aasist(silent_audio, sr=8000)
    assert prepared.shape == (AASIST_NB_SAMPLES,)
    assert any("silent" in w.lower() for w in warnings)

def test_aasist_missing_checkpoint():
    """Verify graceful handling when checkpoint does not exist."""
    clf = PretrainedAASIST(checkpoint_path="models/pretrained/does_not_exist.pth")
    assert not clf.is_loaded
    assert clf.load_status == "checkpoint_not_found"

    res = clf.predict(np.zeros(8000, dtype=np.float32), sr=8000, codec="clean")
    assert res["model"] == "AASIST"
    assert res["model_loaded"] is False
    assert res["predicted_class"] == "unavailable"
    assert res["bona_fide_score"] is None

def test_aasist_real_checkpoint_inference():
    """Verify genuine AASIST inference with the downloaded checkpoint."""
    clf = get_aasist_classifier()
    if not clf.is_loaded:
        pytest.skip("AASIST checkpoint is not loaded.")

    assert clf.is_loaded is True
    assert clf.load_status == "active"

    # Test synthetic human-like sample
    smoke_wav = os.path.join("data", "smoke", "smoke_human_like.wav")
    if os.path.exists(smoke_wav):
        import soundfile as sf
        audio, sr = sf.read(smoke_wav, dtype="float32")
        res = clf.predict(audio, sr=sr, codec="clean")
        assert res["model"] == "AASIST"
        assert res["model_loaded"] is True
        assert res["predicted_class"] in ("bona_fide", "spoof")
        assert 0.0 <= res["bona_fide_score"] <= 1.0
        assert 0 <= res["liveness_score"] <= 100
        assert isinstance(res["raw_score"], list)
        assert len(res["raw_score"]) == 2
        assert res["codec"] == "clean"
        assert res["status"] == "success"

def test_pipeline_with_aasist_pretrained_mode():
    """Verify that run_pipeline supports detection_mode='pretrained'."""
    smoke_wav = os.path.join("data", "smoke", "smoke_human_like.wav")
    if not os.path.exists(smoke_wav):
        pytest.skip("Smoke sample not found.")

    res = run_pipeline(smoke_wav, detection_mode="pretrained")
    assert res["detection_mode"] == "pretrained"
    assert "overall_aasist_liveness" in res
    assert "overall_aasist_class" in res
    assert "aasist" in res
    assert res["overall_aasist_class"] in ("bona_fide", "spoof", "unavailable")

    # In pretrained mode, overall_liveness reflects AASIST when loaded
    if res["aasist_model_loaded"]:
        assert res["overall_liveness"] == res["overall_aasist_liveness"]

def test_pipeline_with_aasist_compare_mode():
    """Verify that run_pipeline supports detection_mode='compare'."""
    smoke_wav = os.path.join("data", "smoke", "smoke_human_like.wav")
    if not os.path.exists(smoke_wav):
        pytest.skip("Smoke sample not found.")

    res = run_pipeline(smoke_wav, detection_mode="compare")
    assert res["detection_mode"] == "compare"
    assert "overall_baseline_liveness" in res
    assert "overall_aasist_liveness" in res
    assert "overall_ml_liveness" in res
    assert res["overall_liveness"] == res["overall_baseline_liveness"]
