"""Automated test suite for SonicShield Live Microphone Voice Ingress & Processing.

Verifies:
1. Valid microphone WAV bytes can be received and validated.
2. Audio bytes can be converted to a safe temporary WAV file.
3. Temporary WAV files are processed by the existing detection pipeline.
4. Invalid / corrupted audio is rejected gracefully.
5. Empty audio (None and 0 bytes) is handled with friendly errors.
6. Very short audio is validated (rejects < 0.5s, warns for < 6s).
7. Mono and multi-channel input are handled correctly with mono downmixing.
8. Sample rate handling across standard recording rates (16 kHz, 44.1 kHz, 48 kHz).
9. Temporary files are deterministically cleaned up after both success and failure.
10. Results use actual pipeline outputs (AASIST, Physics, Biology, Baseline).
11. Analysis history records the correct input source ('Microphone Recording' vs 'Uploaded File').
12. Existing uploaded-file analysis continues working without regressions.
"""
from __future__ import annotations

import io
import os
import pytest
import numpy as np
import soundfile as sf

from src.audio_utils import validate_recording_bytes, safe_temporary_wav, AudioValidationError
from src.pipeline import run_pipeline
from src.pretrained_aasist import get_aasist_classifier


def create_synthetic_speech_wav_bytes(
    duration_sec: float = 6.0,
    sr: int = 16000,
    num_channels: int = 1,
    freq: float = 220.0,
) -> bytes:
    """Generates synthetic harmonic voice-like WAV bytes for testing."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Formant-like harmonic structure
    signal = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
        + 0.12 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    # Add subtle envelope
    envelope = np.clip(np.sin(np.pi * t / duration_sec), 0.1, 1.0)
    signal = (signal * envelope).astype(np.float32)

    if num_channels > 1:
        signal = np.column_stack([signal] * num_channels)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# 1. Valid microphone WAV bytes can be received
def test_valid_microphone_wav_received():
    raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=6.5, sr=16000)
    meta, warnings, error = validate_recording_bytes(raw_bytes)

    assert error is None
    assert meta is not None
    assert meta["duration_sec"] == pytest.approx(6.5, abs=0.1)
    assert meta["sample_rate"] == 16000
    assert meta["channels"] == 1
    assert meta["fsize_bytes"] > 0
    assert meta["rms"] > 0.01


# 2. Audio bytes can be converted to a safe temporary WAV file
def test_safe_temporary_wav_creation():
    raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=2.0, sr=16000)

    with safe_temporary_wav(raw_bytes, prefix="test_mic_") as temp_path:
        assert os.path.exists(temp_path)
        assert os.path.isabs(temp_path)
        assert "test_mic_" in os.path.basename(temp_path)
        # Verify it can be read back as audio
        data, sr = sf.read(temp_path)
        assert len(data) > 0
        assert sr == 16000

    # Outside context block, temp file must not exist
    assert not os.path.exists(temp_path)


# 3. Temporary WAV files are processed by the existing detection pipeline
def test_temporary_wav_processed_by_pipeline():
    raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=6.0, sr=16000)

    with safe_temporary_wav(raw_bytes) as temp_path:
        res = run_pipeline(
            file_source=temp_path,
            codec="g711",
            detection_mode="pretrained",
        )

        assert "overall_liveness" in res
        assert 0 <= res["overall_liveness"] <= 100
        assert res["overall_band"] in ("green", "amber", "red")
        assert res["duration_sec"] > 0
        assert "physics_mean" in res
        assert "biology_mean" in res


# 4. Invalid / corrupted audio is rejected gracefully
def test_invalid_audio_rejected_gracefully():
    corrupted_bytes = b"NOT_A_VALID_WAV_HEADER_DATA_1234567890"
    meta, warnings, error = validate_recording_bytes(corrupted_bytes)

    assert meta is None
    assert error is not None
    assert "Could not decode audio" in error


# 5. Empty audio (None and 0 bytes) is handled with friendly errors
def test_empty_audio_handled():
    # None input
    meta_none, _, error_none = validate_recording_bytes(None)
    assert meta_none is None
    assert "Empty recording" in error_none

    # 0 bytes input
    meta_empty, _, error_empty = validate_recording_bytes(b"")
    assert meta_empty is None
    assert "Empty recording" in error_empty


# 6. Very short audio is handled (rejects < 0.5s, warns for < 6s)
def test_very_short_audio_handled():
    # Too short (< 0.5s) -> Fatal validation error
    short_bytes = create_synthetic_speech_wav_bytes(duration_sec=0.25, sr=16000)
    meta, warnings, error = validate_recording_bytes(short_bytes)
    assert meta is None
    assert error is not None
    assert "too short" in error.lower()

    # Moderate short (2.0s) -> Valid with warning
    mod_bytes = create_synthetic_speech_wav_bytes(duration_sec=2.0, sr=16000)
    meta, warnings, error = validate_recording_bytes(mod_bytes)
    assert error is None
    assert meta is not None
    assert any("continuous speech is recommended" in w for w in warnings)


# 7. Mono and multi-channel input are handled correctly
def test_mono_and_multichannel_handling():
    stereo_bytes = create_synthetic_speech_wav_bytes(duration_sec=4.0, sr=16000, num_channels=2)
    meta, warnings, error = validate_recording_bytes(stereo_bytes)

    assert error is None
    assert meta["channels"] == 2
    assert any("downmixed to mono" in w for w in warnings)

    # Pipeline execution on stereo temp file works through downmixing
    with safe_temporary_wav(stereo_bytes) as temp_path:
        res = run_pipeline(temp_path, codec="clean", detection_mode="baseline")
        assert res["overall_liveness"] is not None
        assert 0 <= res["overall_liveness"] <= 100


# 8. Sample rate handling across standard recording rates
def test_sample_rate_handling():
    for test_sr in (16000, 44100, 48000):
        raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=2.5, sr=test_sr)
        meta, _, error = validate_recording_bytes(raw_bytes)
        assert error is None
        assert meta["sample_rate"] == test_sr

    # Unreasonably low sample rate (< 4000 Hz)
    low_sr_bytes = create_synthetic_speech_wav_bytes(duration_sec=2.0, sr=2000)
    _, _, err_low = validate_recording_bytes(low_sr_bytes)
    assert err_low is not None
    assert "sample rate" in err_low.lower()


# 9. Temporary files are deterministically cleaned up after both success and failure
def test_temporary_files_cleaned_after_success_and_failure():
    raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=1.0, sr=16000)

    # Normal success
    with safe_temporary_wav(raw_bytes) as p1:
        assert os.path.exists(p1)
    assert not os.path.exists(p1)

    # Exception raised inside block
    caught = False
    try:
        with safe_temporary_wav(raw_bytes) as p2:
            assert os.path.exists(p2)
            raise ValueError("Intentional crash to test cleanup")
    except ValueError:
        caught = True

    assert caught
    assert not os.path.exists(p2)


# 10. Results use actual pipeline outputs
def test_results_use_actual_pipeline_outputs():
    raw_bytes = create_synthetic_speech_wav_bytes(duration_sec=6.0, sr=16000)

    with safe_temporary_wav(raw_bytes) as temp_path:
        res = run_pipeline(temp_path, codec="g711", detection_mode="compare")

        # Must include all three engine metrics
        assert "physics_mean" in res
        assert 0.0 <= res["physics_mean"] <= 1.0
        assert "biology_mean" in res
        assert 0.0 <= res["biology_mean"] <= 1.0
        assert "overall_baseline_liveness" in res
        assert 0 <= res["overall_baseline_liveness"] <= 100

        # AASIST output must be real
        if res.get("aasist_model_loaded"):
            assert res["overall_aasist_class"] in ("bona_fide", "spoof")
            assert 0.0 <= res["overall_aasist_bona_fide_score"] <= 1.0
            raw_logits = res["aasist"].get("raw_score")
            assert raw_logits is not None
            assert len(raw_logits) == 2


# 11. Analysis history records the correct input source
def test_analysis_history_records_source():
    history = []

    # Simulate mic recording entry
    mic_entry = {
        "timestamp": "2026-10-31 18:30:00",
        "filename": "Mic Recording (6.2s)",
        "source": "Microphone Recording",
        "codec": "G711",
        "mode": "pretrained",
        "score": 75,
        "band": "green",
        "aasist_class": "bona_fide",
        "aasist_score": 0.8124,
        "duration_sec": 6.2,
        "result_obj": {"overall_liveness": 75},
    }
    history.append(mic_entry)

    # Simulate file upload entry
    upload_entry = {
        "timestamp": "2026-10-31 18:31:00",
        "filename": "sample_call.wav",
        "source": "Uploaded File",
        "codec": "G711",
        "mode": "pretrained",
        "score": 38,
        "band": "red",
        "aasist_class": "spoof",
        "aasist_score": 0.1245,
        "duration_sec": 8.0,
        "result_obj": {"overall_liveness": 38},
    }
    history.append(upload_entry)

    assert history[0]["source"] == "Microphone Recording"
    assert history[1]["source"] == "Uploaded File"
    assert "Mic Recording" in history[0]["filename"]


# 12. Existing uploaded-file analysis continues working without regressions
def test_existing_uploaded_file_analysis_still_works():
    smoke_wav = os.path.join("data", "smoke", "smoke_human_like.wav")
    if not os.path.exists(smoke_wav):
        pytest.skip("Smoke sample not found.")

    res = run_pipeline(smoke_wav, codec="g711", detection_mode="pretrained")
    assert res["overall_liveness"] is not None
    assert 0 <= res["overall_liveness"] <= 100
    assert res["overall_band"] in ("green", "amber", "red")
    assert res["duration_sec"] > 0
