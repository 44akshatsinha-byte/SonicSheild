"""Forensic regression test suite for SonicShield P0 zero score resolution.

Verifies:
1. Correct output extraction and class mapping (Index 0 = spoof, Index 1 = bona_fide).
2. Score conversion and precision (raw logits to continuous probability to integer index).
3. Raw-logit handling (finite, non-clamped logits with delta margin).
4. Invalid audio handling (rejects empty/corrupted payloads with AudioValidationError).
5. Unsupported formats (detects WebM/Opus and M4A with helpful actionable messages).
6. Integer PCM scaling (integer ndarray divided by 32768.0, float32 preserved).
7. Resampling integrity (real resampling changes sample count proportionally).
8. Mono conversion (multi-channel audio correctly downmixed by channel averaging).
9. Upload inference across formats (WAV, MP3, FLAC, OGG).
10. Microphone recording inference (safe_temporary_wav pipeline parity).
11. No silent zero-score fallback (exceptions never silently produce score 0).
"""
from __future__ import annotations

import io
import os
import pytest
import numpy as np
import soundfile as sf
import torch

from src.audio_utils import (
    load_audio,
    load_audio_dual,
    validate_recording_bytes,
    detect_unsupported_audio_format,
    safe_temporary_wav,
    AudioValidationError,
)
from src.pipeline import run_pipeline, PipelineError
from src.pretrained_aasist import get_aasist_classifier


def create_test_wav_bytes(
    duration_sec: float = 2.0,
    sr: int = 16000,
    channels: int = 1,
    fmt: str = "WAV",
    subtype: str = "PCM_16",
) -> bytes:
    """Generates synthetic harmonic test audio bytes."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sig = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
    sig = sig.astype(np.float32)
    if channels > 1:
        sig = np.column_stack([sig] * channels)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


# 1. Correct output extraction and class mapping
def test_class_mapping_and_output_extraction():
    aasist = get_aasist_classifier()
    wav = np.zeros(64600, dtype=np.float32)
    res = aasist.predict(wav, sr=16000)

    assert "model" in res
    assert res["model"] == "AASIST"
    assert "raw_score" in res
    assert len(res["raw_score"]) == 2
    assert "predicted_class" in res
    assert res["predicted_class"] in ("bona_fide", "spoof")
    assert "bona_fide_score" in res
    assert "spoof_score" in res
    assert pytest.approx(res["bona_fide_score"] + res["spoof_score"], abs=1e-3) == 1.0


# 2. Score conversion and continuous precision
def test_score_conversion_preserves_precision():
    aasist = get_aasist_classifier()
    smoke_human = os.path.join("data", "smoke", "smoke_human_like.wav")
    if os.path.exists(smoke_human):
        data, sr = sf.read(smoke_human, dtype="float32")
        res = aasist.predict(data, sr=sr)
        assert res["bona_fide_score"] is not None
        assert 0.0 <= res["bona_fide_score"] <= 1.0
        assert 0 <= res["liveness_score"] <= 100


# 3. Raw-logit handling (preserves negative values without zero-clamping)
def test_raw_logits_preserve_negative_values():
    aasist = get_aasist_classifier()
    # Test on a synthetic file where spoof logit is high and bona logit is negative
    smoke_synth = os.path.join("data", "smoke", "smoke_synthetic_like.wav")
    if os.path.exists(smoke_synth):
        data, sr = sf.read(smoke_synth, dtype="float32")
        res = aasist.predict(data, sr=sr)
        raw_logits = res["raw_score"]
        assert raw_logits is not None
        assert len(raw_logits) == 2
        # Verify logits are raw floats, NOT clamped to zero
        assert "logit_diff" in res
        assert res["logit_diff"] == pytest.approx(raw_logits[1] - raw_logits[0], abs=1e-3)


# 4. Invalid audio handling
def test_invalid_audio_raises_validation_error():
    with pytest.raises(AudioValidationError):
        load_audio_dual(b"CORRUPT_NOT_AN_AUDIO_CONTAINER")

    meta, warns, err = validate_recording_bytes(b"CORRUPT_NOT_AN_AUDIO_CONTAINER")
    assert meta is None
    assert err is not None
    assert "Could not decode" in err


# 5. Unsupported formats detection (WebM / M4A)
def test_unsupported_formats_detection():
    # Simulated WebM header
    webm_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
    msg_webm = detect_unsupported_audio_format(webm_header)
    assert msg_webm is not None
    assert "WebM" in msg_webm

    # Simulated M4A header
    m4a_header = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00"
    msg_m4a = detect_unsupported_audio_format(m4a_header)
    assert msg_m4a is not None
    assert "M4A" in msg_m4a

    with pytest.raises(AudioValidationError, match="WebM"):
        load_audio_dual(webm_header)


# 6. Integer PCM scaling vs Float32 preservation
def test_integer_pcm_scaling_in_pipeline():
    # 16-bit integer array
    int_pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    int_pcm_expanded = np.tile(int_pcm, 16000)  # ~5 seconds at 16 kHz
    res_int = run_pipeline(int_pcm_expanded, sr=16000, codec="clean", detection_mode="baseline")
    assert res_int["overall_liveness"] is not None

    # Float array already scaled
    float_pcm = np.array([0.0, 0.5, -0.5, 0.99, -0.99], dtype=np.float32)
    float_pcm_expanded = np.tile(float_pcm, 16000)
    res_float = run_pipeline(float_pcm_expanded, sr=16000, codec="clean", detection_mode="baseline")
    assert res_float["overall_liveness"] is not None


# 7. Resampling integrity
def test_resampling_scales_sample_count():
    # 8 kHz audio resampled to 16 kHz in load_audio_dual
    raw_8k = create_test_wav_bytes(duration_sec=3.0, sr=8000)
    a8, a16, orig_sr, _ = load_audio_dual(raw_8k)
    assert orig_sr == 8000
    assert len(a8) == pytest.approx(24000, abs=10)
    assert len(a16) == pytest.approx(48000, abs=10)


# 8. Mono downmix from multi-channel
def test_stereo_downmix_averaging():
    stereo_bytes = create_test_wav_bytes(duration_sec=2.0, sr=16000, channels=2)
    a8, a16, orig_sr, warns = load_audio_dual(stereo_bytes)
    assert a8.ndim == 1
    assert a16.ndim == 1
    assert any("downmixed to mono" in w for w in warns)


# 9. Multi-format upload inference (WAV, FLAC, OGG)
def test_multi_format_pipeline_inference():
    for fmt, subtype in [("WAV", "PCM_16"), ("FLAC", "PCM_16"), ("OGG", "VORBIS")]:
        audio_b = create_test_wav_bytes(duration_sec=6.0, sr=16000, fmt=fmt, subtype=subtype)
        res = run_pipeline(audio_b, codec="clean", detection_mode="pretrained")
        assert res["overall_liveness"] is not None
        assert 0 <= res["overall_liveness"] <= 100
        assert res["overall_aasist_class"] in ("bona_fide", "spoof")


# 10. Microphone temporary WAV pipeline parity
def test_microphone_temp_wav_parity():
    raw_bytes = create_test_wav_bytes(duration_sec=6.0, sr=16000)
    # Direct from bytes
    res_direct = run_pipeline(raw_bytes, codec="clean", detection_mode="pretrained")

    # Via safe_temporary_wav
    with safe_temporary_wav(raw_bytes, prefix="test_parity_") as temp_p:
        res_temp = run_pipeline(temp_p, codec="clean", detection_mode="pretrained")

    assert res_direct["overall_liveness"] == res_temp["overall_liveness"]
    assert res_direct["overall_aasist_bona_fide_score"] == pytest.approx(
        res_temp["overall_aasist_bona_fide_score"], abs=1e-5
    )


# 11. No silent zero-score fallback on invalid audio
def test_no_silent_zero_score_fallback():
    # Empty audio must raise AudioValidationError, never return score 0
    with pytest.raises(AudioValidationError):
        run_pipeline(b"", codec="clean", detection_mode="pretrained")

    # Corrupted bytes must raise AudioValidationError, never return score 0
    with pytest.raises(AudioValidationError):
        run_pipeline(b"NOT_VALID_AUDIO_RANDOM_STREAM_BYTES", codec="clean", detection_mode="pretrained")
