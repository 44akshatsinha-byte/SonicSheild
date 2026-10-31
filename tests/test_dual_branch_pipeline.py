"""Unit and integration tests for the SonicShield Dual-Branch Audio Processing Pipeline.

Verifies:
1. Branch A (8 kHz): Preserves existing rolling windows, Physics Engine, and Biology Engine.
2. Branch B (16 kHz): Delivers native 16 kHz wideband audio to AASIST in Clean mode,
   and authentic 8 kHz -> 16 kHz upsampled telephony audio in G.711 mode.
3. Genuine speech receives high bona-fide scores in Clean mode.
4. G.711 telephony channel limitation is documented and reported honestly.
5. Upload and live microphone audio yield identical processing behavior.
"""
import io
import os
import numpy as np
import pytest
import soundfile as sf
import scipy.signal

from src.audio_utils import load_audio, load_audio_dual, safe_temporary_wav
from src.pipeline import run_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SMOKE_DIR = os.path.join(DATA_DIR, "smoke")
IN_THE_WILD_DIR = os.path.join(DATA_DIR, "external", "in_the_wild", "release_in_the_wild", "release_in_the_wild")


def test_load_audio_dual_outputs_both_rates():
    """Verifies that load_audio_dual produces both 8 kHz and 16 kHz mono float32 waveforms."""
    # Generate 3 seconds of 48 kHz stereo chirp
    orig_sr = 48000
    t = np.linspace(0, 3.0, orig_sr * 3, endpoint=False)
    sig_left = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig_right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([sig_left, sig_right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, orig_sr, format="WAV")
    buf.seek(0)

    audio_8k, audio_16k, returned_sr, warnings = load_audio_dual(buf)

    assert returned_sr == orig_sr
    assert audio_8k.ndim == 1
    assert audio_16k.ndim == 1
    assert audio_8k.dtype == np.float32
    assert audio_16k.dtype == np.float32

    # Verify lengths match expected sample counts
    assert len(audio_8k) == 8000 * 3
    assert len(audio_16k) == 16000 * 3
    assert any("downmixed to mono" in w for w in warnings)


def test_clean_mode_preserves_16k_for_aasist():
    """Verifies that in Clean mode, Branch B AASIST receives native 16 kHz wideband audio."""
    human_wav = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(human_wav)

    result = run_pipeline(human_wav, codec="clean", detection_mode="pretrained")

    assert result["codec"] == "clean"
    assert "audio_16k" in result
    assert "raw_audio" in result  # Branch A (8 kHz)
    assert len(result["audio_16k"]) == 2 * len(result["raw_audio"])
    assert result["aasist_model_loaded"] is True
    assert result["overall_aasist_bona_fide_score"] is not None
    assert 0.0 <= result["overall_aasist_bona_fide_score"] <= 1.0


def test_genuine_speech_scores_high_on_clean_branch_b():
    """Verifies that authentic human speech receives high bona-fide score on native 16 kHz."""
    # 9.wav is Alec Guinness (Genuine human speech)
    genuine_path = os.path.join(IN_THE_WILD_DIR, "9.wav")
    if not os.path.exists(genuine_path):
        pytest.skip("In-the-wild dataset not found locally, skipping live sample test.")

    result = run_pipeline(genuine_path, codec="clean", detection_mode="pretrained")

    # In Clean mode, native 16 kHz preserves the 4–8 kHz band: score should exceed 0.85
    assert result["overall_aasist_class"] == "bona_fide"
    assert result["overall_aasist_bona_fide_score"] >= 0.85
    assert result["overall_liveness"] >= 85
    assert result["overall_band"] == "green"


def test_g711_telephony_channel_effect():
    """Verifies that G.711 mode performs PSTN telephony compression and notes out-of-domain effect."""
    human_wav = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(human_wav)

    result = run_pipeline(human_wav, codec="g711", detection_mode="pretrained")

    assert result["codec"] == "g711"
    assert result["telephony_out_of_domain"] is True
    # Verify the out-of-domain telephony explanation is documented in warnings
    assert any("telephone" in w.lower() for w in result["warnings"])
    assert any("out-of-domain" in w.lower() or "bandlimited" in w.lower() for w in result["warnings"])


def test_branch_a_physics_and_biology_remain_intact():
    """Verifies Branch A (8 kHz) physics and biology engines function identically on 6s windows."""
    human_wav = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(human_wav)

    result = run_pipeline(human_wav, codec="clean", mode="both", detection_mode="baseline")

    assert result["sample_rate"] == 8000
    assert result["num_windows"] >= 2
    assert "physics_mean" in result
    assert "biology_mean" in result
    assert 0.0 <= result["physics_mean"] <= 1.0
    assert 0.0 <= result["biology_mean"] <= 1.0

    # Check window contents
    first_win = result["windows"][0]
    assert "physics" in first_win
    assert "biology" in first_win
    assert "reasons" in first_win
    assert first_win["t_end"] - first_win["t_start"] == 6.0


def test_microphone_recording_and_file_upload_equivalence():
    """Verifies that microphone bytes and direct file uploads produce identical pipeline results."""
    smoke_wav = os.path.join(SMOKE_DIR, "smoke_human_like.wav")
    assert os.path.exists(smoke_wav)

    with open(smoke_wav, "rb") as f:
        wav_bytes = f.read()

    # Path 1: Direct bytes
    res_bytes = run_pipeline(wav_bytes, codec="clean", detection_mode="pretrained")

    # Path 2: Microphone workflow (temporary WAV file)
    with safe_temporary_wav(wav_bytes, prefix="test_mic_equiv_") as temp_path:
        res_mic = run_pipeline(temp_path, codec="clean", detection_mode="pretrained")

    assert res_bytes["overall_liveness"] == res_mic["overall_liveness"]
    assert res_bytes["overall_aasist_bona_fide_score"] == res_mic["overall_aasist_bona_fide_score"]
    assert res_bytes["overall_aasist_class"] == res_mic["overall_aasist_class"]
    assert res_bytes["num_windows"] == res_mic["num_windows"]
