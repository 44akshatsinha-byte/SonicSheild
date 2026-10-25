"""Unit tests for audio_utils module."""
import io
import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import load_audio, save_wav, AudioValidationError

def test_load_audio_mono_and_resample():
    # Create 1 second 16000 Hz stereo audio
    sr_orig = 16000
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.25 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([left, right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr_orig, format="WAV", subtype="PCM_16")
    buf.seek(0)

    audio, sr, warnings = load_audio(buf)

    assert sr == 8000
    assert audio.ndim == 1
    assert len(audio) == 8000
    assert audio.dtype == np.float32
    assert any("downmixed to mono" in w for w in warnings)

def test_load_audio_empty_bytes():
    with pytest.raises(AudioValidationError, match="empty"):
        load_audio(b"")

def test_load_audio_corrupted_data():
    with pytest.raises(AudioValidationError, match="Could not read"):
        load_audio(b"RIFF\x00\x00\x00\x00WAVEfmt invalid corrupt data here")

def test_load_audio_silence_warning():
    silence = np.zeros(8000, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, 8000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    audio, sr, warnings = load_audio(buf)
    assert any("silent" in w.lower() for w in warnings)

def test_save_wav_roundtrip():
    audio_orig = np.sin(np.linspace(0, 2 * np.pi, 8000)).astype(np.float32)
    buf = io.BytesIO()
    save_wav(buf, audio_orig, sr=8000)
    buf.seek(0)

    loaded, sr, _ = load_audio(buf)
    assert sr == 8000
    assert np.allclose(audio_orig, loaded, atol=1e-3)
