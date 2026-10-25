"""Unit tests for codec_crusher module."""
import numpy as np
import pytest
from src.codec_crusher import crush, get_supported_codecs, check_amr_support, CodecUnavailableError

def test_get_supported_codecs():
    codecs = get_supported_codecs()
    assert codecs["clean"] is True
    assert codecs["g711"] is True
    assert isinstance(codecs["amr"], bool)

def test_crush_clean_passthrough():
    audio = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    crushed = crush(audio, sr=8000, codec="clean")
    assert np.array_equal(audio, crushed)
    assert crushed is not audio  # Must be a copy

def test_crush_g711_mulaw():
    # Test sine wave through ITU-T G.711 round trip
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    sine = (0.7 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    
    crushed = crush(sine, sr=8000, codec="g711")
    assert crushed.shape == sine.shape
    assert crushed.dtype == np.float32
    assert np.all(np.isfinite(crushed))
    # G.711 introduces 8-bit logarithmic quantization distortion, but keeps high SNR
    diff = np.abs(crushed - sine)
    assert np.max(diff) < 0.05
    assert np.mean(diff) < 0.01

def test_crush_amr_handling():
    audio = np.zeros(8000, dtype=np.float32)
    if not check_amr_support():
        with pytest.raises(CodecUnavailableError, match="AMR-NB codec requires FFmpeg"):
            crush(audio, sr=8000, codec="amr")
    else:
        crushed = crush(audio, sr=8000, codec="amr")
        assert len(crushed) == len(audio)

def test_crush_invalid_codec():
    audio = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown codec"):
        crush(audio, sr=8000, codec="invalid_xyz")

def test_crush_with_line_artifacts():
    audio = np.ones(8000, dtype=np.float32) * 0.5
    crushed = crush(audio, sr=8000, codec="clean", packet_loss_rate=0.2, line_noise_snr=20.0)
    assert len(crushed) == 8000
    assert np.all(np.isfinite(crushed))
    # Packets were dropped, so some samples are 0.0 or noise
    assert not np.all(crushed == 0.5)
