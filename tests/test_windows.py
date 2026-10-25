"""Unit tests for windows module."""
import numpy as np
from src.windows import generate_windows, AudioWindow

def test_generate_windows_standard_8_seconds():
    sr = 8000
    # 8 seconds = 64,000 samples
    audio = np.zeros(64000, dtype=np.float32)
    windows, warnings = generate_windows(audio, sr=sr)

    # With 8 seconds, 6s window, 1s hop:
    # Win 0: 0s to 6s
    # Win 1: 1s to 7s
    # Win 2: 2s to 8s
    assert len(windows) == 3
    assert windows[0].t_start == 0.0
    assert windows[0].t_end == 6.0
    assert windows[0].t == 6.0
    assert len(windows[0].audio) == 48000  # 6 * 8000
    assert not windows[0].is_padded

    assert windows[2].t_start == 2.0
    assert windows[2].t_end == 8.0
    assert windows[2].t == 8.0

def test_generate_windows_short_audio_padding():
    sr = 8000
    # 2.5 seconds = 20,000 samples
    audio = np.ones(20000, dtype=np.float32) * 0.5
    windows, warnings = generate_windows(audio, sr=sr)

    assert len(windows) == 1
    assert windows[0].is_padded
    assert len(windows[0].audio) == 48000
    assert np.all(windows[0].audio[:20000] == 0.5)
    assert np.all(windows[0].audio[20000:] == 0.0)
    assert any("shorter" in w for w in warnings)

def test_generate_windows_empty():
    audio = np.array([], dtype=np.float32)
    windows, warnings = generate_windows(audio)
    assert len(windows) == 0
    assert len(warnings) > 0
