"""Rolling window generator for SonicShield acoustic analysis."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

WINDOW_DURATION_SEC: float = 6.0
HOP_DURATION_SEC: float = 1.0

@dataclass
class AudioWindow:
    """Represents a single rolling analysis window."""
    index: int
    t_start: float
    t_end: float
    t: float  # Display timestamp (t_end of window)
    audio: np.ndarray
    is_padded: bool
    warning: str | None = None

def generate_windows(
    audio: np.ndarray,
    sr: int = 8000,
    window_duration: float = WINDOW_DURATION_SEC,
    hop_duration: float = HOP_DURATION_SEC,
) -> tuple[list[AudioWindow], list[str]]:
    """Segments an audio array into rolling windows of 6 seconds, moving every 1 second.
    
    Args:
        audio: 1D numpy array of 8 kHz audio.
        sr: Sample rate in Hz (default 8000).
        window_duration: Duration of each analysis window in seconds (default 6.0).
        hop_duration: Hop step between windows in seconds (default 1.0).
        
    Returns:
        windows: List of AudioWindow objects.
        warnings: List of top-level warnings about windowing (e.g. short audio).
    """
    warnings: list[str] = []
    window_samples = int(round(window_duration * sr))
    hop_samples = int(round(hop_duration * sr))
    total_samples = len(audio)
    duration_sec = total_samples / sr

    if total_samples == 0:
        return [], ["Audio has 0 samples; cannot generate analysis windows."]

    # Case 1: Audio is shorter than a single window
    if total_samples < window_samples:
        pad_needed = window_samples - total_samples
        padded_audio = np.pad(audio, (0, pad_needed), mode="constant", constant_values=0.0)
        warn_msg = (
            f"Recording ({duration_sec:.2f}s) is shorter than the required {window_duration:.1f}s window. "
            f"Zero-padded to {window_duration:.1f}s; results may have reduced confidence."
        )
        warnings.append(warn_msg)
        return [
            AudioWindow(
                index=0,
                t_start=0.0,
                t_end=window_duration,
                t=round(duration_sec, 2),
                audio=padded_audio,
                is_padded=True,
                warning=warn_msg,
            )
        ], warnings

    # Case 2: Audio is >= window_samples
    windows: list[AudioWindow] = []
    idx = 0
    start_sample = 0

    while start_sample + window_samples <= total_samples:
        end_sample = start_sample + window_samples
        chunk = audio[start_sample:end_sample]
        t_start = round(start_sample / sr, 2)
        t_end = round(end_sample / sr, 2)
        windows.append(
            AudioWindow(
                index=idx,
                t_start=t_start,
                t_end=t_end,
                t=t_end,
                audio=chunk,
                is_padded=False,
            )
        )
        idx += 1
        start_sample += hop_samples

    # Check if there are significant remaining samples (> 0.5s) beyond the last window
    last_covered_sample = (start_sample - hop_samples) + window_samples if idx > 0 else 0
    uncovered_samples = total_samples - last_covered_sample

    if uncovered_samples >= int(0.5 * sr):
        chunk = audio[start_sample:]
        pad_needed = window_samples - len(chunk)
        padded_chunk = np.pad(chunk, (0, pad_needed), mode="constant", constant_values=0.0)
        t_start = round(start_sample / sr, 2)
        t_end = round(duration_sec, 2)
        windows.append(
            AudioWindow(
                index=idx,
                t_start=t_start,
                t_end=t_end,
                t=t_end,
                audio=padded_chunk,
                is_padded=True,
                warning=f"Final window ({len(chunk) / sr:.1f}s) padded to {window_duration:.1f}s.",
            )
        )

    return windows, warnings
