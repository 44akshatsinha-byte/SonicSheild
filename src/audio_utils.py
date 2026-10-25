"""Audio loading, validation, conversion, and preprocessing utilities for SonicShield."""
from __future__ import annotations

import io
import math
import numpy as np
import soundfile as sf
import scipy.signal

TARGET_SR = 8000
SILENCE_RMS_THRESHOLD = 1e-4

class AudioValidationError(Exception):
    """Raised when an uploaded audio file is invalid, empty, or unreadable."""
    pass

class AudioValidationWarning(UserWarning):
    """Warning for non-fatal audio conditions such as excessive silence or clipping."""
    pass

def load_audio(
    file_source: str | io.BytesIO | bytes,
    target_sr: int = TARGET_SR,
    max_duration_seconds: float = 600.0,
) -> tuple[np.ndarray, int, list[str]]:
    """Loads a WAV file from path, bytes, or file-like object.
    
    Returns:
        audio: 1D numpy array of float32 normalized to [-1.0, 1.0] at target_sr (default 8000 Hz).
        sr: target sample rate (8000).
        warnings: list of human-readable warnings (e.g. clipping, digital silence, short length).
    
    Raises:
        AudioValidationError: If the file is unreadable, empty, corrupted, or unsupported.
    """
    warnings: list[str] = []

    # Handle raw bytes
    if isinstance(file_source, bytes):
        if len(file_source) == 0:
            raise AudioValidationError("Audio file is completely empty (0 bytes).")
        file_source = io.BytesIO(file_source)

    try:
        data, orig_sr = sf.read(file_source, dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioValidationError(f"Could not read audio file: {e}") from e

    if data.size == 0 or len(data) == 0:
        raise AudioValidationError("Audio contains no sample data.")

    # Convert stereo / multi-channel to mono
    if data.shape[1] > 1:
        mono = np.mean(data, axis=1)
        warnings.append(f"Multi-channel audio ({data.shape[1]} channels) downmixed to mono.")
    else:
        mono = data[:, 0]

    # Check for NaN or Inf
    if not np.all(np.isfinite(mono)):
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
        warnings.append("Non-finite values (NaN/Inf) were detected and sanitized in the input audio.")

    # Duration check
    raw_duration = len(mono) / orig_sr
    if raw_duration > max_duration_seconds:
        raise AudioValidationError(
            f"Audio duration ({raw_duration:.1f}s) exceeds maximum allowed duration ({max_duration_seconds}s)."
        )

    # Resample to target_sr if necessary
    if orig_sr != target_sr:
        num_target_samples = int(math.ceil(len(mono) * target_sr / orig_sr))
        if num_target_samples == 0:
            raise AudioValidationError("Resampling resulted in 0 samples.")
        resampled = scipy.signal.resample(mono, num_target_samples)
        audio = resampled.astype(np.float32)
    else:
        audio = mono.astype(np.float32)

    # Check clipping
    max_amp = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_amp >= 0.999:
        warnings.append("Severe audio clipping detected (amplitude >= 0.999). Acoustic features may be distorted.")

    # Check RMS energy / silence
    rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
    if rms < SILENCE_RMS_THRESHOLD:
        warnings.append("Audio is virtually silent (RMS < 1e-4). Liveness assessment may be unreliable.")

    final_duration = len(audio) / target_sr
    if final_duration < 1.0:
        warnings.append(f"Recording is very short ({final_duration:.2f}s). At least 6 seconds is recommended for rolling window analysis.")

    return audio, target_sr, warnings

def save_wav(
    file_dest: str | io.BytesIO,
    audio: np.ndarray,
    sr: int = TARGET_SR,
    format: str = "WAV",
    subtype: str = "PCM_16",
) -> None:
    """Saves float32 audio array to WAV format."""
    # Ensure audio is clipped to [-1.0, 1.0]
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(file_dest, clipped, sr, format=format, subtype=subtype)
