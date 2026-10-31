"""Audio loading, validation, conversion, and preprocessing utilities for SonicShield."""
from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
import uuid
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


def detect_unsupported_audio_format(header_bytes: bytes) -> str | None:
    """Detects unsupported audio containers by magic bytes to return clear, actionable guidance."""
    if len(header_bytes) >= 4 and header_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "WebM / Matroska (Opus) container format detected. Please export or record in standard WAV, MP3, FLAC, or OGG format."
    if b"ftyp" in header_bytes[:32]:
        return "M4A / AAC container format detected. Please export or record in standard WAV, MP3, FLAC, or OGG format."
    return None


def analyze_clipping_telemetry(wav: np.ndarray, threshold: float = 0.999) -> dict:
    """Performs rigorous 10-point waveform amplitude and digital rail clipping analysis.
    
    Distinguishes:
    1. Normal Dynamic Range: Signal peaks comfortably below digital full scale.
    2. Legitimate Near-Full-Scale Transient Peak: Isolated samples touching rail,
       with zero flat-topped consecutive saturation and negligible percentage (< 0.01%).
    3. Severe Audio Clipping: True flat-topped waveform saturation with >= 4 consecutive
       samples at the rail and/or significant percentage of saturated samples (>= 0.02%).
       
    Returns:
        dict: Detailed 10-point telemetry metrics and actionable diagnosis.
    """
    if len(wav) == 0:
        return {
            "max_abs": 0.0,
            "min_val": 0.0,
            "max_val": 0.0,
            "rms": 0.0,
            "crest_factor": 0.0,
            "pct_99": 0.0,
            "pct_999": 0.0,
            "max_consecutive_rail": 0,
            "plateau_runs": 0,
            "is_clipped": False,
            "is_transient_peak": False,
            "warning": None,
        }

    mono = wav if wav.ndim == 1 else np.mean(wav, axis=1)
    abs_wav = np.abs(mono)
    max_abs = float(np.max(abs_wav))
    min_val = float(np.min(mono))
    max_val = float(np.max(mono))
    rms = float(np.sqrt(np.mean(mono**2)))
    crest_factor = max_abs / rms if rms > 1e-9 else 0.0

    count_99 = int(np.sum(abs_wav >= 0.99))
    count_999 = int(np.sum(abs_wav >= threshold))
    pct_99 = (count_99 / len(mono)) * 100.0
    pct_999 = (count_999 / len(mono)) * 100.0

    # Consecutive rail samples
    is_at_rail = (abs_wav >= threshold).astype(np.int32)
    max_consecutive = 0
    curr_consecutive = 0
    plateau_runs = 0
    for v in is_at_rail:
        if v == 1:
            curr_consecutive += 1
            if curr_consecutive > max_consecutive:
                max_consecutive = curr_consecutive
        else:
            if curr_consecutive >= 3:
                plateau_runs += 1
            curr_consecutive = 0
    if curr_consecutive >= 3:
        plateau_runs += 1

    # True clipping requires flat-topped peak runs or non-negligible saturation
    is_clipped = (max_consecutive >= 4) or (pct_999 >= 0.02 and max_consecutive >= 2)
    is_transient_peak = (max_abs >= threshold) and not is_clipped

    warning = None
    if is_clipped:
        warning = (
            f"True audio clipping detected ({pct_999:.2f}% saturated samples, {max_consecutive} consecutive "
            f"samples at digital rail). Flat-topped clipping introduces odd-harmonic distortion that causes neural "
            f"anti-spoofing models to misclassify voice as synthetic. Please reduce microphone input gain in system settings."
        )
    elif is_transient_peak:
        warning = (
            f"Near full-scale transient peak detected (max amplitude = {max_abs:.3f}), but waveform is not clipped "
            f"(no flat-topped saturation; dynamic range preserved)."
        )

    return {
        "max_abs": round(max_abs, 4),
        "min_val": round(min_val, 4),
        "max_val": round(max_val, 4),
        "rms": round(rms, 6),
        "crest_factor": round(crest_factor, 2),
        "pct_99": round(pct_99, 3),
        "pct_999": round(pct_999, 4),
        "max_consecutive_rail": max_consecutive,
        "plateau_runs": plateau_runs,
        "is_clipped": is_clipped,
        "is_transient_peak": is_transient_peak,
        "warning": warning,
    }

def load_audio_dual(
    file_source: str | io.BytesIO | bytes,
    max_duration_seconds: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Loads audio from source and produces both 8 kHz and 16 kHz mono float32 waveforms.
    
    Branch A: 8 kHz waveform for Physics and Biology diagnostic engines (telephonic domain).
    Branch B: 16 kHz waveform for Pretrained AASIST neural engine (wideband domain).
    
    Preserves wideband frequencies for AASIST while supplying standard 8 kHz audio
    for acoustic room physics and vocal perturbation analysis.
    
    Returns:
        audio_8k: 1D float32 array at 8000 Hz.
        audio_16k: 1D float32 array at 16000 Hz.
        orig_sr: Original sample rate from audio metadata.
        warnings: List of diagnostic warnings (clipping, multi-channel downmixing, silence).
    
    Raises:
        AudioValidationError: If the file is unreadable, empty, corrupted, or too long.
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
        if isinstance(file_source, io.BytesIO):
            file_source.seek(0)
            header_sample = file_source.read(32)
            detected = detect_unsupported_audio_format(header_sample)
            if detected:
                raise AudioValidationError(detected) from e
        elif isinstance(file_source, str) and os.path.exists(file_source):
            try:
                with open(file_source, "rb") as hf:
                    header_sample = hf.read(32)
                detected = detect_unsupported_audio_format(header_sample)
                if detected:
                    raise AudioValidationError(detected) from e
            except OSError:
                pass
        raise AudioValidationError(f"Could not read audio file: {e}. Supported formats include WAV, MP3, FLAC, and OGG.") from e

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

    # Branch A: 8,000 Hz (Telephony)
    if orig_sr == TARGET_SR:
        audio_8k = mono.astype(np.float32)
    else:
        num_8k = int(math.ceil(len(mono) * TARGET_SR / orig_sr))
        if num_8k == 0:
            raise AudioValidationError("Resampling to 8 kHz resulted in 0 samples.")
        audio_8k = scipy.signal.resample(mono, num_8k).astype(np.float32)

    # Branch B: 16,000 Hz (Wideband AASIST)
    if orig_sr == 16000:
        audio_16k = mono.astype(np.float32)
    elif orig_sr == TARGET_SR:
        # Source was natively 8 kHz: upsample using polyphase filter
        audio_16k = scipy.signal.resample_poly(mono, 2, 1).astype(np.float32)
    else:
        num_16k = int(math.ceil(len(mono) * 16000 / orig_sr))
        if num_16k == 0:
            raise AudioValidationError("Resampling to 16 kHz resulted in 0 samples.")
        audio_16k = scipy.signal.resample(mono, num_16k).astype(np.float32)

    # Check clipping on both representations
    clip_16k = analyze_clipping_telemetry(audio_16k)
    clip_8k = analyze_clipping_telemetry(audio_8k)
    if clip_16k["warning"]:
        warnings.append(clip_16k["warning"])
    elif clip_8k["warning"] and clip_8k["is_clipped"]:
        warnings.append(clip_8k["warning"])

    # Check RMS energy / silence
    rms = float(np.sqrt(np.mean(audio_8k**2))) if len(audio_8k) > 0 else 0.0
    if rms < SILENCE_RMS_THRESHOLD:
        warnings.append("Audio is virtually silent (RMS < 1e-4). Liveness assessment may be unreliable.")

    final_duration = len(audio_8k) / TARGET_SR
    if final_duration < 1.0:
        warnings.append(f"Recording is very short ({final_duration:.2f}s). At least 6 seconds is recommended for rolling window analysis.")

    return audio_8k, audio_16k, orig_sr, warnings


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
        if isinstance(file_source, io.BytesIO):
            file_source.seek(0)
            header_sample = file_source.read(32)
            detected = detect_unsupported_audio_format(header_sample)
            if detected:
                raise AudioValidationError(detected) from e
        elif isinstance(file_source, str) and os.path.exists(file_source):
            try:
                with open(file_source, "rb") as hf:
                    header_sample = hf.read(32)
                detected = detect_unsupported_audio_format(header_sample)
                if detected:
                    raise AudioValidationError(detected) from e
            except OSError:
                pass
        raise AudioValidationError(f"Could not read audio file: {e}. Supported formats include WAV, MP3, FLAC, and OGG.") from e

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
    clip_info = analyze_clipping_telemetry(audio)
    if clip_info["warning"]:
        warnings.append(clip_info["warning"])

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


def validate_recording_bytes(
    raw_bytes: bytes | None,
    min_duration: float = 0.5,
    recommended_duration: float = 6.0,
    max_duration: float = 600.0,
) -> tuple[dict | None, list[str], str | None]:
    """Validates live microphone audio recording bytes before running inference.
    
    Performs comprehensive safety checks:
    - Empty or None byte payloads
    - File format and decoding integrity via soundfile
    - Duration bounds (minimum threshold, optimal rolling window threshold, maximum cap)
    - Channel count and mono downmix detection
    - Sample rate integrity
    - Energy levels (silence, near-zero RMS, severe clipping)
    
    Args:
        raw_bytes: Audio container bytes from microphone input.
        min_duration: Minimum duration (seconds) required for acoustic analysis.
        recommended_duration: Optimal duration (seconds) for rolling window analysis.
        max_duration: Maximum permissible duration (seconds).
        
    Returns:
        metadata: Dictionary with signal diagnostics, or None if validation fails.
        warnings: List of non-fatal user-facing diagnostic observations.
        error: Fatal error description if audio is invalid, or None if valid.
    """
    warnings: list[str] = []

    if raw_bytes is None or len(raw_bytes) == 0:
        return None, [], "Empty recording. No audio data was captured from the microphone."

    try:
        data, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    except Exception as e:
        detected = detect_unsupported_audio_format(raw_bytes[:32])
        if detected:
            return None, [], f"Could not decode audio recording: {detected}"
        return None, [], f"Could not decode audio recording: {e}. Supported formats include WAV, MP3, FLAC, and OGG."

    if data.size == 0 or len(data) == 0:
        return None, [], "Audio recording contains no sample frames."

    num_channels = data.shape[1]
    raw_duration = len(data) / float(orig_sr)

    # Duration validation
    if raw_duration < min_duration:
        return None, [], (
            f"Recording is too short ({raw_duration:.2f}s). "
            f"Please speak for at least {recommended_duration:.0f}–10 seconds for reliable acoustic and neural analysis."
        )

    if raw_duration > max_duration:
        return None, [], f"Recording duration ({raw_duration:.1f}s) exceeds maximum allowed duration ({max_duration:.0f}s)."

    # Sample rate validation
    if orig_sr < 4000:
        return None, [], f"Microphone sample rate ({orig_sr} Hz) is too low for telephonic speech analysis."

    # Channel downmixing warning
    if num_channels > 1:
        mono = np.mean(data, axis=1)
        warnings.append(f"Multi-channel recording ({num_channels} channels) detected; downmixed to mono.")
    else:
        mono = data[:, 0]

    # Sanitize non-finite values
    if not np.all(np.isfinite(mono)):
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
        warnings.append("Non-finite values (NaN/Inf) were detected and sanitized in microphone input.")

    # Recommended duration advisory
    if raw_duration < recommended_duration:
        warnings.append(
            f"Recording is {raw_duration:.1f}s. At least {recommended_duration:.0f}–10s of continuous speech is "
            f"recommended for full 6-second rolling-window analysis. Shorter clips will evaluate on a partial window."
        )

    # Amplitude and RMS energy validation
    max_amp = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0

    if rms < SILENCE_RMS_THRESHOLD:
        warnings.append(
            "Extremely low audio signal / silence detected (RMS < 1e-4). "
            "Please check microphone input volume or keep your microphone closer."
        )
    elif rms < 1e-3:
        warnings.append("Low microphone volume detected. Speaking closer to your microphone will improve detection confidence.")

    clip_stats = analyze_clipping_telemetry(mono)
    if clip_stats["warning"]:
        warnings.append(clip_stats["warning"])

    metadata = {
        "duration_sec": round(raw_duration, 2),
        "sample_rate": orig_sr,
        "channels": num_channels,
        "fsize_bytes": len(raw_bytes),
        "rms": round(rms, 6),
        "max_amp": round(max_amp, 4),
        "num_samples": len(data),
        "clipping": clip_stats,
    }

    return metadata, warnings, None


@contextlib.contextmanager
def safe_temporary_wav(audio_bytes: bytes, prefix: str = "sonicshield_mic_"):
    """Safely writes audio bytes to an isolated temporary WAV file and guarantees deletion on exit.
    
    Guarantees:
    - Unique filename using UUID4 (never uses user-supplied names as filesystem paths).
    - Isolated location in system temp directory outside project repository.
    - Deterministic cleanup in finally block even if analysis or pipeline throws an error.
    - Zero permanent storage of microphone recordings by default.
    
    Args:
        audio_bytes: Raw audio recording bytes.
        prefix: Filename prefix for temp file identification.
        
    Yields:
        temp_path: Absolute filesystem path to the temporary WAV file.
    """
    temp_dir = tempfile.gettempdir()
    unique_filename = f"{prefix}{uuid.uuid4().hex[:12]}.wav"
    temp_path = os.path.join(temp_dir, unique_filename)
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
