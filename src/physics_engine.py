"""Physics Engine: Room reverberation, echo, and ambient room tone analysis."""
from __future__ import annotations

import numpy as np
import scipy.signal

def _frame_energy(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Computes RMS energy per frame."""
    num_frames = max(1, (len(signal) - frame_size) // hop_size + 1)
    energies = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start : start + frame_size]
        energies[i] = np.sqrt(np.mean(frame**2) + 1e-12)
    return energies

def _measure_decay_tail(signal: np.ndarray, sr: int = 8000) -> tuple[float, float]:
    """Estimates energy decay slope following speech offsets (reverberation proxy).
    
    Returns:
        decay_slope: Average slope (dB/ms) of post-speech decay (more negative = faster drop).
        reverb_score: Heuristic score in [0.0, 1.0] where 1.0 indicates natural room decay.
    """
    frame_ms = 20
    hop_ms = 10
    frame_size = int(frame_ms * sr / 1000)
    hop_size = int(hop_ms * sr / 1000)
    
    energies = _frame_energy(signal, frame_size, hop_size)
    if len(energies) < 10:
        return 0.0, 0.5

    energy_db = 20.0 * np.log10(energies + 1e-6)
    peak_db = np.max(energy_db)
    
    # Identify voice activity transitions (drop from speech > peak - 15dB to < peak - 30dB)
    speech_thresh = peak_db - 15.0
    slopes = []

    # Look for offsets: frame i is speech, frame i+1 drops
    for i in range(1, len(energy_db) - 10):
        if energy_db[i - 1] > speech_thresh and energy_db[i] <= speech_thresh:
            # Measure next 200ms (20 hops)
            window_len = min(20, len(energy_db) - i)
            if window_len > 5:
                decay_segment = energy_db[i : i + window_len]
                time_axis = np.arange(window_len) * hop_ms  # in ms
                # Fit linear regression: energy = slope * t + intercept
                if np.std(time_axis) > 1e-5:
                    p = np.polyfit(time_axis, decay_segment, 1)
                    slopes.append(float(p[0]))

    if not slopes:
        return -0.2, 0.5  # Neutral default when no distinct offsets found

    avg_slope = float(np.median(slopes))
    # Natural rooms: slope typically -0.05 to -0.20 dB/ms (smooth decay)
    # Dry synthetic cuts: slope < -0.35 dB/ms (steep falloff)
    # Very noisy/flat: slope near 0.0
    if -0.25 <= avg_slope <= -0.03:
        reverb_score = float(np.interp(avg_slope, [-0.25, -0.10, -0.03], [0.75, 0.95, 0.70]))
    elif avg_slope < -0.25:
        # Sharp drop, dry or gated
        reverb_score = float(np.interp(avg_slope, [-0.60, -0.25], [0.15, 0.75]))
    else:
        # Near zero slope: continuous noise
        reverb_score = 0.50

    return avg_slope, float(np.clip(reverb_score, 0.0, 1.0))

def _analyze_pause_room_tone(signal: np.ndarray, sr: int = 8000) -> tuple[float, float, bool, float]:
    """Analyzes ambient noise floor during pauses.
    
    Returns:
        pause_rms_db: Average RMS level during pauses in dBFS.
        spectral_flatness: Wiener entropy of pause spectrum [0, 1].
        is_digital_silence: True if pause is near-zero digital silence or gated.
        pause_score: Heuristic score in [0.0, 1.0].
    """
    frame_size = int(0.025 * sr)  # 25 ms
    hop_size = int(0.010 * sr)    # 10 ms
    energies = _frame_energy(signal, frame_size, hop_size)
    
    if len(energies) < 5:
        return -80.0, 0.0, True, 0.2

    energy_db = 20.0 * np.log10(energies + 1e-7)
    min_db = float(np.min(energy_db))
    p20_db = float(np.percentile(energy_db, 20))  # 20th percentile as pause level

    # Check for digital silence (zeros or < -65 dBFS)
    if p20_db < -60.0 or min_db < -75.0:
        is_digital_silence = True
        return p20_db, 0.0, True, 0.15

    is_digital_silence = False

    # Extract pause frames to measure spectral flatness
    pause_mask = energy_db <= (p20_db + 3.0)
    pause_samples = []
    num_frames = len(energy_db)
    for i in range(num_frames):
        if pause_mask[i]:
            st = i * hop_size
            pause_samples.extend(signal[st : st + frame_size])

    if len(pause_samples) > 256:
        pause_arr = np.array(pause_samples[:4000], dtype=np.float32)
        fft_mag = np.abs(np.fft.rfft(pause_arr * np.hanning(len(pause_arr)))) + 1e-8
        power = fft_mag ** 2
        geom_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        flatness = float(geom_mean / (arith_mean + 1e-12))
    else:
        flatness = 0.4

    # Natural room tone typically sits between -55 dBFS and -35 dBFS with organic spectral shape
    # Dry synthesis or noise gate is either <-60 dBFS or has synthetic white flatness (>0.85)
    if -55.0 <= p20_db <= -30.0 and 0.10 <= flatness <= 0.75:
        pause_score = 0.85
    elif p20_db > -30.0:
        # Loud noise floor
        pause_score = 0.55
    else:
        pause_score = float(np.interp(p20_db, [-60.0, -50.0], [0.20, 0.70]))

    return p20_db, flatness, is_digital_silence, float(np.clip(pause_score, 0.0, 1.0))

def _detect_cepstral_micro_echo(signal: np.ndarray, sr: int = 8000) -> tuple[float, float]:
    """Detects short-delay micro-reflections (room echo) via cepstrum analysis.
    
    Quefrency range: 2ms to 20ms (~16 to 160 samples at 8 kHz).
    """
    if len(signal) < 512:
        return 0.0, 0.5

    # Take high-energy voiced frames
    frame_size = int(0.040 * sr)  # 40ms frame
    hop_size = int(0.020 * sr)
    energies = _frame_energy(signal, frame_size, hop_size)
    median_e = np.median(energies)
    active_indices = np.where(energies > median_e * 1.5)[0]

    if len(active_indices) == 0:
        return 0.0, 0.5

    peaks = []
    # Test up to 10 active frames
    sample_indices = active_indices[:: max(1, len(active_indices) // 10)]
    for idx in sample_indices:
        st = idx * hop_size
        frame = signal[st : st + frame_size] * np.hamming(frame_size)
        spectrum = np.fft.rfft(frame)
        log_spec = np.log(np.abs(spectrum) + 1e-8)
        cepstrum = np.abs(np.fft.irfft(log_spec))

        # Check quefrency window: 2ms to 20ms
        q_min = int(0.002 * sr)
        q_max = min(len(cepstrum) - 1, int(0.020 * sr))
        if q_max > q_min:
            q_band = cepstrum[q_min:q_max]
            peak_val = np.max(q_band)
            mean_val = np.mean(q_band) + 1e-8
            peaks.append(float(peak_val / mean_val))

    if not peaks:
        return 0.0, 0.5

    median_peak_ratio = float(np.median(peaks))
    # In reflective rooms, peak-to-average ratio shows distinct micro-echo peaks (2.0 - 4.5)
    # In anechoic / direct synthetic voice, spectrum is smooth harmonic decay with minimal wall echo
    echo_score = float(np.interp(median_peak_ratio, [1.2, 2.2, 4.5], [0.35, 0.85, 0.90]))
    return median_peak_ratio, float(np.clip(echo_score, 0.0, 1.0))

def analyze(window: np.ndarray, sr: int = 8000) -> dict:
    """Analyzes a 6-second window for room acoustics, echo, and pause room tone.
    
    Returns:
        dict containing:
            - 'score': float in [0.0, 1.0] (1.0 = organic room acoustics; 0.0 = suspicious/dry)
            - 'reasons': list of plain-English explanations
            - 'features': dict of extracted physical features
    """
    if len(window) == 0:
        return {
            "score": 0.5,
            "reasons": ["Window is empty; unable to evaluate room acoustics."],
            "features": {"usable": False},
        }

    rms = float(np.sqrt(np.mean(window**2)))
    if rms < 1e-4:
        return {
            "score": 0.20,
            "reasons": ["Near-zero audio energy throughout window (digital silence)."],
            "features": {
                "usable": False,
                "rms_db": -80.0,
                "decay_slope": 0.0,
                "pause_rms_db": -80.0,
                "spectral_flatness": 0.0,
                "is_digital_silence": True,
                "cepstral_peak_ratio": 0.0,
            },
        }

    # 1. Decay tail slope
    decay_slope, reverb_score = _measure_decay_tail(window, sr=sr)

    # 2. Room tone in pauses
    pause_rms_db, flatness, is_dig_silence, pause_score = _analyze_pause_room_tone(window, sr=sr)

    # 3. Echo / cepstral micro-reflections
    peak_ratio, echo_score = _detect_cepstral_micro_echo(window, sr=sr)

    # Weighted combination for Physics Score
    if is_dig_silence:
        combined_score = 0.20 * reverb_score + 0.15 * pause_score + 0.25 * echo_score
        combined_score = min(0.35, combined_score)
    else:
        combined_score = 0.40 * reverb_score + 0.35 * pause_score + 0.25 * echo_score

    combined_score = float(np.clip(combined_score, 0.0, 1.0))

    # Formulate plain-English reasons
    reasons: list[str] = []
    if is_dig_silence:
        reasons.append("Digital silence or artificial noise gating detected in pauses.")
    elif pause_score > 0.75:
        reasons.append("Organic ambient room tone observed during speech pauses.")
    elif pause_rms_db < -55.0:
        reasons.append("Extremely low background room tone; environment is abnormally quiet.")

    if reverb_score > 0.80:
        reasons.append("Natural acoustic reverberation tail detected following speech.")
    elif reverb_score < 0.40:
        reasons.append("Sharp, abrupt speech energy cutoff with minimal room decay tail.")

    if echo_score > 0.80 and len(reasons) < 2:
        reasons.append("Micro-echo reflections characteristic of an enclosed physical room detected.")

    if not reasons:
        reasons.append("Acoustic room indicators appear consistent with baseline environment.")

    return {
        "score": round(combined_score, 3),
        "reasons": reasons[:2],
        "features": {
            "usable": True,
            "rms_db": round(float(20.0 * np.log10(rms + 1e-7)), 1),
            "decay_slope_db_ms": round(decay_slope, 4),
            "reverb_score": round(reverb_score, 3),
            "pause_rms_db": round(pause_rms_db, 1),
            "spectral_flatness": round(flatness, 3),
            "is_digital_silence": is_dig_silence,
            "cepstral_peak_ratio": round(peak_ratio, 2),
        },
    }
