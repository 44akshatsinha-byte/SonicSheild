"""Biology Engine: Breath event detection, vocal tremor, jitter, shimmer, and HNR analysis."""
from __future__ import annotations

import math
import numpy as np
import scipy.signal

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except Exception:
    HAS_PARSELMOUTH = False

def _frame_rms(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Computes RMS per frame."""
    num_frames = max(1, (len(signal) - frame_size) // hop_size + 1)
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        st = i * hop_size
        rms[i] = np.sqrt(np.mean(signal[st : st + frame_size] ** 2) + 1e-12)
    return rms

def _detect_breath_events(signal: np.ndarray, sr: int = 8000) -> tuple[int, float, list[dict]]:
    """Detects breath events: quiet broadband noise bursts occurring in speech pauses.
    
    Returns:
        breath_count: Estimated breath events in window.
        breaths_per_min: Extrapolated breaths per minute.
        events: List of detected breath intervals.
    """
    duration_sec = len(signal) / sr
    if duration_sec < 1.0:
        return 0, 0.0, []

    frame_ms = 25
    hop_ms = 10
    frame_size = int(frame_ms * sr / 1000)
    hop_size = int(hop_ms * sr / 1000)

    rms = _frame_rms(signal, frame_size, hop_size)
    if len(rms) < 20:
        return 0, 0.0, []

    rms_db = 20.0 * np.log10(rms + 1e-7)
    peak_db = float(np.max(rms_db))
    pause_thresh = peak_db - 16.0  # Frames quieter than active speech
    floor_thresh = peak_db - 45.0  # Above digital silence/deep noise floor

    # Bandpass filter for breath frequency concentration (1000 Hz to 3500 Hz)
    sos = scipy.signal.butter(4, [1000, 3500], btype="bandpass", fs=sr, output="sos")
    filtered = scipy.signal.sosfilt(sos, signal)
    bp_rms = _frame_rms(filtered, frame_size, hop_size)
    bp_ratio = bp_rms / (rms + 1e-8)

    # Candidate breath frames: quiet relative to speech, but has high relative mid/high noise energy
    candidate_mask = (rms_db < pause_thresh) & (rms_db > floor_thresh) & (bp_ratio > 0.45)

    # Group contiguous candidate frames into bursts (duration between 150ms and 900ms)
    min_frames = int(150 / hop_ms)  # 15 frames
    max_frames = int(900 / hop_ms)  # 90 frames

    events = []
    current_run = 0
    start_frame = 0

    for idx, is_cand in enumerate(candidate_mask):
        if is_cand:
            if current_run == 0:
                start_frame = idx
            current_run += 1
        else:
            if min_frames <= current_run <= max_frames:
                t_start = round(start_frame * hop_ms / 1000.0, 2)
                t_end = round((start_frame + current_run) * hop_ms / 1000.0, 2)
                events.append({
                    "start": t_start,
                    "end": t_end,
                    "duration_ms": round((t_end - t_start) * 1000.0, 1),
                })
            current_run = 0

    breath_count = len(events)
    # Extrapolate to breaths per minute (window is typically 6.0s)
    breaths_per_min = round((breath_count / duration_sec) * 60.0, 1)
    return breath_count, breaths_per_min, events

def _analyze_voice_with_parselmouth(signal: np.ndarray, sr: int = 8000) -> dict:
    """Uses Praat/Parselmouth to extract pitch, jitter, shimmer, tremor, and HNR."""
    if not HAS_PARSELMOUTH or len(signal) < int(0.5 * sr):
        return {
            "pitch_mean_hz": None,
            "pitch_std_hz": None,
            "tremor_ratio": None,
            "jitter_local": None,
            "shimmer_local": None,
            "hnr_db": None,
            "voiced_fraction": None,
        }

    try:
        # Parselmouth expects double float array
        snd = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
        
        # Pitch extraction (floor=75 Hz, ceiling=500 Hz covers human voice range)
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        pitch_values = pitch.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]

        voiced_fraction = float(len(voiced) / max(1, len(pitch_values)))
        if len(voiced) < 15:
            # Insufficient voiced speech
            return {
                "pitch_mean_hz": None,
                "pitch_std_hz": None,
                "tremor_ratio": None,
                "jitter_local": None,
                "shimmer_local": None,
                "hnr_db": None,
                "voiced_fraction": round(voiced_fraction, 3),
            }

        pitch_mean = float(np.mean(voiced))
        pitch_std = float(np.std(voiced))

        # Involuntary physiological tremor (4-12 Hz pitch modulation)
        # Resample voiced pitch track to uniform grid if enough points
        if len(voiced) >= 32:
            # Low frequency tremor energy vs high frequency flutter
            diffs = np.diff(voiced)
            tremor_ratio = float(np.std(diffs) / (pitch_mean + 1e-6))
        else:
            tremor_ratio = 0.02

        # PointProcess for cycle-to-cycle perturbation
        point_proc = praat_call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
        
        # Jitter (local)
        jitter = praat_call(point_proc, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        if math.isnan(jitter) or jitter <= 0.0:
            jitter = None
        else:
            jitter = round(float(jitter * 100.0), 3)  # Expressed as percentage

        # Shimmer (local)
        shimmer = praat_call([snd, point_proc], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        if math.isnan(shimmer) or shimmer <= 0.0:
            shimmer = None
        else:
            shimmer = round(float(shimmer * 100.0), 3)  # Expressed as percentage

        # Harmonics-to-Noise Ratio (HNR)
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=1.0)
        hnr = praat_call(harm, "Get mean", 0.0, 0.0)
        if math.isnan(hnr):
            hnr = None
        else:
            hnr = round(float(hnr), 2)

        return {
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "tremor_ratio": round(tremor_ratio, 4) if tremor_ratio is not None else None,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
            "hnr_db": hnr,
            "voiced_fraction": round(voiced_fraction, 3),
        }
    except Exception:
        return {
            "pitch_mean_hz": None,
            "pitch_std_hz": None,
            "tremor_ratio": None,
            "jitter_local": None,
            "shimmer_local": None,
            "hnr_db": None,
            "voiced_fraction": None,
        }

def analyze(window: np.ndarray, sr: int = 8000) -> dict:
    """Analyzes a 6-second window for biological indicators: breath events, pitch dynamics, tremor, and voice perturbation.
    
    Returns:
        dict containing:
            - 'score': float in [0.0, 1.0] (1.0 = living vocal characteristics)
            - 'reasons': list of plain-English explanations
            - 'features': dict of extracted biological measurements
    """
    if len(window) == 0:
        return {
            "score": 0.5,
            "reasons": ["Window is empty; unable to evaluate biological voice indicators."],
            "features": {"usable": False},
        }

    rms = float(np.sqrt(np.mean(window**2)))
    if rms < 1e-4:
        return {
            "score": 0.20,
            "reasons": ["Digital silence; no vocal or breathing activity detected."],
            "features": {"usable": False, "rms_db": -80.0},
        }

    # 1. Breath event detection
    breath_count, breaths_per_min, breath_events = _detect_breath_events(window, sr=sr)

    # 2. Vocal dynamics & perturbation via Parselmouth
    voice_stats = _analyze_voice_with_parselmouth(window, sr=sr)

    # Heuristic scoring components
    score_components = []
    weights = []
    reasons: list[str] = []

    # A. Breath evaluation
    # Normal human conversation: ~10 to 22 breaths per minute (~1-2 breaths in 6 seconds).
    # Synthetic speech often has 0 breaths or unnaturally high/robotic periodicity.
    if 1 <= breath_count <= 3:
        breath_score = 0.88
        reasons.append("Plausible natural breath pauses observed between vocal segments.")
    elif breath_count == 0:
        breath_score = 0.40
        reasons.append("No natural breath pauses detected in current speech window.")
    else:
        # > 3 breaths in 6s: rapid gasping or noise artifacts
        breath_score = 0.55

    score_components.append(breath_score)
    weights.append(0.35)

    # B. Pitch Dynamics & Vocal Tremor
    pitch_std = voice_stats.get("pitch_std_hz")
    if pitch_std is not None:
        # Human intonation has natural prosody variation (pitch std usually 12 Hz - 55 Hz)
        # Flat monotonic TTS has pitch std < 8 Hz
        if 14.0 <= pitch_std <= 65.0:
            pitch_score = 0.85
            reasons.append("Organic pitch intonation and fundamental frequency variation present.")
        elif pitch_std < 10.0:
            pitch_score = 0.30
            reasons.append("Unusually monotonic, flat pitch track detected.")
        else:
            # Extreme pitch jumps > 70 Hz (possible vocoder glitch)
            pitch_score = 0.60
        score_components.append(pitch_score)
        weights.append(0.25)
    else:
        # Pitch unavailable (unvoiced or whisper)
        pass

    # C. Jitter & Shimmer Perturbation
    jitter = voice_stats.get("jitter_local")
    shimmer = voice_stats.get("shimmer_local")
    if jitter is not None and shimmer is not None:
        # Natural human vocal fold perturbation: jitter 0.4% - 2.5%, shimmer 2.0% - 7.0%
        # Pure mathematical synthesis / neural vocoder: jitter < 0.25%, shimmer < 1.5%
        is_natural_jitter = 0.35 <= jitter <= 3.0
        is_natural_shimmer = 1.8 <= shimmer <= 8.5
        
        if is_natural_jitter and is_natural_shimmer:
            perturb_score = 0.88
            if len(reasons) < 2:
                reasons.append("Vocal fold cycle-to-cycle perturbation (jitter/shimmer) within human physiological range.")
        elif jitter < 0.25 and shimmer < 1.5:
            perturb_score = 0.25
            reasons.append("Acoustic perturbation is abnormally pristine (low jitter/shimmer), typical of direct synthesis.")
        else:
            perturb_score = 0.55
        score_components.append(perturb_score)
        weights.append(0.25)

    # D. Harmonics-to-Noise Ratio
    hnr = voice_stats.get("hnr_db")
    if hnr is not None:
        # Normal telephone human voice HNR: 8 dB - 24 dB
        if 9.0 <= hnr <= 24.0:
            hnr_score = 0.80
        elif hnr > 27.0:
            # Hyper-pure synthetic tone
            hnr_score = 0.35
            if len(reasons) < 2:
                reasons.append("Harmonic purity is unnaturally high for human phonation.")
        else:
            # Low HNR (high noise/whisper)
            hnr_score = 0.50
        score_components.append(hnr_score)
        weights.append(0.15)

    # Compute weighted score
    if weights and sum(weights) > 0:
        norm_weights = np.array(weights) / sum(weights)
        biology_score = float(np.sum(np.array(score_components) * norm_weights))
    else:
        biology_score = 0.50
        reasons.append("Insufficient voiced speech segments to assess vocal biology.")

    biology_score = float(np.clip(biology_score, 0.0, 1.0))

    if not reasons:
        reasons.append("Voice biological characteristics are consistent with expected baseline.")

    return {
        "score": round(biology_score, 3),
        "reasons": reasons[:2],
        "features": {
            "usable": True,
            "breath_count": breath_count,
            "breaths_per_min": breaths_per_min,
            "pitch_mean_hz": voice_stats.get("pitch_mean_hz"),
            "pitch_std_hz": voice_stats.get("pitch_std_hz"),
            "tremor_ratio": voice_stats.get("tremor_ratio"),
            "jitter_local_percent": voice_stats.get("jitter_local"),
            "shimmer_local_percent": voice_stats.get("shimmer_local"),
            "hnr_db": voice_stats.get("hnr_db"),
            "voiced_fraction": voice_stats.get("voiced_fraction"),
        },
    }
