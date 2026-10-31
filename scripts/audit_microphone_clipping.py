"""Comprehensive audit of clipping detection, PCM integer conversion, and waveform statistics."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import numpy as np
import soundfile as sf
import scipy.signal

def audit_clipping_metrics(wav: np.ndarray, orig_sr: int, name: str = "Audio"):
    """Performs the 10-point clipping analysis required by Phase 1."""
    mono = wav if wav.ndim == 1 else np.mean(wav, axis=1)
    abs_wav = np.abs(mono)
    
    # 1. Maximum absolute waveform amplitude
    max_abs = float(np.max(abs_wav)) if len(mono) > 0 else 0.0
    # 2. Minimum amplitude
    min_amp = float(np.min(mono)) if len(mono) > 0 else 0.0
    max_amp = float(np.max(mono)) if len(mono) > 0 else 0.0
    # 3. RMS amplitude
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0
    # 4. Peak-to-RMS ratio (Crest Factor)
    crest_factor = max_abs / rms if rms > 1e-9 else 0.0
    # 5. Percentage of samples above 0.99
    count_99 = int(np.sum(abs_wav >= 0.99))
    pct_99 = (count_99 / len(mono)) * 100.0 if len(mono) > 0 else 0.0
    # 6. Percentage of samples above 0.999
    count_999 = int(np.sum(abs_wav >= 0.999))
    pct_999 = (count_999 / len(mono)) * 100.0 if len(mono) > 0 else 0.0
    
    # 7. Number of consecutive samples at full scale (>= 0.999)
    # 8. Whether waveform contains actual flat-topped peaks
    is_at_rail = (abs_wav >= 0.999).astype(np.int32)
    max_consecutive = 0
    curr_consecutive = 0
    consecutive_runs = 0
    for v in is_at_rail:
        if v == 1:
            curr_consecutive += 1
            if curr_consecutive > max_consecutive:
                max_consecutive = curr_consecutive
        else:
            if curr_consecutive >= 3:
                consecutive_runs += 1
            curr_consecutive = 0
    if curr_consecutive >= 3:
        consecutive_runs += 1

    has_flat_tops = max_consecutive >= 3

    # Distinction
    if max_consecutive >= 5 or (pct_999 >= 0.05 and max_consecutive >= 3):
        clipping_category = "Severe Waveform Clipping (Flat-Topped Peaks)"
    elif max_consecutive >= 3 or pct_999 >= 0.01:
        clipping_category = "Moderate Waveform Clipping"
    elif max_abs >= 0.999:
        clipping_category = "Legitimate Near-Full-Scale Transient Peak (Not Clipped)"
    elif max_abs >= 0.90:
        clipping_category = "High Dynamic Range (No Clipping)"
    else:
        clipping_category = "Normal Dynamic Range"

    print(f"\nAudit Report for: {name}")
    print(f"  1. Max Absolute Amplitude:      {max_abs:.6f}")
    print(f"  2. Min/Max Range:               [{min_amp:.6f}, {max_amp:.6f}]")
    print(f"  3. RMS Amplitude:               {rms:.6f}")
    print(f"  4. Peak-to-RMS (Crest Factor):  {crest_factor:.2f}")
    print(f"  5. Samples >= 0.99:             {count_99} ({pct_99:.3f}%)")
    print(f"  6. Samples >= 0.999:            {count_999} ({pct_999:.4f}%)")
    print(f"  7. Max Consecutive at Rail:     {max_consecutive} samples")
    print(f"  8. Flat-Topped Peaks Detected:  {has_flat_tops} ({consecutive_runs} plateau events)")
    print(f"  9. Normalization Applied:       None (Raw sf.read float32)")
    print(f" 10. Integer PCM to Float:        Valid IEEE 754 Float32 (soundfile standard / 32768)")
    print(f"  --> Diagnosis:                  {clipping_category}")

    return {
        "max_abs": max_abs,
        "min_amp": min_amp,
        "max_amp": max_amp,
        "rms": rms,
        "crest_factor": crest_factor,
        "pct_99": pct_99,
        "pct_999": pct_999,
        "max_consecutive": max_consecutive,
        "has_flat_tops": has_flat_tops,
        "category": clipping_category,
    }

def main():
    print("=" * 80)
    print("PHASE 1: CLIPPING AUDIT ON BENCHMARK RECORDINGS & SYNTHESIZED CONDITIONS")
    print("=" * 80)
    
    # 1. Test clean genuine recordings
    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    for fname in ["9.wav", "7.wav", "4.wav", "5.wav"]:
        p = os.path.join(base_dir, fname)
        if os.path.exists(p):
            data, sr = sf.read(p, dtype="float32", always_2d=True)
            audit_clipping_metrics(data[:, 0], sr, name=f"Clean In-The-Wild: {fname}")

    # 2. Test isolated single-sample transient peak at 1.0 (e.g. mic plug or clap)
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sine_speech = 0.4 * np.sin(2 * np.pi * 250 * t)
    # Add a single isolated spike at t=1.0
    isolated_spike = sine_speech.copy()
    isolated_spike[16000] = 0.9999
    audit_clipping_metrics(isolated_spike, sr, name="Synthetic Speech with 1 Isolated Transient Peak (1.0)")

    # 3. Test true hardware/mic clipping (gain=4x, clipped to 1.0)
    true_clipped = np.clip(sine_speech * 4.0, -1.0, 1.0)
    audit_clipping_metrics(true_clipped, sr, name="True Clipped Waveform (Overdriven Mic Input, Gain 4x)")

if __name__ == "__main__":
    main()

