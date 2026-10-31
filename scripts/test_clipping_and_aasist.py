"""Investigate clipping, amplitude levels, and AASIST inference behavior.
"""
import os
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from src.pretrained_aasist import get_aasist_classifier, pad_waveform
from src.pipeline import run_pipeline
from src.audio_utils import load_audio_dual

def analyze_clipping_statistics(wav: np.ndarray):
    """Calculates detailed clipping and amplitude metrics on a waveform."""
    abs_wav = np.abs(wav)
    max_amp = float(np.max(abs_wav)) if len(wav) > 0 else 0.0
    min_val = float(np.min(wav)) if len(wav) > 0 else 0.0
    max_val = float(np.max(wav)) if len(wav) > 0 else 0.0
    rms = float(np.sqrt(np.mean(wav**2))) if len(wav) > 0 else 0.0
    peak_to_rms = max_amp / rms if rms > 1e-9 else 0.0

    # Saturated samples
    count_99 = int(np.sum(abs_wav >= 0.99))
    count_999 = int(np.sum(abs_wav >= 0.999))
    pct_99 = (count_99 / len(wav)) * 100.0 if len(wav) > 0 else 0.0
    pct_999 = (count_999 / len(wav)) * 100.0 if len(wav) > 0 else 0.0

    # Consecutive full-scale samples
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

    return {
        "max_amp": max_amp,
        "min_val": min_val,
        "max_val": max_val,
        "rms": rms,
        "peak_to_rms": peak_to_rms,
        "count_99": count_99,
        "count_999": count_999,
        "pct_99": pct_99,
        "pct_999": pct_999,
        "max_consecutive_clipped": max_consecutive,
        "consecutive_flat_runs": consecutive_runs,
        "is_severely_clipped": max_consecutive >= 5 or pct_999 > 0.05,
        "is_peak_touch": max_amp >= 0.999 and max_consecutive < 5 and pct_999 <= 0.05,
    }

def main():
    aasist = get_aasist_classifier()
    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    test_files = [
        ("9.wav", "Alec Guinness (Genuine)"),
        ("7.wav", "Donald Trump (Genuine)"),
        ("4.wav", "Christopher Hitchens (Genuine)"),
        ("5.wav", "Ayn Rand (Genuine)"),
    ]

    print("=" * 110)
    print("EXPERIMENT 1: HOW AMPLITUDE AND CLIPPING AFFECT AASIST ON GENUINE RECORDINGS")
    print("=" * 110)

    for fname, desc in test_files:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            continue
        data, sr = sf.read(fpath, dtype="float32", always_2d=True)
        mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
        # Resample to 16k if needed
        if sr != 16000:
            import scipy.signal
            num_16k = int(round(len(mono) * 16000 / sr))
            mono = scipy.signal.resample(mono, num_16k).astype(np.float32)

        print(f"\n--- {fname}: {desc} (Original len={len(mono)}, max={np.max(np.abs(mono)):.4f}, rms={np.sqrt(np.mean(mono**2)):.4f}) ---")

        # Condition 1: Original untouched
        res_orig = aasist.predict(mono, sr=16000, codec="clean")
        print(f"  [1] Original Audio:            Bona={res_orig['bona_fide_score']:.4f} | Class={res_orig['predicted_class']:<10} | Logits={res_orig['raw_score']}")

        # Condition 2: Peak-normalized to 0.999 (near full scale, 0 dBFS)
        scaled_peak = (mono / (np.max(np.abs(mono)) + 1e-9)) * 0.999
        res_peak = aasist.predict(scaled_peak, sr=16000, codec="clean")
        print(f"  [2] Peak Normalized to 0.999:  Bona={res_peak['bona_fide_score']:.4f} | Class={res_peak['predicted_class']:<10} | Logits={res_peak['raw_score']}")

        # Condition 3: Hard-clipped (clipping top 10% of amplitude, hard saturated at 0.8)
        hard_clip = np.clip(mono * 2.0, -0.999, 0.999)
        res_clip = aasist.predict(hard_clip, sr=16000, codec="clean")
        stats_clip = analyze_clipping_statistics(hard_clip)
        print(f"  [3] Hard-Clipped (gain 2x):    Bona={res_clip['bona_fide_score']:.4f} | Class={res_clip['predicted_class']:<10} | Logits={res_clip['raw_score']}")
        print(f"      Clipping Stats: pct_999={stats_clip['pct_999']:.3f}%, max_consec={stats_clip['max_consecutive_clipped']}, flat_runs={stats_clip['consecutive_flat_runs']}")

        # Condition 4: Attenuated / low volume (-12 dB, gain=0.25)
        low_vol = mono * 0.25
        res_low = aasist.predict(low_vol, sr=16000, codec="clean")
        print(f"  [4] Attenuated (-12 dB, 0.25x): Bona={res_low['bona_fide_score']:.4f} | Class={res_low['predicted_class']:<10} | Logits={res_low['raw_score']}")

        # Condition 5: Very low volume (-24 dB, gain=0.0625)
        very_low = mono * 0.0625
        res_vlow = aasist.predict(very_low, sr=16000, codec="clean")
        print(f"  [5] Very Low (-24 dB, 0.06x):  Bona={res_vlow['bona_fide_score']:.4f} | Class={res_vlow['predicted_class']:<10} | Logits={res_vlow['raw_score']}")

if __name__ == "__main__":
    main()

