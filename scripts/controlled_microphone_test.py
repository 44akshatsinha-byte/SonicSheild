"""Phase 5: Controlled test of audio pathways (Conditions A, B, C, D).

A. Original WAV, native 16 kHz, directly to AASIST
B. Same audio through Clean pipeline
C. Same audio after G.711 processing & resampling
D. Same audio through simulated upload interface
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import numpy as np
import soundfile as sf
import scipy.signal

from src.pretrained_aasist import get_aasist_classifier
from src.pipeline import run_pipeline
from src.audio_utils import safe_temporary_wav, load_audio_dual

def run_controlled_test(audio_path: str, label: str):
    aasist = get_aasist_classifier()
    
    with open(audio_path, "rb") as f:
        raw_bytes = f.read()

    data, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    mono = data[:, 0] if data.shape[1] > 1 else data[:, 0]
    
    print(f"\n================================================================================")
    print(f"CONTROLLED TEST FOR: {label} ({os.path.basename(audio_path)})")
    print(f"Header SR: {orig_sr} Hz | Duration: {len(mono)/orig_sr:.2f}s | Max Amp: {np.max(np.abs(mono)):.4f} | RMS: {np.sqrt(np.mean(mono**2)):.4f}")
    print(f"================================================================================")

    # Condition A: Native 16 kHz audio directly to AASIST model
    if orig_sr != 16000:
        num_16k = int(round(len(mono) * 16000 / orig_sr))
        wav_16k_direct = scipy.signal.resample(mono, num_16k).astype(np.float32)
    else:
        wav_16k_direct = mono.astype(np.float32)

    res_a = aasist.predict(wav_16k_direct, sr=16000, codec="clean")
    print("Condition A (Direct Native 16 kHz):")
    print(f"  Class:       {res_a['predicted_class']}")
    print(f"  Bona Score:  {res_a['bona_fide_score']:.4f}")
    print(f"  Spoof Score: {1.0 - res_a['bona_fide_score']:.4f}")
    print(f"  Raw Logits:  {res_a['raw_score']}")

    # Condition B: Through the Clean processing pipeline
    res_b = run_pipeline(raw_bytes, codec="clean", detection_mode="pretrained")
    print("\nCondition B (Clean Processing Pipeline):")
    print(f"  Class:       {res_b['overall_aasist_class']}")
    print(f"  Bona Score:  {res_b['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Spoof Score: {1.0 - res_b['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Raw Logits:  {res_b['aasist']['raw_score']}")
    print(f"  Overall Score Index: {res_b['overall_liveness']}/100")

    # Condition C: Through G.711 processing & resampling
    res_c = run_pipeline(raw_bytes, codec="g711", detection_mode="pretrained")
    print("\nCondition C (G.711 Telephony Pipeline):")
    print(f"  Class:       {res_c['overall_aasist_class']}")
    print(f"  Bona Score:  {res_c['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Spoof Score: {1.0 - res_c['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Raw Logits:  {res_c['aasist']['raw_score']}")
    print(f"  Overall Score Index: {res_c['overall_liveness']}/100")

    # Condition D: Through safe_temporary_wav (Microphone workflow simulation)
    with safe_temporary_wav(raw_bytes, prefix="test_cond_d_") as temp_p:
        res_d = run_pipeline(temp_p, codec="clean", detection_mode="pretrained")
    print("\nCondition D (Upload / Temp-WAV Interface):")
    print(f"  Class:       {res_d['overall_aasist_class']}")
    print(f"  Bona Score:  {res_d['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Spoof Score: {1.0 - res_d['overall_aasist_bona_fide_score']:.4f}")
    print(f"  Raw Logits:  {res_d['aasist']['raw_score']}")
    print(f"  Overall Score Index: {res_d['overall_liveness']}/100")

    # Verification of parity
    diff_ab = abs(res_a['bona_fide_score'] - res_b['overall_aasist_bona_fide_score'])
    diff_bd = abs(res_b['overall_aasist_bona_fide_score'] - res_d['overall_aasist_bona_fide_score'])
    print(f"\nDiscrepancy Check:")
    print(f"  |Score A - Score B| = {diff_ab:.6f} ({'EXACT MATCH' if diff_ab < 1e-4 else 'DISCREPANCY'})")
    print(f"  |Score B - Score D| = {diff_bd:.6f} ({'EXACT MATCH' if diff_bd < 1e-4 else 'DISCREPANCY'})")

def main():
    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    test_files = [
        (os.path.join(base_dir, "9.wav"), "Alec Guinness (Genuine Speech)"),
        (os.path.join(base_dir, "7.wav"), "Donald Trump (Genuine Speech)"),
        (os.path.join(base_dir, "4.wav"), "Christopher Hitchens (Genuine Speech)"),
        (os.path.join("data", "smoke", "smoke_human_like.wav"), "Synthetic Human-like Benchmark Smoke File"),
    ]

    for fpath, label in test_files:
        if os.path.exists(fpath):
            run_controlled_test(fpath, label)

if __name__ == "__main__":
    main()
