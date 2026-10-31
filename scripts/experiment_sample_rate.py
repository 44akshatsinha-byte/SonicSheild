"""Controlled experimental evaluation of AASIST sample-rate downsampling/upsampling effect.

Compares:
Condition A: Original 16 kHz audio directly to AASIST (no 8 kHz bottleneck)
Condition B: Downsampled to 8 kHz, then upsampled to 16 kHz before AASIST
Condition C: Current SonicShield Clean pipeline
Condition D: Current SonicShield G.711 pipeline
"""
import os
import sys
import time
import io
import numpy as np
import soundfile as sf
import scipy.signal

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from src.pretrained_aasist import get_aasist_classifier, AASIST_SAMPLE_RATE, AASIST_NB_SAMPLES, pad_waveform
from src.audio_utils import load_audio
from src.pipeline import run_pipeline

def run_controlled_experiment(wav_path: str, label: str):
    print("=" * 80)
    print(f"EXPERIMENT ON: {os.path.basename(wav_path)} ({label})")
    print("=" * 80)

    aasist = get_aasist_classifier()
    assert aasist.is_loaded, "AASIST model failed to load!"

    # Load original audio at native rate
    data_orig, sr_orig = sf.read(wav_path, dtype="float32", always_2d=True)
    if data_orig.shape[1] > 1:
        mono_orig = np.mean(data_orig, axis=1)
    else:
        mono_orig = data_orig[:, 0]

    # If original is not 16k, resample directly to 16k once
    if sr_orig != 16000:
        num_target = int(round(len(mono_orig) * 16000 / sr_orig))
        wav_16k_native = scipy.signal.resample(mono_orig, num_target).astype(np.float32)
    else:
        wav_16k_native = mono_orig.astype(np.float32)

    # -------------------------------------------------------------
    # Condition A: Original 16 kHz waveform directly into AASIST
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    res_a = aasist.predict(wav_16k_native, sr=16000, codec="native_16k")
    time_a = (time.perf_counter() - t0) * 1000

    rms_a = float(np.sqrt(np.mean(wav_16k_native**2)))
    peak_a = float(np.max(np.abs(wav_16k_native)))
    finite_a = int(np.sum(np.isfinite(wav_16k_native)))

    print("\n--- CONDITION A: Native 16 kHz Direct (No 8 kHz bottleneck) ---")
    print(f"Actual Sample Rate:  16000 Hz")
    print(f"Waveform Shape:      {wav_16k_native.shape}")
    print(f"Audio Duration:      {len(wav_16k_native)/16000:.3f} s")
    print(f"RMS Energy:          {rms_a:.6f}")
    print(f"Peak Amplitude:      {peak_a:.6f}")
    print(f"Finite Samples:      {finite_a} / {len(wav_16k_native)}")
    print(f"Raw AASIST Logits:   {res_a.get('raw_score')}")
    print(f"Bona-Fide Score:     {res_a.get('bona_fide_score')}")
    print(f"Spoof Score:         {round(1.0 - res_a.get('bona_fide_score', 0), 4)}")
    print(f"Predicted Class:     {res_a.get('predicted_class')}")
    print(f"Inference Time:      {time_a:.2f} ms")

    # -------------------------------------------------------------
    # Condition B: Downsampled to 8 kHz, then upsampled to 16 kHz
    # -------------------------------------------------------------
    num_8k = int(round(len(wav_16k_native) * 8000 / 16000))
    wav_8k = scipy.signal.resample(wav_16k_native, num_8k).astype(np.float32)
    # Upsampled using polyphase (matches current prepare_audio_for_aasist)
    wav_16k_upsampled = scipy.signal.resample_poly(wav_8k, 2, 1).astype(np.float32)

    t0 = time.perf_counter()
    res_b = aasist.predict(wav_8k, sr=8000, codec="8k_to_16k")
    time_b = (time.perf_counter() - t0) * 1000

    rms_b = float(np.sqrt(np.mean(wav_16k_upsampled**2)))
    peak_b = float(np.max(np.abs(wav_16k_upsampled)))
    finite_b = int(np.sum(np.isfinite(wav_16k_upsampled)))

    print("\n--- CONDITION B: 8 kHz Downsampled -> 16 kHz Upsampled ---")
    print(f"Actual Sample Rate:  8000 Hz -> 16000 Hz (polyphase)")
    print(f"Waveform Shape:      {wav_16k_upsampled.shape}")
    print(f"Audio Duration:      {len(wav_16k_upsampled)/16000:.3f} s")
    print(f"RMS Energy:          {rms_b:.6f}")
    print(f"Peak Amplitude:      {peak_b:.6f}")
    print(f"Finite Samples:      {finite_b} / {len(wav_16k_upsampled)}")
    print(f"Raw AASIST Logits:   {res_b.get('raw_score')}")
    print(f"Bona-Fide Score:     {res_b.get('bona_fide_score')}")
    print(f"Spoof Score:         {round(1.0 - res_b.get('bona_fide_score', 0), 4)}")
    print(f"Predicted Class:     {res_b.get('predicted_class')}")
    print(f"Inference Time:      {time_b:.2f} ms")

    # -------------------------------------------------------------
    # Condition C: Current SonicShield Clean Pipeline
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    res_c = run_pipeline(wav_path, codec="clean", detection_mode="pretrained")
    time_c = (time.perf_counter() - t0) * 1000

    print("\n--- CONDITION C: Current SonicShield Clean Pipeline ---")
    print(f"Reported Codec:      {res_c.get('codec')}")
    print(f"Overall Liveness:    {res_c.get('overall_liveness')} / 100")
    print(f"AASIST Bona-Fide:    {res_c.get('overall_aasist_bona_fide_score')}")
    print(f"AASIST Class:        {res_c.get('overall_aasist_class')}")
    print(f"Raw AASIST Logits:   {res_c['aasist'].get('raw_score')}")
    print(f"Physics Mean:        {res_c.get('physics_mean')}")
    print(f"Biology Mean:        {res_c.get('biology_mean')}")
    print(f"Pipeline Time:       {time_c:.2f} ms")

    # -------------------------------------------------------------
    # Condition D: Current SonicShield G.711 Pipeline
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    res_d = run_pipeline(wav_path, codec="g711", detection_mode="pretrained")
    time_d = (time.perf_counter() - t0) * 1000

    print("\n--- CONDITION D: Current SonicShield G.711 Pipeline ---")
    print(f"Reported Codec:      {res_d.get('codec')}")
    print(f"Overall Liveness:    {res_d.get('overall_liveness')} / 100")
    print(f"AASIST Bona-Fide:    {res_d.get('overall_aasist_bona_fide_score')}")
    print(f"AASIST Class:        {res_d.get('overall_aasist_class')}")
    print(f"Raw AASIST Logits:   {res_d['aasist'].get('raw_score')}")
    print(f"Physics Mean:        {res_d.get('physics_mean')}")
    print(f"Biology Mean:        {res_d.get('biology_mean')}")
    print(f"Pipeline Time:       {time_d:.2f} ms")

if __name__ == "__main__":
    test_files = [
        ("data/external/in_the_wild/release_in_the_wild/release_in_the_wild/9.wav", "Genuine Human (Alec Guinness, 7.9s)"),
        ("data/external/in_the_wild/release_in_the_wild/release_in_the_wild/8.wav", "Genuine Human (Donald Trump, 5.1s)"),
    ]
    for p, lbl in test_files:
        if os.path.exists(p):
            run_controlled_experiment(p, lbl)
        else:
            print(f"File not found: {p}")
