"""Forensic reproduction test script for SonicShield zero scores.

Tests at least:
- 5 known genuine speech samples
- 5 known AI-generated speech samples
- 1 browser-recorded microphone sample (simulated via 16 kHz WAV)
- 1 uploaded WAV sample
- 1 compressed audio sample (MP3 / compressed)
- 1 WebM/Opus or simulated compressed recording
"""
from __future__ import annotations

import io
import os
import sys
import time
import numpy as np
import soundfile as sf
import scipy.signal

sys.path.insert(0, os.path.abspath("."))

from src.audio_utils import load_audio, load_audio_dual, validate_recording_bytes, safe_temporary_wav
from src.pipeline import run_pipeline
from src.pretrained_aasist import get_aasist_classifier


def run_reproduction_suite():
    print("=" * 80)
    print("SONICSHIELD P0 ZERO SCORE REPRODUCTION SUITE")
    print("=" * 80)

    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    aasist = get_aasist_classifier()

    # 5 genuine samples
    genuine_files = ["9.wav", "7.wav", "4.wav", "5.wav", "11.wav"]
    # 5 spoof samples
    spoof_files = ["0.wav", "1.wav", "2.wav", "3.wav", "6.wav"]

    samples_to_test = []

    for f in genuine_files:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            samples_to_test.append((p, "Genuine Speech", f"InTheWild Genuine: {f}"))

    for f in spoof_files:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            samples_to_test.append((p, "Synthetic / Spoof", f"InTheWild Spoof: {f}"))

    # 1 browser-recorded microphone sample (16 kHz WAV with typical room acoustics)
    mic_path = os.path.join("data", "smoke", "smoke_human_like.wav")
    samples_to_test.append((mic_path, "Microphone Voice", "Browser Mic Simulated 16k"))

    # 1 uploaded WAV sample
    samples_to_test.append((os.path.join("data", "smoke", "smoke_synthetic_like.wav"), "Synthetic Upload", "Uploaded WAV Sample"))

    # Create 1 compressed sample (e.g. simulated GSM/telephony compressed or OGG/MP3 if available)
    # Generate MP3 or OGG in memory using soundfile
    data_tmp, sr_tmp = sf.read(os.path.join(base_dir, "9.wav"), dtype="float32")
    if data_tmp.ndim > 1:
        data_tmp = data_tmp.mean(axis=1)
    
    # Save as compressed OGG in memory
    buf_ogg = io.BytesIO()
    sf.write(buf_ogg, data_tmp[:16000*4], 16000, format="OGG", subtype="VORBIS")
    samples_to_test.append((buf_ogg.getvalue(), "Compressed Audio (OGG Vorbis)", "Compressed Audio OGG"))

    # WebM / Opus container simulation
    # If soundfile cannot decode raw WebM without external codec, test how the system reacts
    samples_to_test.append((b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04WebMCorruptOrRawTestBytes", "WebM Container", "Raw WebM Bytes"))

    report_entries = []

    for item, category, label in samples_to_test:
        print(f"\n--- Testing: {label} [{category}] ---")
        errors = []
        is_bytes = isinstance(item, bytes)

        # 1. Ingress Decoding
        try:
            if is_bytes:
                data_raw, orig_sr = sf.read(io.BytesIO(item), dtype="float32", always_2d=True)
            else:
                data_raw, orig_sr = sf.read(item, dtype="float32", always_2d=True)
            channels = data_raw.shape[1]
            mono = data_raw[:, 0] if channels > 1 else data_raw[:, 0]
            duration = len(mono) / float(orig_sr)
            min_amp = float(np.min(mono))
            max_amp = float(np.max(mono))
            rms = float(np.sqrt(np.mean(mono ** 2)))
            pct_clipped = float(np.mean(np.abs(mono) >= 0.999) * 100.0)
            decoded_ok = True
        except Exception as e:
            errors.append(f"Decoding failed: {e}")
            decoded_ok = False
            duration = 0.0
            orig_sr = 0
            channels = 0
            min_amp = max_amp = rms = pct_clipped = 0.0
            mono = np.array([], dtype=np.float32)

        # 2. Pipeline Execution
        pipeline_res = None
        if decoded_ok:
            try:
                pipeline_res = run_pipeline(
                    file_source=item,
                    codec="clean",
                    mode="both",
                    detection_mode="pretrained",
                )
            except Exception as e:
                errors.append(f"Pipeline execution failed: {e}")

        # 3. Direct AASIST Execution (Pathway A)
        direct_res = None
        if decoded_ok:
            try:
                if orig_sr != 16000:
                    num_target = int(round(len(mono) * 16000 / orig_sr))
                    mono_16k = scipy.signal.resample(mono, num_target).astype(np.float32)
                else:
                    mono_16k = mono.astype(np.float32)
                direct_res = aasist.predict(mono_16k, sr=16000, codec="clean")
            except Exception as e:
                errors.append(f"Direct AASIST failed: {e}")

        # Extract values
        raw_logits = direct_res.get("raw_score") if direct_res else None
        bona_score = direct_res.get("bona_fide_score") if direct_res else None
        liveness_score = direct_res.get("liveness_score") if direct_res else None
        pipeline_liveness = pipeline_res.get("overall_liveness") if pipeline_res else None
        pipeline_baseline = pipeline_res.get("overall_baseline_liveness") if pipeline_res else None
        pipeline_aasist_liveness = pipeline_res.get("overall_aasist_liveness") if pipeline_res else None
        pipeline_aasist_bona = pipeline_res.get("overall_aasist_bona_fide_score") if pipeline_res else None

        print(f"  Decoded: SR={orig_sr} Hz, Dur={duration:.2f}s, Chans={channels}, Peak={max_amp:.4f}, RMS={rms:.6f}, Clip={pct_clipped:.2f}%")
        print(f"  Direct AASIST Logits: {raw_logits} | Bona: {bona_score} | Liveness: {liveness_score}")
        print(f"  Pipeline Overall Liveness: {pipeline_liveness} (AASIST: {pipeline_aasist_liveness}, Baseline: {pipeline_baseline})")
        if errors:
            print(f"  ERRORS: {errors}")

        entry = {
            "label": label,
            "category": category,
            "orig_sr": orig_sr,
            "duration": round(duration, 2),
            "channels": channels,
            "peak": round(max_amp, 4),
            "rms": round(rms, 6),
            "pct_clipped": round(pct_clipped, 4),
            "raw_logits": raw_logits,
            "direct_bona": bona_score,
            "direct_liveness": liveness_score,
            "pipeline_liveness": pipeline_liveness,
            "pipeline_aasist_liveness": pipeline_aasist_liveness,
            "pipeline_baseline": pipeline_baseline,
            "pipeline_aasist_bona": pipeline_aasist_bona,
            "errors": errors,
        }
        report_entries.append(entry)

    return report_entries

if __name__ == "__main__":
    run_reproduction_suite()
