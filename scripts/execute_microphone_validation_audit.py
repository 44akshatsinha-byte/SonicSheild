"""Comprehensive Validation & QA Audit for SonicShield Microphone Detection Pipeline.

Executes Phases 2, 3, and 4 with real metrics, timers, and signal quality audits.
"""
from __future__ import annotations

import io
import os
import sys
import time
import math
import tempfile
import numpy as np
import soundfile as sf
import scipy.signal

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.audio_utils import (
    load_audio,
    load_audio_dual,
    validate_recording_bytes,
    analyze_clipping_telemetry,
    safe_temporary_wav,
    AudioValidationError,
)
from src.pipeline import run_pipeline
from src.pretrained_aasist import get_aasist_classifier


def create_acoustic_replay_simulation(clean_wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulates physical loudspeaker playback captured by an in-room microphone.
    
    Models:
    1. Loudspeaker bandpass / resonant acoustic response (150 Hz - 7000 Hz).
    2. Room Impulse Response (RIR) with exponential decay reverberation (RT60 ~ 300ms).
    3. Low-level ambient room noise (SNR ~ 32 dB).
    4. Subtle speaker driver harmonic saturation.
    """
    # 1. Loudspeaker bandpass filter
    sos = scipy.signal.butter(4, [150, 7000], btype="bandpass", fs=sr, output="sos")
    filtered = scipy.signal.sosfilt(sos, clean_wav)

    # 2. Synthetic Room Impulse Response (300 ms reverberation)
    rir_len = int(sr * 0.30)
    t = np.linspace(0, 0.30, rir_len)
    decay = np.exp(-t / 0.05)  # exponential decay
    np.random.seed(42)
    noise = np.random.randn(rir_len)
    rir = noise * decay
    # Add early reflections
    for delay_ms, amp in [(15, 0.4), (28, 0.25), (45, 0.15)]:
        idx = int(sr * delay_ms / 1000)
        if idx < rir_len:
            rir[idx] += amp
    rir = rir / np.max(np.abs(rir))

    # Convolve with RIR
    reverbed = scipy.signal.fftconvolve(filtered, rir, mode="same")

    # 3. Soft speaker driver saturation
    saturated = np.tanh(reverbed * 1.2) / 1.2

    # 4. Ambient acoustic room noise (approx 32 dB SNR)
    sig_power = np.mean(saturated ** 2)
    noise_power = sig_power / (10 ** (32 / 10))
    ambient = np.random.randn(len(saturated)) * np.sqrt(noise_power)
    replayed = saturated + ambient

    # Peak normalize to safe microphone level
    peak = np.max(np.abs(replayed))
    if peak > 0:
        replayed = (replayed / peak) * 0.82

    return replayed.astype(np.float32)


def run_phase_2_microphone_testing():
    """Phase 2: Evaluate 5 distinct acoustic scenarios."""
    print("\n" + "="*80)
    print("PHASE 2: MICROPHONE SCENARIO TESTING")
    print("="*80)

    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    aasist = get_aasist_classifier()

    # Load base files
    path_human = os.path.join(base_dir, "9.wav")   # Alec Guinness (bona-fide)
    path_spoof = os.path.join(base_dir, "0.wav")   # Alec Guinness (spoof)
    
    data_human, sr_human = sf.read(path_human, dtype="float32")
    if data_human.ndim > 1:
        data_human = data_human.mean(axis=1)
    if sr_human != 16000:
        data_human = scipy.signal.resample(data_human, int(round(len(data_human) * 16000 / sr_human)))

    data_spoof, sr_spoof = sf.read(path_spoof, dtype="float32")
    if data_spoof.ndim > 1:
        data_spoof = data_spoof.mean(axis=1)
    if sr_spoof != 16000:
        data_spoof = scipy.signal.resample(data_spoof, int(round(len(data_spoof) * 16000 / sr_spoof)))

    # Scenario 1: Clean Human Speech (Standard Mic Gain)
    s1_wav = data_human.copy()
    s1_wav = (s1_wav / np.max(np.abs(s1_wav))) * 0.85

    # Scenario 2: Human Speech with High Mic Gain (Severe ADC Rail Clipping)
    s2_wav = np.clip(data_human * 4.0, -1.0, 1.0)

    # Scenario 3: Human Speech with Low Mic Gain (Faint, Under-Driven)
    s3_wav = (data_human / np.max(np.abs(data_human))) * 0.035

    # Scenario 4: Synthetic Speech (Direct AI-Generated Sample)
    s4_wav = (data_spoof / np.max(np.abs(data_spoof))) * 0.85

    # Scenario 5: Synthetic Speech Re-recorded Through Physical Speaker & Mic (Acoustic Replay)
    s5_wav = create_acoustic_replay_simulation(data_spoof, sr=16000)

    scenarios = [
        ("SCEN-1-CLEAN-HUMAN", "Genuine Human", s1_wav, "Alec Guinness 9.wav (Clean Gain)"),
        ("SCEN-2-HIGH-GAIN-HUMAN", "Genuine Human", s2_wav, "Alec Guinness 9.wav (4.0x Gain Overdrive)"),
        ("SCEN-3-LOW-GAIN-HUMAN", "Genuine Human", s3_wav, "Alec Guinness 9.wav (0.035x Faint Gain)"),
        ("SCEN-4-SYNTHETIC-AI", "Synthetic / Clone", s4_wav, "Alec Guinness 0.wav (Direct AI Spoof)"),
        ("SCEN-5-REPLAY-SPEAKER-MIC", "Synthetic / Clone", s5_wav, "Alec Guinness 0.wav (Speaker-to-Mic Replay)"),
    ]

    results = []

    for sc_id, gt_label, wav, desc in scenarios:
        # Create in-memory WAV bytes
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        # Telemetry
        duration = len(wav) / 16000.0
        peak = float(np.max(np.abs(wav)))
        rms = float(np.sqrt(np.mean(wav**2)))
        clip_telemetry = analyze_clipping_telemetry(wav)
        clip_pct = clip_telemetry["pct_999"]

        # Run pipeline and measure precise inference time
        t_start = time.perf_counter()
        pipeline_res = run_pipeline(wav_bytes, codec="clean", detection_mode="pretrained")
        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        bona_score = pipeline_res["overall_aasist_bona_fide_score"]
        spoof_score = 1.0 - bona_score if bona_score is not None else None
        phys_score = int(round(pipeline_res["physics_mean"] * 100))
        bio_score = int(round(pipeline_res["biology_mean"] * 100))
        final_pred = pipeline_res["overall_aasist_class"]
        overall_index = pipeline_res["overall_liveness"]

        rec = {
            "id": sc_id,
            "description": desc,
            "ground_truth": gt_label,
            "duration": round(duration, 2),
            "sample_rate": 16000,
            "peak_amplitude": round(peak, 4),
            "rms": round(rms, 6),
            "clipping_pct": round(clip_pct, 4),
            "max_consecutive_rail": clip_telemetry["max_consecutive_rail"],
            "is_clipped": clip_telemetry["is_clipped"],
            "aasist_bona": round(bona_score, 4) if bona_score is not None else None,
            "aasist_spoof": round(spoof_score, 4) if spoof_score is not None else None,
            "physics_score": phys_score,
            "biology_score": bio_score,
            "final_prediction": final_pred,
            "overall_index": overall_index,
            "inference_time_ms": round(inference_time_ms, 1),
        }
        results.append(rec)

        print(f"\n[{sc_id}] {desc}")
        print(f"  Ground Truth:      {gt_label}")
        print(f"  Duration:          {duration:.2f}s | Sample Rate: 16000 Hz")
        print(f"  Peak Amp:          {peak:.4f} | RMS: {rms:.6f} | Clipping: {clip_pct:.2f}% (Consec: {clip_telemetry['max_consecutive_rail']})")
        print(f"  AASIST Bona-Fide:  {bona_score:.4f} | Spoof: {spoof_score:.4f}")
        print(f"  Physics Score:     {phys_score}/100 | Biology Score: {bio_score}/100")
        print(f"  Final Prediction:  {final_pred} (Overall Index: {overall_index}/100)")
        print(f"  Inference Latency: {inference_time_ms:.1f} ms")

    return results


def run_phase_3_signal_quality_audit():
    """Phase 3: Rigorous signal quality, edge cases, and safety checks."""
    print("\n" + "="*80)
    print("PHASE 3: SIGNAL QUALITY & BOUNDARY AUDIT")
    print("="*80)

    audit_findings = {}

    # 1. Resampling Gibbs Overshoot vs True Clipping
    # Create sinusoidal burst just touching 0.94, resample up/down, check analyze_clipping_telemetry
    t = np.linspace(0, 0.5, 4000)
    sine_wave = 0.95 * np.sin(2 * np.pi * 440 * t)
    # Upsample to induce Gibbs overshoot
    overshot = scipy.signal.resample(sine_wave, 8000)
    telemetry_overshoot = analyze_clipping_telemetry(overshot)
    audit_findings["gibbs_overshoot_check"] = {
        "max_abs": telemetry_overshoot["max_abs"],
        "is_clipped": telemetry_overshoot["is_clipped"],
        "is_transient_peak": telemetry_overshoot["is_transient_peak"],
        "verdict": "PASS: Resampling transient not flagged as hard clipping" if not telemetry_overshoot["is_clipped"] else "FAIL",
    }
    print(f"1. Gibbs Overshoot Distinction: max={telemetry_overshoot['max_abs']:.4f}, is_clipped={telemetry_overshoot['is_clipped']}, is_transient={telemetry_overshoot['is_transient_peak']}")

    # 2. Low-Volume Recording Warning Check
    low_vol = np.random.randn(16000 * 3) * 0.0003  # RMS ~ 0.0003
    buf_low = io.BytesIO()
    sf.write(buf_low, low_vol, 16000, format="WAV", subtype="PCM_16")
    meta_low, warns_low, err_low = validate_recording_bytes(buf_low.getvalue())
    has_low_vol_warn = any("Low microphone volume" in w for w in warns_low)
    audit_findings["low_volume_check"] = {
        "rms": meta_low["rms"],
        "warning_issued": has_low_vol_warn,
        "warnings": warns_low,
        "verdict": "PASS" if has_low_vol_warn else "FAIL",
    }
    print(f"2. Low-Volume Warning: rms={meta_low['rms']:.6f}, warning_issued={has_low_vol_warn}")

    # 3. Digital Silence Detection
    silence = np.zeros(16000 * 3, dtype=np.float32)
    buf_sil = io.BytesIO()
    sf.write(buf_sil, silence, 16000, format="WAV", subtype="PCM_16")
    meta_sil, warns_sil, err_sil = validate_recording_bytes(buf_sil.getvalue())
    has_sil_warn = any("silence detected" in w.lower() for w in warns_sil)
    audit_findings["silence_check"] = {
        "rms": meta_sil["rms"],
        "warning_issued": has_sil_warn,
        "verdict": "PASS" if has_sil_warn else "FAIL",
    }
    print(f"3. Silence Detection: rms={meta_sil['rms']:.6f}, warning_issued={has_sil_warn}")

    # 4. Invalid Audio Rejection
    invalid_bytes = b"NOT_A_WAV_HEADER_CORRUPTED_STREAM"
    meta_inv, warns_inv, err_inv = validate_recording_bytes(invalid_bytes)
    audit_findings["invalid_audio_check"] = {
        "rejected": err_inv is not None,
        "error_message": err_inv,
        "verdict": "PASS" if err_inv is not None else "FAIL",
    }
    print(f"4. Invalid Audio Rejection: rejected={err_inv is not None}, error='{err_inv}'")

    # 5. Stereo Audio Handling Check
    stereo_data = np.random.randn(16000 * 3, 2).astype(np.float32) * 0.2
    buf_stereo = io.BytesIO()
    sf.write(buf_stereo, stereo_data, 16000, format="WAV", subtype="PCM_16")
    meta_stereo, warns_stereo, err_stereo = validate_recording_bytes(buf_stereo.getvalue())
    has_mono_downmix = any("downmixed to mono" in w for w in warns_stereo)
    audit_findings["stereo_handling_check"] = {
        "channels_detected": meta_stereo["channels"],
        "downmixed": has_mono_downmix,
        "verdict": "PASS" if (meta_stereo["channels"] == 2 and has_mono_downmix) else "FAIL",
    }
    print(f"5. Stereo Downmixing: channels={meta_stereo['channels']}, downmixed_warning={has_mono_downmix}")

    # 6. Temporary File Deletion Verification
    temp_file_deleted = False
    temp_target_path = None
    with safe_temporary_wav(b"RIFFdummydataWAVEfmt ", prefix="audit_test_") as tpath:
        temp_target_path = tpath
        file_existed_in_context = os.path.exists(tpath)
    file_exists_after_context = os.path.exists(temp_target_path)
    audit_findings["temp_file_cleanup"] = {
        "existed_during": file_existed_in_context,
        "exists_after": file_exists_after_context,
        "verdict": "PASS: Temporary file safely deleted" if file_existed_in_context and not file_exists_after_context else "FAIL",
    }
    print(f"6. Temporary WAV File Cleanup: existed_in_context={file_existed_in_context}, exists_after_exit={file_exists_after_context}")

    return audit_findings


def run_phase_4_parity_check():
    """Phase 4: Exact parity verification between Direct, Clean Pipeline, and Mic Temporary WAV."""
    print("\n" + "="*80)
    print("PHASE 4: MULTI-PATHWAY PARITY COMPARISON")
    print("="*80)

    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    aasist = get_aasist_classifier()
    
    test_files = [
        (os.path.join(base_dir, "9.wav"), "Alec Guinness Genuine Speech"),
        (os.path.join(base_dir, "7.wav"), "Donald Trump Genuine Speech"),
        (os.path.join(base_dir, "0.wav"), "Alec Guinness Synthetic Speech"),
        (os.path.join(base_dir, "2.wav"), "Barack Obama Synthetic Speech"),
    ]

    parity_records = []

    for fpath, label in test_files:
        with open(fpath, "rb") as f:
            raw_bytes = f.read()

        data, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        mono = data[:, 0]
        if orig_sr != 16000:
            mono = scipy.signal.resample(mono, int(round(len(mono) * 16000 / orig_sr)))

        # Pathway A: Direct AASIST Inference
        res_a = aasist.predict(mono, sr=16000, codec="clean")
        score_a = res_a["bona_fide_score"]

        # Pathway B: Clean Upload Pipeline
        res_b = run_pipeline(raw_bytes, codec="clean", detection_mode="pretrained")
        score_b = res_b["overall_aasist_bona_fide_score"]

        # Pathway C: Microphone Temporary-WAV Pipeline
        with safe_temporary_wav(raw_bytes, prefix="parity_mic_") as temp_p:
            res_c = run_pipeline(temp_p, codec="clean", detection_mode="pretrained")
            score_c = res_c["overall_aasist_bona_fide_score"]

        diff_ab = abs(score_a - score_b)
        diff_bc = abs(score_b - score_c)
        exact_match = (diff_ab < 1e-6) and (diff_bc < 1e-6)

        rec = {
            "file": os.path.basename(fpath),
            "label": label,
            "score_a_direct": round(score_a, 6),
            "score_b_clean_pipeline": round(score_b, 6),
            "score_c_mic_temp_wav": round(score_c, 6),
            "diff_ab": diff_ab,
            "diff_bc": diff_bc,
            "exact_match": exact_match,
        }
        parity_records.append(rec)

        print(f"[{os.path.basename(fpath)}] {label}:")
        print(f"  Path A (Direct AASIST):        {score_a:.6f}")
        print(f"  Path B (Clean Pipeline):       {score_b:.6f}")
        print(f"  Path C (Mic Temporary WAV):    {score_c:.6f}")
        print(f"  |A - B| = {diff_ab:.8f} | |B - C| = {diff_bc:.8f} -> {'EXACT 100% PARITY' if exact_match else 'DISCREPANCY'}")

    return parity_records


if __name__ == "__main__":
    p2_results = run_phase_2_microphone_testing()
    p3_results = run_phase_3_signal_quality_audit()
    p4_results = run_phase_4_parity_check()
    print("\nAUDIT EXECUTION COMPLETE.")
