"""Generates deterministic smoke test audio files for SonicShield pipeline verification.

NOTICE: These synthetic signals are strictly for local pipeline, unit test, and UI smoke testing.
They are NOT real human recordings, validated voice clones, or evidence of detector accuracy.
"""
from __future__ import annotations

import os
import numpy as np
import soundfile as sf

def generate_smoke_signals(output_dir: str = "data/smoke") -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    sr = 8000

    created_files = {}

    # 1. smoke_human_like.wav (7.0 seconds)
    # Simulated voiced speech segments with organic pitch modulation, room decay tail, and pause breath noise
    t_total = 7.0
    t = np.linspace(0, t_total, int(sr * t_total), endpoint=False)
    human_sig = np.zeros_like(t)

    # Active speech: 0.5s - 3.0s and 3.8s - 6.2s
    # Pitch contour with natural vibrato/intonation (130 Hz + 15 Hz vibrato at 5 Hz)
    f0 = 135.0 + 12.0 * np.sin(2 * np.pi * 3.5 * t) + 1.5 * np.random.normal(0, 1, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sr

    speech_mask_1 = (t >= 0.5) & (t <= 3.0)
    speech_mask_2 = (t >= 3.8) & (t <= 6.2)

    # Harmonics (1st to 5th formant approximation)
    vocal = (
        0.5 * np.sin(phase) +
        0.3 * np.sin(2 * phase) +
        0.18 * np.sin(3 * phase) +
        0.08 * np.sin(4 * phase)
    )

    # Apply speech envelope with room reverberation decay tail
    env = np.zeros_like(t)
    env[speech_mask_1] = 0.8
    env[speech_mask_2] = 0.8

    # Add simulated exponential room reverberation tail after 3.0s (decay over 300ms)
    reverb_tail_1 = (t > 3.0) & (t <= 3.4)
    dt_1 = (t[reverb_tail_1] - 3.0)
    env[reverb_tail_1] = 0.8 * np.exp(-dt_1 / 0.08)

    reverb_tail_2 = (t > 6.2) & (t <= 6.6)
    dt_2 = (t[reverb_tail_2] - 6.2)
    env[reverb_tail_2] = 0.8 * np.exp(-dt_2 / 0.08)

    # Simulated breath burst during pause at 3.3s - 3.65s (broadband noise, bandpassed)
    breath_mask = (t >= 3.35) & (t <= 3.65)
    noise = np.random.normal(0, 0.08, len(t))

    # Ambient room tone floor (~ -42 dBFS)
    room_tone = np.random.normal(0, 0.008, len(t))

    human_sig = vocal * env + noise * breath_mask + room_tone
    human_sig = np.clip(human_sig, -0.95, 0.95).astype(np.float32)

    path_human = os.path.join(output_dir, "smoke_human_like.wav")
    sf.write(path_human, human_sig, sr, subtype="PCM_16")
    created_files["human_like"] = path_human

    # 2. smoke_synthetic_like.wav (7.0 seconds)
    # Dry square/saw harmonic pulses, perfectly flat pitch, digital silence pauses, instant cutoffs
    f0_flat = 150.0
    phase_flat = 2 * np.pi * f0_flat * t
    synth_vocal = 0.5 * np.sin(phase_flat) + 0.3 * np.sin(2 * phase_flat) + 0.2 * np.sin(3 * phase_flat)

    synth_env = np.zeros_like(t)
    # Abrupt boxcar window with 0ms decay tail
    synth_env[(t >= 0.5) & (t <= 3.0)] = 0.8
    synth_env[(t >= 3.8) & (t <= 6.2)] = 0.8

    # Exact digital silence (0.000) during pause
    synth_sig = synth_vocal * synth_env
    synth_sig = np.clip(synth_sig, -0.95, 0.95).astype(np.float32)

    path_synth = os.path.join(output_dir, "smoke_synthetic_like.wav")
    sf.write(path_synth, synth_sig, sr, subtype="PCM_16")
    created_files["synthetic_like"] = path_synth

    # 3. smoke_short.wav (2.0 seconds)
    short_t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    short_sig = 0.5 * np.sin(2 * np.pi * 200.0 * short_t).astype(np.float32)
    path_short = os.path.join(output_dir, "smoke_short.wav")
    sf.write(path_short, short_sig, sr, subtype="PCM_16")
    created_files["short"] = path_short

    # 4. smoke_silence.wav (6.0 seconds)
    silence_sig = np.zeros(int(sr * 6.0), dtype=np.float32)
    path_silence = os.path.join(output_dir, "smoke_silence.wav")
    sf.write(path_silence, silence_sig, sr, subtype="PCM_16")
    created_files["silence"] = path_silence

    print("Generated smoke test files:")
    for k, v in created_files.items():
        print(f"  {k:15s}: {v}")

    return created_files

if __name__ == "__main__":
    generate_smoke_signals()
