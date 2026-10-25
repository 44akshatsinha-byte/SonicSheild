"""SonicShield Unified Acoustic Processing Pipeline."""
from __future__ import annotations

import io
import time
import numpy as np

from src.audio_utils import load_audio, AudioValidationError
from src.windows import generate_windows, AudioWindow
from src.codec_crusher import crush, CodecUnavailableError
from src.physics_engine import analyze as analyze_physics
from src.biology_engine import analyze as analyze_biology
from src.scorer import score_window, get_band, get_band_label

class PipelineError(Exception):
    """Raised when an unrecoverable pipeline failure occurs."""
    pass

def run_pipeline(
    file_source: str | io.BytesIO | bytes | np.ndarray,
    codec: str = "clean",
    mode: str = "both",
    sr: int = 8000,
    packet_loss_rate: float = 0.0,
    line_noise_snr: float | None = None,
) -> dict:
    """Executes the full SonicShield acoustic analysis pipeline.
    
    Args:
        file_source: Filepath, bytes, BytesIO, or preloaded 8 kHz float32 numpy array.
        codec: 'clean', 'g711' (ITU-T G.711 mu-law), or 'amr'.
        mode: 'both', 'physics', or 'biology' (ablation support).
        sr: Audio sample rate (default 8000).
        packet_loss_rate: Optional simulated network packet loss rate [0.0 - 0.1].
        line_noise_snr: Optional telephone line noise SNR in dB.
        
    Returns:
        Structured dictionary containing overall and per-window analysis metrics.
    """
    start_time = time.perf_counter()
    warnings: list[str] = []

    # 1. Audio Loading & Preprocessing
    if isinstance(file_source, np.ndarray):
        raw_audio = file_source.astype(np.float32)
        audio_warnings = []
    else:
        raw_audio, sr, audio_warnings = load_audio(file_source, target_sr=sr)
    warnings.extend(audio_warnings)

    duration_sec = round(len(raw_audio) / sr, 2)

    # 2. Codec Crusher Simulation
    try:
        crushed_audio = crush(
            raw_audio,
            sr=sr,
            codec=codec,
            packet_loss_rate=packet_loss_rate,
            line_noise_snr=line_noise_snr,
        )
    except CodecUnavailableError as e:
        raise PipelineError(f"Codec error: {e}") from e
    except Exception as e:
        raise PipelineError(f"Failed applying codec '{codec}': {e}") from e

    # 3. Rolling Window Generation (6s windows, 1s hop)
    windows, win_warnings = generate_windows(crushed_audio, sr=sr)
    warnings.extend(win_warnings)

    if not windows:
        raise PipelineError("No analysis windows could be generated from input audio.")

    # 4. Engine Analysis & Scorer Fusion per Window
    window_results: list[dict] = []
    history: list[dict] = []

    for win in windows:
        try:
            phys_res = analyze_physics(win.audio, sr=sr)
        except Exception as e:
            phys_res = {
                "score": 0.5,
                "reasons": [f"Physics engine error: {e}"],
                "features": {"usable": False, "error": str(e)},
            }

        try:
            bio_res = analyze_biology(win.audio, sr=sr)
        except Exception as e:
            bio_res = {
                "score": 0.5,
                "reasons": [f"Biology engine error: {e}"],
                "features": {"usable": False, "error": str(e)},
            }

        scored = score_window(phys_res, bio_res, history=history, mode=mode)
        history.append(scored)

        win_payload = {
            "window_index": win.index,
            "t_start": win.t_start,
            "t_end": win.t_end,
            "t": win.t,
            "is_padded": win.is_padded,
            "liveness": scored["liveness"],
            "raw_liveness": scored["raw_liveness"],
            "band": scored["band"],
            "band_label": scored["band_label"],
            "physics": scored["physics"],
            "biology": scored["biology"],
            "reasons": scored["reasons"],
            "agent_tip": scored["agent_tip"],
            "features": {
                "physics": phys_res.get("features", {}),
                "biology": bio_res.get("features", {}),
            },
        }
        if win.warning:
            win_payload["warning"] = win.warning

        window_results.append(win_payload)

    # 5. Aggregate Overall Session Score
    valid_scores = [w["liveness"] for w in window_results]
    valid_phys = [w["physics"] for w in window_results]
    valid_bio = [w["biology"] for w in window_results]

    overall_liveness = int(round(float(np.mean(valid_scores)))) if valid_scores else 50
    overall_band = get_band(overall_liveness)
    overall_band_label = get_band_label(overall_band)

    # Compile overall top reasons from window results
    all_reasons = []
    for w in window_results:
        for r in w.get("reasons", []):
            if r not in all_reasons:
                all_reasons.append(r)
    overall_reasons = all_reasons[:2] if all_reasons else ["Analysis complete; baseline acoustic properties observed."]

    # Recommended agent follow-up prompt
    agent_tip = None
    if overall_band in ("amber", "red"):
        for w in reversed(window_results):
            if w.get("agent_tip"):
                agent_tip = w["agent_tip"]
                break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    return {
        "overall_liveness": overall_liveness,
        "overall_band": overall_band,
        "overall_band_label": overall_band_label,
        "physics_mean": round(float(np.mean(valid_phys)), 3) if valid_phys else 0.5,
        "biology_mean": round(float(np.mean(valid_bio)), 3) if valid_bio else 0.5,
        "codec": codec,
        "mode": mode,
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "num_windows": len(window_results),
        "reasons": overall_reasons,
        "agent_tip": agent_tip,
        "warnings": warnings,
        "processing_time_ms": elapsed_ms,
        "windows": window_results,
        "crushed_audio": crushed_audio,
        "raw_audio": raw_audio,
    }
