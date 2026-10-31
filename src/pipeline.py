"""SonicShield Unified Acoustic Processing Pipeline.

Supports rule-based baseline scoring, genuine trained ML model scoring,
and side-by-side comparison mode across 6s rolling windows.
"""
from __future__ import annotations

import io
import math
import time
import numpy as np
import scipy.signal

from src.audio_utils import load_audio, load_audio_dual, analyze_clipping_telemetry, AudioValidationError
from src.windows import generate_windows, AudioWindow
from src.codec_crusher import crush, CodecUnavailableError
from src.physics_engine import analyze as analyze_physics
from src.biology_engine import analyze as analyze_biology
from src.scorer import score_window, get_band, get_band_label
from src.ml_features import extract_features
from src.ml_classifier import get_ml_classifier
from src.pretrained_aasist import get_aasist_classifier

class PipelineError(Exception):
    """Raised when an unrecoverable pipeline failure occurs."""
    pass

def run_pipeline(
    file_source: str | io.BytesIO | bytes | np.ndarray,
    codec: str = "clean",
    mode: str = "both",
    detection_mode: str = "baseline",
    sr: int = 8000,
    packet_loss_rate: float = 0.0,
    line_noise_snr: float | None = None,
) -> dict:
    """Executes the full SonicShield acoustic analysis pipeline.
    
    Args:
        file_source: Filepath, bytes, BytesIO, or preloaded 8 kHz float32 numpy array.
        codec: 'clean', 'g711' (ITU-T G.711 mu-law), or 'amr'.
        mode: Ablation mode for engines: 'both', 'physics', or 'biology'.
        detection_mode: 'baseline' (rule-based), 'ml' (custom ML), 'pretrained' (AASIST), or 'compare' (all).
        sr: Audio sample rate (default 8000).
        packet_loss_rate: Optional simulated network packet loss rate [0.0 - 0.1].
        line_noise_snr: Optional telephone line noise SNR in dB.
        
    Returns:
        Structured dictionary containing overall and per-window analysis metrics.
    """
    start_time = time.perf_counter()
    warnings: list[str] = []

    detection_mode = detection_mode.lower().strip()
    if detection_mode not in ("baseline", "ml", "pretrained", "compare"):
        detection_mode = "baseline"

    # Get ML classifier instance
    ml_clf = get_ml_classifier()
    if detection_mode in ("ml", "compare") and not ml_clf.is_loaded:
        warnings.append(
            f"Optional Custom ML Classifier (scikit-learn 18-feature model) is not loaded ({ml_clf.load_status}). "
            f"Pretrained AASIST Deep Neural Engine is ACTIVE and unaffected. "
            f"(To train the optional secondary ML classifier, run 'python scripts/train_ml_model.py')."
        )

    # Get Pretrained AASIST model instance
    aasist_clf = get_aasist_classifier()
    if detection_mode in ("pretrained", "compare") and not aasist_clf.is_loaded:
        warnings.append(
            f"Pretrained AASIST model is unavailable ({aasist_clf.load_status}). "
            f"Verify models/pretrained/AASIST.pth exists. "
            f"Baseline rule-based scoring will be used for primary decisions."
        )

    # 1. Dual-Branch Audio Ingress & Preprocessing
    # Branch A: 8,000 Hz float32 mono for Physics and Biology diagnostic engines (Telephony domain)
    # Branch B: 16,000 Hz float32 mono for AASIST Neural Anti-Spoofing engine (Wideband domain)
    if isinstance(file_source, np.ndarray):
        raw_arr = file_source.astype(np.float32)
        if sr == 8000:
            audio_8k = raw_arr
            audio_16k = scipy.signal.resample_poly(raw_arr, 2, 1).astype(np.float32)
        elif sr == 16000:
            audio_16k = raw_arr
            num_8k = int(math.ceil(len(raw_arr) * 8000 / 16000))
            audio_8k = scipy.signal.resample(raw_arr, num_8k).astype(np.float32)
        else:
            num_8k = int(math.ceil(len(raw_arr) * 8000 / sr))
            audio_8k = scipy.signal.resample(raw_arr, num_8k).astype(np.float32)
            num_16k = int(math.ceil(len(raw_arr) * 16000 / sr))
            audio_16k = scipy.signal.resample(raw_arr, num_16k).astype(np.float32)
        raw_audio = audio_8k
        audio_warnings = []
    else:
        audio_8k, audio_16k, orig_sr, audio_warnings = load_audio_dual(file_source)
        raw_audio = audio_8k
    warnings.extend(audio_warnings)

    duration_sec = round(len(audio_8k) / 8000, 2)

    # 2. Branch A Codec Crusher Simulation (8 kHz Telephony Domain)
    try:
        crushed_audio = crush(
            audio_8k,
            sr=8000,
            codec=codec,
            packet_loss_rate=packet_loss_rate,
            line_noise_snr=line_noise_snr,
        )
    except CodecUnavailableError as e:
        raise PipelineError(f"Codec error: {e}") from e
    except Exception as e:
        raise PipelineError(f"Failed applying codec '{codec}': {e}") from e

    # 3. Branch B Preparation for AASIST Neural Engine (16 kHz Domain)
    # For CLEAN codec: AASIST directly evaluates native 16 kHz wideband speech.
    # For Telephony codecs (G.711 / AMR): AASIST evaluates authentic telephone audio
    # resampled from the 8 kHz crushed channel to 16 kHz.
    if codec == "clean":
        aasist_input_audio = audio_16k
    else:
        # Authentic telephone channel simulation: 8 kHz crushed audio upsampled to 16 kHz
        aasist_input_audio = scipy.signal.resample_poly(crushed_audio, 2, 1).astype(np.float32)

    # 4. Rolling Window Generation (6s windows, 1s hop at 8 kHz)
    windows, win_warnings = generate_windows(crushed_audio, sr=8000)
    warnings.extend(win_warnings)

    if not windows:
        raise PipelineError("No analysis windows could be generated from input audio.")

    # 5. Engine Analysis, Scorer Fusion, & ML Inference per Window
    window_results: list[dict] = []
    history: list[dict] = []

    for win in windows:
        try:
            phys_res = analyze_physics(win.audio, sr=8000)
        except Exception as e:
            phys_res = {
                "score": 0.5,
                "reasons": [f"Physics engine error: {e}"],
                "features": {"usable": False, "error": str(e)},
            }

        try:
            bio_res = analyze_biology(win.audio, sr=8000)
        except Exception as e:
            bio_res = {
                "score": 0.5,
                "reasons": [f"Biology engine error: {e}"],
                "features": {"usable": False, "error": str(e)},
            }

        # Rule-based baseline scoring
        scored = score_window(phys_res, bio_res, history=history, mode=mode)
        history.append(scored)

        # Numerical ML Feature Vector Extraction
        feat_vec, feat_dict, quality = extract_features(
            phys_res,
            bio_res,
            window_audio=win.audio,
            sr=8000,
        )

        # ML Model Inference
        ml_res = ml_clf.predict_window(feat_vec)

        # Pretrained AASIST Anti-Spoofing Inference (Branch B: 16 kHz)
        try:
            if codec == "clean":
                start_samp_16k = int(round(win.t_start * 16000))
                end_samp_16k = int(round(win.t_end * 16000))
                win_audio_aasist = audio_16k[start_samp_16k:end_samp_16k]
            else:
                win_audio_aasist = scipy.signal.resample_poly(win.audio, 2, 1).astype(np.float32)

            aasist_win = aasist_clf.predict(win_audio_aasist, sr=16000, codec=codec)
        except Exception as e:
            aasist_win = {
                "model": "AASIST",
                "model_loaded": False,
                "predicted_class": "error",
                "bona_fide_score": None,
                "liveness_score": None,
                "raw_score": None,
                "sample_rate": 16000,
                "codec": codec,
                "status": f"inference_error: {e}",
                "warnings": [f"AASIST window analysis failed: {e}"],
            }

        # Determine active score based on detection_mode
        baseline_score = scored["liveness"]
        baseline_band = scored["band"]
        baseline_band_label = scored["band_label"]

        if detection_mode == "pretrained" and aasist_win["model_loaded"] and aasist_win["liveness_score"] is not None:
            active_score = aasist_win["liveness_score"]
            active_band = get_band(active_score)
            active_band_label = get_band_label(active_band)
        elif detection_mode == "ml" and ml_res["is_available"]:
            active_score = ml_res["ml_score"]
            active_band = ml_res["band"]
            active_band_label = ml_res["band_label"]
        else:
            active_score = baseline_score
            active_band = baseline_band
            active_band_label = baseline_band_label

        win_payload = {
            "window_index": win.index,
            "t_start": win.t_start,
            "t_end": win.t_end,
            "t": win.t,
            "is_padded": win.is_padded,
            "liveness": active_score,
            "band": active_band,
            "band_label": active_band_label,
            "baseline_score": baseline_score,
            "baseline_band": baseline_band,
            "baseline_band_label": baseline_band_label,
            "ml_score": ml_res["ml_score"],
            "ml_band": ml_res["band"],
            "ml_band_label": ml_res["band_label"],
            "ml_prob": ml_res["raw_prob"],
            "ml_status": ml_res["status"],
            "aasist_score": aasist_win["liveness_score"],
            "aasist_bona_fide_score": aasist_win["bona_fide_score"],
            "aasist_class": aasist_win["predicted_class"],
            "aasist_raw": aasist_win["raw_score"],
            "aasist": aasist_win,
            "physics": scored["physics"],
            "biology": scored["biology"],
            "reasons": scored["reasons"],
            "agent_tip": scored["agent_tip"],
            "features": {
                "physics": phys_res.get("features", {}),
                "biology": bio_res.get("features", {}),
                "ml_vector": feat_dict,
                "quality": quality,
            },
        }
        if win.warning:
            win_payload["warning"] = win.warning

        window_results.append(win_payload)

    # 5. Aggregate Overall Session Scores
    valid_baseline_scores = [w["baseline_score"] for w in window_results]
    valid_ml_scores = [w["ml_score"] for w in window_results if w["ml_score"] is not None]
    valid_aasist_scores = [w["aasist_score"] for w in window_results if w["aasist_score"] is not None]
    valid_phys = [w["physics"] for w in window_results]
    valid_bio = [w["biology"] for w in window_results]

    overall_baseline_liveness = int(round(float(np.mean(valid_baseline_scores)))) if valid_baseline_scores else 50
    overall_ml_liveness = (
        int(round(float(np.mean(valid_ml_scores)))) if valid_ml_scores and ml_clf.is_loaded else None
    )

    # Recording-level AASIST inference on Branch B audio (16 kHz Domain)
    try:
        aasist_overall = aasist_clf.predict(aasist_input_audio, sr=16000, codec=codec)
        if codec != "clean" and aasist_overall.get("model_loaded"):
            telephony_warning = (
                "AASIST evaluated 8 kHz telephone-band audio upsampled to 16 kHz. "
                "Because standard telephone codecs filter out frequencies above 3.4 kHz, "
                "AASIST high-frequency filterbanks lack signal energy, which naturally "
                "depresses bona-fide scores due to channel bandlimiting (out-of-domain telephone speech)."
            )
            if telephony_warning not in aasist_overall.get("warnings", []):
                aasist_overall.setdefault("warnings", []).append(telephony_warning)
                warnings.append(telephony_warning)
    except Exception as e:
        aasist_overall = {
            "model": "AASIST",
            "model_loaded": False,
            "predicted_class": "error",
            "bona_fide_score": None,
            "liveness_score": None,
            "raw_score": None,
            "sample_rate": 16000,
            "codec": codec,
            "status": f"inference_error: {e}",
            "warnings": [f"AASIST overall analysis failed: {e}"],
        }

    overall_aasist_liveness = (
        aasist_overall.get("liveness_score") if aasist_overall.get("model_loaded") else None
    )
    if overall_aasist_liveness is None and valid_aasist_scores:
        overall_aasist_liveness = int(round(float(np.mean(valid_aasist_scores))))

    if detection_mode == "pretrained":
        if overall_aasist_liveness is not None:
            overall_liveness = overall_aasist_liveness
        else:
            overall_liveness = overall_baseline_liveness
            warnings.append(
                f"AASIST inference unavailable or failed ({aasist_overall.get('status', 'error')}); "
                f"fell back to rule-based baseline scoring."
            )
    elif detection_mode == "ml" and overall_ml_liveness is not None:
        overall_liveness = overall_ml_liveness
    else:
        overall_liveness = overall_baseline_liveness

    overall_band = get_band(overall_liveness)
    overall_band_label = get_band_label(overall_band)

    # Compile overall top reasons from window results
    all_reasons = []
    if detection_mode == "pretrained" and aasist_overall.get("predicted_class") in ("bona_fide", "spoof"):
        pred_label = "Bona fide-like audio" if aasist_overall.get("predicted_class") == "bona_fide" else "Spoof-like synthetic audio"
        all_reasons.append(f"AASIST: {pred_label} (score: {aasist_overall.get('bona_fide_score')})")

    for w in window_results:
        for r in w.get("reasons", []):
            if r not in all_reasons:
                all_reasons.append(r)
    overall_reasons = all_reasons[:2] if all_reasons else ["Analysis complete; acoustic properties observed."]

    # Recommended agent follow-up prompt
    agent_tip = None
    if overall_band in ("amber", "red"):
        for w in reversed(window_results):
            if w.get("agent_tip"):
                agent_tip = w["agent_tip"]
                break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
    clip_telemetry = analyze_clipping_telemetry(audio_16k)

    return {
        "overall_liveness": overall_liveness,
        "overall_band": overall_band,
        "overall_band_label": overall_band_label,
        "overall_baseline_liveness": overall_baseline_liveness,
        "overall_ml_liveness": overall_ml_liveness,
        "overall_aasist_liveness": overall_aasist_liveness,
        "overall_aasist_bona_fide_score": aasist_overall.get("bona_fide_score"),
        "overall_aasist_class": aasist_overall.get("predicted_class"),
        "aasist": aasist_overall,
        "aasist_model_loaded": aasist_clf.is_loaded,
        "aasist_model_status": aasist_clf.load_status,
        "detection_mode": detection_mode,
        "model_loaded": ml_clf.is_loaded,
        "model_status": ml_clf.load_status,
        "model_type": ml_clf.metadata.get("model_type"),
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
        "audio_16k": audio_16k,
        "aasist_input_audio": aasist_input_audio,
        "telephony_out_of_domain": codec != "clean",
        "clipping": clip_telemetry,
    }
