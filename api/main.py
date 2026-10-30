"""SonicShield FastAPI backend server."""
from __future__ import annotations

import io
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.audio_utils import AudioValidationError
from src.codec_crusher import get_supported_codecs, CodecUnavailableError
from src.pipeline import run_pipeline, PipelineError
from src.physics_engine import analyze as analyze_physics
from src.biology_engine import analyze as analyze_biology
from src.scorer import score_window
from src.ml_features import extract_features
from src.ml_classifier import get_ml_classifier
from src.pretrained_aasist import get_aasist_classifier

app = FastAPI(
    title="SonicShield API",
    description="Zero-Friction Acoustic Liveness Detection Backend with Supervised ML and Pretrained AASIST",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Returns runtime health status, codec availability, ML model status, and AASIST status."""
    ml_clf = get_ml_classifier()
    aasist_clf = get_aasist_classifier()
    return {
        "status": "healthy",
        "service": "SonicShield Acoustic Liveness Detection",
        "version": "0.3.0",
        "timestamp": time.time(),
        "supported_codecs": get_supported_codecs(),
        "supported_detection_modes": ["baseline", "ml", "pretrained", "compare"],
        "ml_model": {
            "loaded": ml_clf.is_loaded,
            "status": ml_clf.load_status,
            "type": ml_clf.metadata.get("model_type"),
            "features_expected": len(ml_clf.metadata.get("feature_names", [])),
        },
        "aasist_model": {
            "loaded": aasist_clf.is_loaded,
            "status": aasist_clf.load_status,
            "architecture": "AASIST",
            "checkpoint_path": aasist_clf.checkpoint_path,
            "error": aasist_clf.load_error,
        },
        "window_duration_sec": 6.0,
        "hop_duration_sec": 1.0,
        "sample_rate_hz": 8000,
    }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="WAV audio file to inspect"),
    codec: str = Query("clean", description="Codec simulation: 'clean', 'g711', or 'amr'"),
    mode: str = Query("both", description="Engine ablation: 'both', 'physics', or 'biology'"),
    detection_mode: str = Query("baseline", description="Scoring mode: 'baseline', 'ml', 'pretrained' (AASIST), or 'compare'"),
    packet_loss_rate: float = Query(0.0, ge=0.0, le=0.5, description="Simulated packet loss rate"),
    line_noise_snr: float | None = Query(None, description="Simulated line noise SNR in dB"),
):
    """Analyzes an uploaded audio file and returns per-window and aggregate acoustic liveness results."""
    # Validate content type / filename
    filename = (file.filename or "").lower()
    if not (filename.endswith(".wav") or "audio" in (file.content_type or "")):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a standard WAV audio file.",
        )

    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # Limit file size to 25 MB for prototype safety
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds the 25MB limit.")

    try:
        result = run_pipeline(
            file_source=content,
            codec=codec,
            mode=mode,
            detection_mode=detection_mode,
            packet_loss_rate=packet_loss_rate,
            line_noise_snr=line_noise_snr,
        )
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (CodecUnavailableError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected processing error: {e}")

    # Remove raw numpy arrays before JSON serialization
    response_payload = {k: v for k, v in result.items() if k not in ("crushed_audio", "raw_audio")}
    return JSONResponse(content=response_payload)

@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint.
    
    Expects 1-second chunks of raw 16-bit PCM (8000 Hz mono = 16,000 bytes)
    or float32 audio arrays. Maintains a rolling 6-second window buffer.
    Pushes back 1 JSON result per second with both baseline and ML scores.
    """
    await websocket.accept()
    buffer: list[float] = []
    sr = 8000
    window_samples = 6 * sr  # 48,000 samples (6s)
    hop_samples = 1 * sr     # 8,000 samples (1s)
    current_time_sec = 0.0
    history: list[dict] = []
    ml_clf = get_ml_classifier()
    aasist_clf = get_aasist_classifier()

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Assume 16-bit signed PCM audio chunk
            pcm_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.extend(pcm_chunk.tolist())
            current_time_sec += len(pcm_chunk) / sr

            # When we have enough audio to slide a window
            while len(buffer) >= window_samples:
                window_slice = np.array(buffer[:window_samples], dtype=np.float32)
                
                # Analyze window with physics & biology engines
                phys_res = analyze_physics(window_slice, sr=sr)
                bio_res = analyze_biology(window_slice, sr=sr)
                scored = score_window(phys_res, bio_res, history=history, mode="both")
                history.append(scored)

                # ML feature extraction & prediction
                feat_vec, _, _ = extract_features(phys_res, bio_res, window_audio=window_slice, sr=sr)
                ml_res = ml_clf.predict_window(feat_vec)

                # Pretrained AASIST inference
                aasist_win = aasist_clf.predict(window_slice, sr=sr)

                update_msg = {
                    "t": round(current_time_sec, 1),
                    "liveness": scored["liveness"],
                    "baseline_liveness": scored["liveness"],
                    "ml_liveness": ml_res["ml_score"],
                    "ml_status": ml_res["status"],
                    "aasist_liveness": aasist_win["liveness_score"],
                    "aasist_bona_fide": aasist_win["bona_fide_score"],
                    "aasist_class": aasist_win["predicted_class"],
                    "band": scored["band"],
                    "band_label": scored["band_label"],
                    "physics": scored["physics"],
                    "biology": scored["biology"],
                    "reasons": scored["reasons"],
                    "agent_tip": scored["agent_tip"],
                }
                await websocket.send_json(update_msg)

                # Slide by hop (1 second)
                buffer = buffer[hop_samples:]

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Streaming error: {e}"})
            await websocket.close()
        except Exception:
            pass
