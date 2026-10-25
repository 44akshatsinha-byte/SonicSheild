"""Confidence Scorer: Fusion, EMA smoothing, banding, and agent guidance for SonicShield."""
from __future__ import annotations

import numpy as np

# Configurable prototype score bands
BAND_GREEN_THRESHOLD = 70.0
BAND_AMBER_THRESHOLD = 40.0
DEFAULT_EMA_ALPHA = 0.45

# Follow-up conversational verification prompts for agents
AGENT_FOLLOW_UP_TIPS = [
    "Sorry, the connection crackled for a second. Could you repeat that and tell me the last thing you did today?",
    "Could you please clarify that in a bit more detail? The line was slightly unclear.",
    "Apologies, there was brief static on the call. What was your original account setup date?",
]

def get_band(score: float) -> str:
    """Maps a 0-100 score to risk band ('green', 'amber', 'red')."""
    if score >= BAND_GREEN_THRESHOLD:
        return "green"
    elif score >= BAND_AMBER_THRESHOLD:
        return "amber"
    else:
        return "red"

def get_band_label(band: str) -> str:
    """Human-readable status label."""
    if band == "green":
        return "Likely Live Human"
    elif band == "amber":
        return "Uncertain / Further Verification Recommended"
    else:
        return "Suspicious / Possible Synthetic Audio"

def score_window(
    physics_result: dict,
    biology_result: dict,
    history: list[dict] | None = None,
    mode: str = "both",
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict:
    """Combines independent Physics and Biology engine outputs into a calibrated liveness score.
    
    Args:
        physics_result: Output dict from physics_engine.analyze().
        biology_result: Output dict from biology_engine.analyze().
        history: Optional list of previous window scoring dicts for EMA smoothing.
        mode: 'both', 'physics', or 'biology' (ablation support).
        ema_alpha: EMA smoothing weight for current window [0.1 to 1.0].
        
    Returns:
        dict containing:
            - 'liveness': int (0-100)
            - 'raw_liveness': float (0-100 before smoothing)
            - 'band': 'green' | 'amber' | 'red'
            - 'band_label': str
            - 'physics': float (0.0 to 1.0)
            - 'biology': float (0.0 to 1.0)
            - 'reasons': list[str] (top 2 explanatory points)
            - 'agent_tip': str or None (suggested conversational prompt for amber/red)
            - 'mode': str
            - 'is_usable': bool
    """
    mode = mode.lower().strip()
    if mode not in ("both", "physics", "biology"):
        mode = "both"

    phys_usable = physics_result.get("features", {}).get("usable", True)
    bio_usable = biology_result.get("features", {}).get("usable", True)

    raw_phys = float(physics_result.get("score", 0.5))
    raw_bio = float(biology_result.get("score", 0.5))

    phys_reasons = physics_result.get("reasons", [])
    bio_reasons = biology_result.get("reasons", [])

    # Case: Unusable window (e.g. digital silence)
    if not phys_usable and not bio_usable:
        reasons = (phys_reasons + bio_reasons)[:2] or ["Insufficient audio energy to determine acoustic liveness."]
        return {
            "liveness": 20,
            "raw_liveness": 20.0,
            "band": "red",
            "band_label": get_band_label("red"),
            "physics": raw_phys,
            "biology": raw_bio,
            "reasons": reasons,
            "agent_tip": AGENT_FOLLOW_UP_TIPS[0],
            "mode": mode,
            "is_usable": False,
        }

    # Determine fused score based on ablation mode and availability
    if mode == "physics":
        fused = raw_phys
        reasons = phys_reasons[:2]
    elif mode == "biology":
        fused = raw_bio
        reasons = bio_reasons[:2]
    else:  # mode == 'both'
        if phys_usable and bio_usable:
            # Balanced fusion: 50% room physics, 50% vocal biology
            fused = 0.50 * raw_phys + 0.50 * raw_bio
            # Interleave top reason from physics and top reason from biology
            reasons = []
            if phys_reasons:
                reasons.append(phys_reasons[0])
            if bio_reasons:
                reasons.append(bio_reasons[0])
            if len(reasons) < 2 and len(phys_reasons) > 1:
                reasons.append(phys_reasons[1])
        elif phys_usable:
            fused = raw_phys
            reasons = phys_reasons[:2]
        else:
            fused = raw_bio
            reasons = bio_reasons[:2]

    # Map to 0-100 scale
    raw_score = float(np.clip(round(fused * 100.0, 1), 0.0, 100.0))

    # Apply Exponential Moving Average (EMA) across sequential valid windows
    smoothed_score = raw_score
    if history and len(history) > 0:
        prev_scores = [h["liveness"] for h in history if h.get("is_usable", True)]
        if prev_scores:
            prev_smoothed = float(prev_scores[-1])
            smoothed_score = ema_alpha * raw_score + (1.0 - ema_alpha) * prev_smoothed

    final_liveness = int(np.clip(round(smoothed_score), 0, 100))
    band = get_band(final_liveness)
    band_label = get_band_label(band)

    agent_tip = None
    if band in ("amber", "red"):
        # Select appropriate conversational prompt
        agent_tip = AGENT_FOLLOW_UP_TIPS[0]

    return {
        "liveness": final_liveness,
        "raw_liveness": raw_score,
        "band": band,
        "band_label": band_label,
        "physics": round(raw_phys, 3),
        "biology": round(raw_bio, 3),
        "reasons": reasons[:2],
        "agent_tip": agent_tip,
        "mode": mode,
        "is_usable": True,
    }
