"""Unit tests for scorer module."""
from src.scorer import score_window, get_band, get_band_label

def test_get_band_and_labels():
    assert get_band(85) == "green"
    assert get_band(70) == "green"
    assert get_band(69) == "amber"
    assert get_band(40) == "amber"
    assert get_band(39) == "red"
    assert get_band(0) == "red"

    assert "Live" in get_band_label("green")
    assert "Uncertain" in get_band_label("amber")
    assert "Suspicious" in get_band_label("red")

def test_scorer_both_engines():
    phys = {"score": 0.80, "reasons": ["Natural reverb"], "features": {"usable": True}}
    bio = {"score": 0.84, "reasons": ["Breath detected"], "features": {"usable": True}}

    res = score_window(phys, bio, mode="both")
    assert 80 <= res["liveness"] <= 85
    assert res["band"] == "green"
    assert res["physics"] == 0.80
    assert res["biology"] == 0.84
    assert len(res["reasons"]) == 2
    assert res["agent_tip"] is None

def test_scorer_red_agent_tip():
    phys = {"score": 0.15, "reasons": ["Digital silence"], "features": {"usable": True}}
    bio = {"score": 0.20, "reasons": ["No breath detected"], "features": {"usable": True}}

    res = score_window(phys, bio, mode="both")
    assert res["liveness"] < 40
    assert res["band"] == "red"
    assert res["agent_tip"] is not None
    assert "repeat" in res["agent_tip"].lower()

def test_scorer_ablation_physics_only():
    phys = {"score": 0.90, "reasons": ["Reverb present"], "features": {"usable": True}}
    bio = {"score": 0.20, "reasons": ["No breaths"], "features": {"usable": True}}

    res = score_window(phys, bio, mode="physics")
    assert res["liveness"] == 90
    assert res["band"] == "green"

def test_scorer_ema_smoothing():
    phys1 = {"score": 0.20, "reasons": ["r1"], "features": {"usable": True}}
    bio1 = {"score": 0.20, "reasons": ["r2"], "features": {"usable": True}}
    w1 = score_window(phys1, bio1, history=[])

    phys2 = {"score": 0.90, "reasons": ["r1"], "features": {"usable": True}}
    bio2 = {"score": 0.90, "reasons": ["r2"], "features": {"usable": True}}
    # EMA should smooth jump from 20 to 90
    w2 = score_window(phys2, bio2, history=[w1], ema_alpha=0.45)
    # Expected: 0.45 * 90 + 0.55 * 20 = 40.5 + 11.0 = 51.5 (approx 52)
    assert 48 <= w2["liveness"] <= 55
