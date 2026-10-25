"""Unit tests for ML model training script and dataset validation."""
import os
import tempfile
import csv
from scripts.train_ml_model import load_and_validate_manifest, create_speaker_split, train_and_evaluate

def test_manifest_validation_missing_file():
    records, issues = load_and_validate_manifest("nonexistent_manifest.csv")
    assert len(records) == 0
    assert any("does not exist" in iss for iss in issues)

def test_manifest_validation_with_missing_audio_clips():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "speaker", "source", "codec", "room", "attack_type"])
        writer.writeheader()
        writer.writerow({
            "file": "nonexistent_clip.wav",
            "label": "real",
            "speaker": "spk01",
            "source": "test",
            "codec": "clean",
            "room": "office",
            "attack_type": "none",
        })
        manifest_path = f.name

    try:
        records, issues = load_and_validate_manifest(manifest_path)
        assert len(records) == 0
        assert any("does not exist on disk" in iss for iss in issues)
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

def test_speaker_split_zero_leakage():
    # Construct 10 fake records across 4 real speakers and 4 fake speakers
    records = []
    for spk_idx in range(1, 5):
        spk_id = f"spk_real_{spk_idx}"
        for clip in range(3):
            records.append({
                "path": f"dummy_{spk_id}_{clip}.wav",
                "label": 1,
                "speaker": spk_id,
            })

    for spk_idx in range(1, 5):
        spk_id = f"spk_fake_{spk_idx}"
        for clip in range(3):
            records.append({
                "path": f"dummy_{spk_id}_{clip}.wav",
                "label": 0,
                "speaker": spk_id,
            })

    train_recs, val_recs, test_recs, meta = create_speaker_split(records, train_ratio=0.6, val_ratio=0.2)

    train_spks = set(meta["train_speakers"])
    val_spks = set(meta["val_speakers"])
    test_spks = set(meta["test_speakers"])

    # Strict disjoint speaker partition verification: ZERO LEAKAGE
    assert train_spks.isdisjoint(val_spks), "Train and Val must share 0 speakers"
    assert train_spks.isdisjoint(test_spks), "Train and Test must share 0 speakers"
    assert val_spks.isdisjoint(test_spks), "Val and Test must share 0 speakers"

    # All records in each split must have speakers belonging only to that split
    for r in train_recs:
        assert r["speaker"] in train_spks
    for r in val_recs:
        assert r["speaker"] in val_spks
    for r in test_recs:
        assert r["speaker"] in test_spks

def test_train_and_evaluate_insufficient_dataset():
    # Running on template with missing audio files should return False and exit cleanly
    success = train_and_evaluate("data/labels_template.csv")
    assert success is False
