"""Detailed controlled evaluation of 5 genuine and 5 synthetic voice recordings.

Compares:
1. Native 16 kHz Clean -> AASIST (Direct Wideband Branch B)
2. 8 kHz Bottleneck -> Resampled to 16 kHz -> AASIST (Old pipeline bug)
3. G.711 Codec (8 kHz PSTN telephony -> Resampled to 16 kHz) -> AASIST

Outputs:
- Bona-fide score (0-1)
- Spoof score (0-1)
- Predicted class
- Pre-softmax logits [z_spoof, z_bonafide]
"""
import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal

sys.path.insert(0, os.path.abspath("."))
from src.pretrained_aasist import get_aasist_classifier
from src.codec_crusher import crush

def evaluate_recording(aasist, path: str, speaker_name: str, true_label: str):
    if not os.path.exists(path):
        return None

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]

    # Condition 1: Native 16 kHz Clean (Direct Wideband Branch B)
    if sr != 16000:
        num_target = int(round(len(mono) * 16000 / sr))
        wav_16k = scipy.signal.resample(mono, num_target).astype(np.float32)
    else:
        wav_16k = mono.astype(np.float32)

    res_native = aasist.predict(wav_16k, sr=16000, codec="clean")

    # Condition 2: 8 kHz Bottleneck (Downsampled to 8k then upsampled to 16k)
    num_8k = int(round(len(wav_16k) * 8000 / 16000))
    wav_8k = scipy.signal.resample(wav_16k, num_8k).astype(np.float32)
    res_8k = aasist.predict(wav_8k, sr=8000, codec="clean_8k")

    # Condition 3: G.711 Telephony (Crushed at 8k then resampled to 16k)
    wav_g711_8k = crush(wav_8k, sr=8000, codec="g711")
    res_g711 = aasist.predict(wav_g711_8k, sr=8000, codec="g711")

    return {
        "file": os.path.basename(path),
        "speaker": speaker_name,
        "true_label": true_label,
        "native": {
            "bona": res_native["bona_fide_score"],
            "spoof": round(1.0 - res_native["bona_fide_score"], 4),
            "class": res_native["predicted_class"],
            "logits": res_native["raw_score"],
        },
        "bottleneck_8k": {
            "bona": res_8k["bona_fide_score"],
            "spoof": round(1.0 - res_8k["bona_fide_score"], 4),
            "class": res_8k["predicted_class"],
            "logits": res_8k["raw_score"],
        },
        "g711": {
            "bona": res_g711["bona_fide_score"],
            "spoof": round(1.0 - res_g711["bona_fide_score"], 4),
            "class": res_g711["predicted_class"],
            "logits": res_g711["raw_score"],
        },
    }

def main():
    aasist = get_aasist_classifier()
    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"

    dataset = [
        # 5 Genuine Recordings
        ("4.wav", "Christopher Hitchens", "genuine"),
        ("5.wav", "Ayn Rand", "genuine"),
        ("7.wav", "Donald Trump", "genuine"),
        ("9.wav", "Alec Guinness", "genuine"),
        ("12.wav", "Calvin Coolidge", "genuine"),
        # 5 Synthetic Recordings
        ("0.wav", "Alec Guinness (Clone)", "spoof"),
        ("1.wav", "Alec Guinness (Clone)", "spoof"),
        ("2.wav", "Barack Obama (Clone)", "spoof"),
        ("6.wav", "Barack Obama (Clone)", "spoof"),
        ("10.wav", "Bernie Sanders (Clone)", "spoof"),
    ]

    results = []
    for fname, spk, lbl in dataset:
        p = os.path.join(base_dir, fname)
        res = evaluate_recording(aasist, p, spk, lbl)
        if res:
            results.append(res)

    print("=" * 130)
    print("DETAILED CONTROLLED EVALUATION: 5 GENUINE VS 5 SYNTHETIC RECORDINGS")
    print("=" * 130)

    for item in results:
        print(f"\n[{item['file']}] {item['speaker']} (True Label: {item['true_label'].upper()})")
        print(f"  Condition A: Native 16 kHz Direct (Branch B):")
        print(f"    Class: {item['native']['class']:<10} | Bona Score: {item['native']['bona']:.4f} | Spoof Score: {item['native']['spoof']:.4f} | Logits: {item['native']['logits']}")
        print(f"  Condition B: 8 kHz Bottleneck (Old Bug):")
        print(f"    Class: {item['bottleneck_8k']['class']:<10} | Bona Score: {item['bottleneck_8k']['bona']:.4f} | Spoof Score: {item['bottleneck_8k']['spoof']:.4f} | Logits: {item['bottleneck_8k']['logits']}")
        print(f"  Condition C: G.711 Telephony (8 kHz PSTN -> 16 kHz):")
        print(f"    Class: {item['g711']['class']:<10} | Bona Score: {item['g711']['bona']:.4f} | Spoof Score: {item['g711']['spoof']:.4f} | Logits: {item['g711']['logits']}")

if __name__ == "__main__":
    main()
