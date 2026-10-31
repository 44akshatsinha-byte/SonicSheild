"""Test severe clipping effects on AASIST bona-fide score."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import soundfile as sf
import scipy.signal
from src.pretrained_aasist import get_aasist_classifier

def main():
    aasist = get_aasist_classifier()
    for fname in ['9.wav', '7.wav', '4.wav', '5.wav']:
        data, sr = sf.read('data/external/in_the_wild/release_in_the_wild/release_in_the_wild/' + fname, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = scipy.signal.resample(data, int(round(len(data)*16000/sr)))
        
        print(f"\n=== {fname} ===")
        for g in [1.0, 2.0, 4.0, 8.0, 16.0]:
            clipped = np.clip(data * g, -1.0, 1.0)
            pct_clip = float(np.mean(np.abs(clipped) >= 0.999) * 100)
            res = aasist.predict(clipped, sr=16000)
            print(f"Gain {g:4.1f}x | Clipped: {pct_clip:6.2f}% | Bona: {res['bona_fide_score']:.4f} | Class: {res['predicted_class']:<10} | Logits: {res['raw_score']}")

if __name__ == "__main__":
    main()
