"""Check amplitude distribution of dataset files.
"""
import os
import numpy as np
import soundfile as sf

def main():
    base_dir = "data/external/in_the_wild/release_in_the_wild/release_in_the_wild"
    files = [f for f in os.listdir(base_dir) if f.endswith(".wav")][:50]
    
    max_amps = []
    rms_vals = []
    
    for f in files:
        data, sr = sf.read(os.path.join(base_dir, f), dtype="float32", always_2d=True)
        mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
        max_amps.append(float(np.max(np.abs(mono))))
        rms_vals.append(float(np.sqrt(np.mean(mono**2))))
        
    print(f"Sampled {len(files)} files:")
    print(f"Max Amplitude: Min={min(max_amps):.4f}, Mean={np.mean(max_amps):.4f}, Max={max(max_amps):.4f}")
    print(f"RMS Amplitude: Min={min(rms_vals):.4f}, Mean={np.mean(rms_vals):.4f}, Max={max(rms_vals):.4f}")

if __name__ == "__main__":
    main()
