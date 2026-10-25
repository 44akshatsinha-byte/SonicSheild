"""Codec Crusher: Telephone audio simulation engine for SonicShield."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf

class CodecUnavailableError(Exception):
    """Raised when the requested codec is not supported or not installed."""
    pass

# ITU-T G.711 mu-law constants
# Compands 14-bit linear PCM to 8-bit mu-law logarithmic PCM
BIAS = 0x84  # 132
CLIP = 32635

def _linear_to_mulaw(pcm: np.ndarray) -> np.ndarray:
    """Encodes linear 16-bit PCM integer samples to 8-bit mu-law per ITU-T G.711."""
    # Convert float [-1.0, 1.0] to signed 16-bit integer scale
    pcm16 = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int32)
    
    sign = np.where(pcm16 < 0, 0x80, 0x00).astype(np.uint8)
    mag = np.abs(pcm16)
    mag = np.clip(mag + BIAS, 0, CLIP)
    
    # Exponent lookup (finding MSB in 14-bit range)
    # Segments correspond to bit positions 7..14
    exponent = np.zeros_like(mag, dtype=np.uint8)
    exponent[mag >= 0x4000] = 7
    exponent[(mag >= 0x2000) & (mag < 0x4000)] = 6
    exponent[(mag >= 0x1000) & (mag < 0x2000)] = 5
    exponent[(mag >= 0x0800) & (mag < 0x1000)] = 4
    exponent[(mag >= 0x0400) & (mag < 0x0800)] = 3
    exponent[(mag >= 0x0200) & (mag < 0x0400)] = 2
    exponent[(mag >= 0x0100) & (mag < 0x0200)] = 1
    exponent[mag < 0x0100] = 0
    
    # Mantissa extraction (4 bits)
    mantissa = ((mag >> (exponent + 3)) & 0x0F).astype(np.uint8)
    
    # Combine sign (1 bit), exponent (3 bits), mantissa (4 bits)
    ulaw = sign | (exponent << 4) | mantissa
    # G.711 specifies bitwise inversion of all 8 bits
    return np.bitwise_not(ulaw).astype(np.uint8)

def _mulaw_to_linear(ulaw: np.ndarray) -> np.ndarray:
    """Decodes 8-bit mu-law samples to linear float32 PCM in [-1.0, 1.0]."""
    # Invert bits per G.711 specification
    ulaw_inv = np.bitwise_not(ulaw).astype(np.uint8)
    
    sign = (ulaw_inv & 0x80) != 0
    exponent = ((ulaw_inv >> 4) & 0x07).astype(np.int32)
    mantissa = (ulaw_inv & 0x0F).astype(np.int32)
    
    # Linear reconstruction
    linear = ((mantissa << 3) + BIAS) << exponent
    linear = linear - BIAS
    linear = np.where(sign, -linear, linear).astype(np.float32)
    
    # Scale back to float32 [-1.0, 1.0]
    return linear / 32768.0

def _g711_mulaw_roundtrip(audio: np.ndarray) -> np.ndarray:
    """Executes a true ITU-T G.711 mu-law 8-bit companding round trip."""
    encoded = _linear_to_mulaw(audio)
    decoded = _mulaw_to_linear(encoded)
    return decoded.astype(np.float32)

def check_amr_support() -> bool:
    """Checks whether FFmpeg is available and compiled with AMR-NB support."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-codecs"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return "libopencore_amrnb" in proc.stdout or "amr_nb" in proc.stdout
    except Exception:
        return False

def get_supported_codecs() -> dict[str, bool]:
    """Returns a dictionary indicating the availability of each codec."""
    return {
        "clean": True,
        "g711": True,  # Native ITU-T G.711 mu-law is always supported
        "amr": check_amr_support(),
    }

def _amr_roundtrip_ffmpeg(audio: np.ndarray, sr: int = 8000) -> np.ndarray:
    """Performs AMR-NB 12.2 kbps encode/decode via FFmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise CodecUnavailableError("FFmpeg is not installed or not in PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = os.path.join(tmpdir, "in.wav")
        amr_file = os.path.join(tmpdir, "audio.amr")
        out_wav = os.path.join(tmpdir, "out.wav")

        # Save float32 to temporary 16-bit PCM WAV at 8 kHz mono
        sf.write(in_wav, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")

        # Encode to AMR-NB at 12.2 kbps
        cmd_enc = [
            ffmpeg_bin, "-y", "-i", in_wav,
            "-ar", "8000", "-ac", "1", "-ab", "12.2k",
            amr_file
        ]
        res_enc = subprocess.run(cmd_enc, capture_output=True, text=True)
        if res_enc.returncode != 0 or not os.path.exists(amr_file):
            raise CodecUnavailableError(f"AMR-NB encoding failed: {res_enc.stderr}")

        # Decode AMR-NB back to 8 kHz WAV
        cmd_dec = [
            ffmpeg_bin, "-y", "-i", amr_file,
            "-ar", "8000", "-ac", "1",
            out_wav
        ]
        res_dec = subprocess.run(cmd_dec, capture_output=True, text=True)
        if res_dec.returncode != 0 or not os.path.exists(out_wav):
            raise CodecUnavailableError(f"AMR-NB decoding failed: {res_dec.stderr}")

        crushed, _ = sf.read(out_wav, dtype="float32")
        return crushed.astype(np.float32)

def simulate_line_artifacts(
    audio: np.ndarray,
    packet_loss_rate: float = 0.0,
    packet_size_ms: float = 20.0,
    sr: int = 8000,
    add_line_noise_snr: float | None = None,
) -> np.ndarray:
    """Simulates telephony network impairments (packet loss bursts and telephone line hiss)."""
    processed = audio.copy()
    
    # 1. Packet Loss Simulation (typical VoIP 20ms packets)
    if packet_loss_rate > 0.0:
        packet_samples = int(round(packet_size_ms * sr / 1000.0))
        num_packets = len(processed) // packet_samples
        if num_packets > 0:
            loss_mask = np.random.rand(num_packets) < packet_loss_rate
            for p_idx, is_lost in enumerate(loss_mask):
                if is_lost:
                    start = p_idx * packet_samples
                    end = min(start + packet_samples, len(processed))
                    processed[start:end] = 0.0

    # 2. Add Telephone Line White/Pink Hiss
    if add_line_noise_snr is not None:
        signal_power = np.mean(processed ** 2)
        if signal_power > 1e-7:
            noise_power = signal_power / (10.0 ** (add_line_noise_snr / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), size=len(processed))
            processed = processed + noise

    return np.clip(processed, -1.0, 1.0).astype(np.float32)

def crush(
    audio: np.ndarray,
    sr: int = 8000,
    codec: str = "clean",
    packet_loss_rate: float = 0.0,
    line_noise_snr: float | None = None,
) -> np.ndarray:
    """Applies telephone codec simulation and optional network impairments.
    
    Args:
        audio: 1D numpy array of 8 kHz audio.
        sr: Sample rate (must be 8000 Hz).
        codec: 'clean', 'g711' (ITU-T G.711 mu-law), or 'amr' (AMR-NB).
        packet_loss_rate: Fractional VoIP packet loss rate (e.g. 0.02 for 2%).
        line_noise_snr: Optional SNR in dB for telephone line noise floor.
        
    Returns:
        Processed 1D numpy array of float32 samples.
        
    Raises:
        CodecUnavailableError: When requested codec is not available.
        ValueError: For invalid codec names.
    """
    codec = codec.lower().strip()
    
    if codec == "clean":
        out = audio.copy()
    elif codec in ("g711", "g.711", "mulaw", "mu-law"):
        out = _g711_mulaw_roundtrip(audio)
    elif codec in ("amr", "amr-nb", "amr_nb"):
        if not check_amr_support():
            raise CodecUnavailableError(
                "AMR-NB codec requires FFmpeg with libopencore_amrnb support, which is not available in the current environment."
            )
        out = _amr_roundtrip_ffmpeg(audio, sr=sr)
    else:
        raise ValueError(f"Unknown codec '{codec}'. Supported codecs are: 'clean', 'g711', 'amr'.")

    if packet_loss_rate > 0.0 or line_noise_snr is not None:
        out = simulate_line_artifacts(
            out,
            packet_loss_rate=packet_loss_rate,
            sr=sr,
            add_line_noise_snr=line_noise_snr,
        )

    return out.astype(np.float32)
