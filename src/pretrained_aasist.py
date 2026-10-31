"""Pretrained AASIST Anti-Spoofing Model Integration for SonicShield.

This module provides local, standalone inference using the pretrained AASIST
(Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
model evaluated on ASVspoof benchmarks.

Original paper: "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"
(Jung et al., ICASSP 2022).
Official repository: https://github.com/clovaai/aasist
Pretrained checkpoint: AASIST.pth (SHA256: 51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0)
License: MIT
"""
from __future__ import annotations

import hashlib
import os
from typing import Union
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CHECKPOINT_PATH = os.path.join("models", "pretrained", "AASIST.pth")
EXPECTED_CHECKPOINT_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
AASIST_SAMPLE_RATE = 16000
AASIST_NB_SAMPLES = 64600  # Exactly 64,600 samples (~4.0375s at 16 kHz)

DEFAULT_AASIST_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


# ============================================================================
# Official AASIST Model Architecture (NAVER Corp. Clova AI / MIT License)
# ============================================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_exp.transpose(1, 2)
        return x_exp * x_mirror

    def _derive_att_map(self, x: Tensor) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x_flat = x.view(-1, org_size[-1])
        x_bn = self.bn(x_flat)
        return x_bn.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1: Tensor, x2: Tensor, master: Tensor | None = None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1_out = x.narrow(1, 0, num_type1)
        x2_out = x.narrow(1, num_type1, num_type2)
        return x1_out, x2_out, master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_exp.transpose(1, 2)
        return x_exp * x_mirror

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x: Tensor, num_type1: int, num_type2: int) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )
        att_board = att_board / self.temp
        att_board = F.softmax(att_board, dim=-2)
        return att_board

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x: Tensor, master: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x_flat = x.view(-1, org_size[-1])
        x_bn = self.bn(x_flat)
        return x_bn.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        z = self.drop(h)
        weights = self.proj(z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores: Tensor, h: Tensor, k: float) -> Tensor:
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        mask: bool = False,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports 1 input channel")
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        nfft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            h_high = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            h_low = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            h_ideal = h_high - h_low
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(h_ideal)

    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        band_pass_filter = self.band_pass.clone().to(x.device)
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        out = self.mp(out)
        return out


class AASISTModel(nn.Module):
    """AASIST Architecture matching the official ICASSP 2022 implementation."""

    def __init__(self, d_args: dict):
        super().__init__()
        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.conv_time = SincConv(
            out_channels=filts[0],
            kernel_size=d_args["first_conv"],
            in_channels=1,
            sample_rate=16000,
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: Tensor, Freq_aug: bool = False):
        """Forward pass. Input x shape: (batch_size, 64600)."""
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)

        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return last_hidden, output


# ============================================================================
# Pretrained AASIST Inference Engine & Wrapper
# ============================================================================

def pad_waveform(x: np.ndarray, max_len: int = AASIST_NB_SAMPLES) -> np.ndarray:
    """Deterministic padding / truncation to exactly 64,600 samples matching official AASIST eval."""
    x = np.asarray(x, dtype=np.float32).ravel()
    x_len = len(x)
    if x_len == 0:
        return np.zeros(max_len, dtype=np.float32)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(x, num_repeats)[:max_len]
    return padded.astype(np.float32)


class PretrainedAASIST:
    """Manages loading and inference for the official pretrained AASIST model."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model: AASISTModel | None = None
        self.is_loaded = False
        self.load_status = "unloaded"
        self.load_error: str | None = None

        self._load_checkpoint()

    def _verify_checksum(self) -> bool:
        """Verifies SHA256 checksum of the local checkpoint file."""
        if not os.path.exists(self.checkpoint_path):
            return False
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest() == EXPECTED_CHECKPOINT_SHA256

    def _load_checkpoint(self) -> None:
        """Loads and verifies the AASIST model weights."""
        if not os.path.exists(self.checkpoint_path):
            self.load_status = "checkpoint_not_found"
            self.load_error = f"AASIST checkpoint not found at {self.checkpoint_path}."
            return

        try:
            # Initialize model architecture
            model = AASISTModel(DEFAULT_AASIST_CONFIG)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.model = model
            self.is_loaded = True
            self.load_status = "active"
            self.load_error = None
        except Exception as e:
            self.load_status = "load_error"
            self.load_error = f"Failed initializing AASIST model: {e}"
            self.model = None

    def reload(self) -> None:
        """Reloads the model checkpoint from disk."""
        self._load_checkpoint()

    def prepare_audio_for_aasist(
        self,
        audio: np.ndarray,
        sr: int = 8000,
    ) -> tuple[np.ndarray, list[str]]:
        """Preprocesses audio into mono, 16,000 Hz, float32, and exactly 64,600 samples."""
        warnings: list[str] = []
        if len(audio) == 0:
            return np.zeros(AASIST_NB_SAMPLES, dtype=np.float32), ["Audio is empty (0 samples)."]

        # Ensure 1D float32
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Check silence
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < 1e-4:
            warnings.append("Audio is virtually silent (RMS < 1e-4); AASIST inference may be unreliable.")

        # Resample to 16,000 Hz if necessary
        if sr != AASIST_SAMPLE_RATE:
            if sr == 8000:
                # Efficient polyphase resample: 8 kHz -> 16 kHz (up=2, down=1)
                wav_16k = scipy.signal.resample_poly(wav, 2, 1).astype(np.float32)
            else:
                num_target = int(round(len(wav) * AASIST_SAMPLE_RATE / sr))
                wav_16k = scipy.signal.resample(wav, num_target).astype(np.float32)
        else:
            wav_16k = wav.astype(np.float32)

        # Truncate or tile-pad to exact 64,600 samples
        orig_16k_len = len(wav_16k)
        if orig_16k_len < AASIST_NB_SAMPLES:
            warnings.append(
                f"Audio segment ({orig_16k_len / AASIST_SAMPLE_RATE:.2f}s) shorter than "
                f"AASIST required window (4.04s); applied cyclic tile-padding."
            )
        padded = pad_waveform(wav_16k, max_len=AASIST_NB_SAMPLES)
        return padded, warnings

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 8000,
        codec: str = "clean",
    ) -> dict:
        """Runs AASIST anti-spoofing inference on an audio waveform.
        
        Args:
            audio: 1D float32 audio waveform.
            sr: Input sample rate in Hz (e.g. 8000 for phone audio).
            codec: Codec condition under which audio was generated.
            
        Returns:
            Structured dictionary matching the benchmark schema.
        """
        if not self.is_loaded or self.model is None:
            return {
                "model": "AASIST",
                "model_loaded": False,
                "predicted_class": "unavailable",
                "bona_fide_score": None,
                "liveness_score": None,
                "raw_score": None,
                "sample_rate": AASIST_SAMPLE_RATE,
                "codec": codec,
                "status": self.load_status,
                "warnings": [self.load_error or "AASIST model is not loaded."],
            }

        wav_64k, prep_warnings = self.prepare_audio_for_aasist(audio, sr=sr)

        try:
            with torch.inference_mode():
                tensor_input = torch.from_numpy(wav_64k).unsqueeze(0).to(self.device)  # (1, 64600)
                _, logits = self.model(tensor_input)  # (1, 2)
                raw_logits = logits[0].cpu().numpy().tolist()  # [spoof_logit, bona_fide_logit]

                # Softmax normalized scores (uncalibrated)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                prob_spoof = float(probs[0])
                prob_bonafide = float(probs[1])

                # Classification mapping: 1 = bona_fide, 0 = spoof
                predicted_class = "bona_fide" if prob_bonafide >= 0.5 else "spoof"
                liveness_score = int(np.clip(round(prob_bonafide * 100.0), 0, 100))

                return {
                    "model": "AASIST",
                    "model_loaded": True,
                    "predicted_class": predicted_class,
                    "bona_fide_score": round(prob_bonafide, 4),
                    "liveness_score": liveness_score,
                    "raw_score": [round(float(raw_logits[0]), 4), round(float(raw_logits[1]), 4)],
                    "sample_rate": AASIST_SAMPLE_RATE,
                    "codec": codec,
                    "status": "success",
                    "warnings": prep_warnings,
                }
        except Exception as e:
            return {
                "model": "AASIST",
                "model_loaded": True,
                "predicted_class": "error",
                "bona_fide_score": None,
                "liveness_score": None,
                "raw_score": None,
                "sample_rate": AASIST_SAMPLE_RATE,
                "codec": codec,
                "status": f"inference_error: {e}",
                "warnings": prep_warnings + [str(e)],
            }


# Module-level cached instance
_CACHED_AASIST: PretrainedAASIST | None = None

def get_aasist_classifier(reload: bool = False) -> PretrainedAASIST:
    """Returns the cached PretrainedAASIST instance."""
    global _CACHED_AASIST
    if _CACHED_AASIST is None or reload:
        _CACHED_AASIST = PretrainedAASIST()
    return _CACHED_AASIST
