"""Spectrum → Morgan/ECFP4 fingerprint model (1D CNN + self-attention).

All sequence axes are the configured static lengths (top-N peaks, mz bins) so
PyTorch/XLA compiles a single graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_ADDUCTS, Config, get_config


def _sinusoidal_mz_encoding(mz: torch.Tensor, dim: int, max_mz: float = 1000.0) -> torch.Tensor:
    """Fourier features of m/z. ``mz`` is ``(B, N)`` in Daltons."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal dim must be even")
    half = dim // 2
    device = mz.device
    freq = torch.exp(torch.linspace(0.0, math.log(max_mz), half, device=device))
    # (B, N, half)
    angles = mz.unsqueeze(-1) / freq.view(1, 1, half).clamp_min(1e-3)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.norm1(self.conv1(x))))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return x + h


class Spec2FP(nn.Module):
    """Encode a padded peak list + binned spectrum into 2048 fingerprint logits."""

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        cfg = cfg or get_config()
        self.cfg = cfg
        d = cfg.d_model
        self.peak_in = nn.Linear(6 + 32, d)  # raw feats + fourier(mz)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=d * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.peak_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_transformer_layers, enable_nested_tensor=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.bin_proj = nn.Conv1d(1, cfg.n_conv_channels, kernel_size=7, padding=3)
        self.bin_blocks = nn.Sequential(
            ConvBlock(cfg.n_conv_channels, 7, cfg.dropout),
            ConvBlock(cfg.n_conv_channels, 5, cfg.dropout),
            ConvBlock(cfg.n_conv_channels, 3, cfg.dropout),
        )
        self.bin_pool = nn.AdaptiveAvgPool1d(1)
        self.bin_out = nn.Linear(cfg.n_conv_channels, d)

        self.adduct_emb = nn.Embedding(NUM_ADDUCTS, cfg.adduct_embed_dim)
        self.meta = nn.Sequential(
            nn.Linear(cfg.precursor_feat_dim + cfg.adduct_embed_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )
        fused = d * 3
        self.head = nn.Sequential(
            nn.Linear(fused, d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, cfg.fp_bits),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_nl: torch.Tensor,
        peak_mask: torch.Tensor,
        binned: torch.Tensor,
        precursor_feat: torch.Tensor,
        adduct_id: torch.Tensor,
    ) -> torch.Tensor:
        # peak tokens: (B, N, d)
        mz_n = peak_mz / 1000.0
        nl_n = peak_nl / 1000.0
        fourier = _sinusoidal_mz_encoding(peak_mz, 32)
        raw = torch.stack(
            [
                mz_n,
                peak_intensity,
                nl_n,
                torch.log1p(peak_mz.clamp_min(0.0)),
                torch.log1p(peak_nl.clamp_min(0.0)),
                peak_intensity.sqrt(),
            ],
            dim=-1,
        )
        tokens = self.peak_in(torch.cat([raw, fourier], dim=-1))
        bsz = tokens.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Transformer key padding: True = ignore. CLS is never padding.
        pad = peak_mask <= 0
        cls_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=pad.device)
        key_padding = torch.cat([cls_pad, pad], dim=1)
        encoded = self.peak_encoder(tokens, src_key_padding_mask=key_padding)
        peak_vec = encoded[:, 0, :]

        bin_h = self.bin_proj(binned.unsqueeze(1))
        bin_h = self.bin_blocks(bin_h)
        bin_vec = self.bin_out(self.bin_pool(bin_h).squeeze(-1))

        meta_in = torch.cat([precursor_feat, self.adduct_emb(adduct_id.long())], dim=-1)
        meta_vec = self.meta(meta_in)
        fused = torch.cat([peak_vec, bin_vec, meta_vec], dim=-1)
        return self.head(fused)


class FocalBCELoss(nn.Module):
    """Multi-label focal BCE on logits (fingerprint bits)."""

    def __init__(self, gamma: float = 1.5, pos_weight: float = 6.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)
        loss = bce * (1.0 - pt).clamp_min(0.0).pow(self.gamma)
        return loss.mean()


def model_from_config(cfg: Config | None = None) -> Spec2FP:
    return Spec2FP(cfg or get_config())
