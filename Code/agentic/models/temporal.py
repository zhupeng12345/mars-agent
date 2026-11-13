"""Small, dependency-free temporal encoders for ROI feature sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Encode ``[B,T,D]`` into ``[B,out_dim]``.

    The four profiles are deliberately small enough for CPU smoke tests:
    ``conv`` (also accepted as ``cnn``), ``gru``, ``tcn`` and ``transformer``.
    ``mask`` marks usable frames and is respected by every profile.
    """

    def __init__(self, input_dim: int, out_dim: int = 64, encoder_type: str = "gru", num_heads: int = 4):
        super().__init__()
        self.input_dim, self.out_dim = int(input_dim), int(out_dim)
        self.encoder_type = "conv" if encoder_type == "cnn" else encoder_type.lower()
        if self.encoder_type == "conv":
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, out_dim, 3, padding=1), nn.GELU(),
                nn.Conv1d(out_dim, out_dim, 3, padding=1), nn.GELU(),
            )
        elif self.encoder_type == "gru":
            self.net = nn.GRU(input_dim, out_dim, batch_first=True)
        elif self.encoder_type == "tcn":
            self.blocks = nn.ModuleList([
                nn.Conv1d(input_dim if i == 0 else out_dim, out_dim, 3, padding=2**i, dilation=2**i)
                for i in range(3)
            ])
            # GroupNorm also works for the one-frame/one-sample CPU smoke case.
            self.norms = nn.ModuleList([nn.GroupNorm(1, out_dim) for _ in self.blocks])
        elif self.encoder_type == "transformer":
            heads = max(1, min(int(num_heads), out_dim))
            while out_dim % heads:
                heads -= 1
            self.input_proj = nn.Linear(input_dim, out_dim)
            layer = nn.TransformerEncoderLayer(out_dim, heads, dim_feedforward=out_dim * 2, batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=1)
        else:
            raise ValueError(f"unknown temporal encoder: {encoder_type}")
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape [B,T,{self.input_dim}]")
        b, t, _ = x.shape
        if t < 1:
            raise ValueError("temporal sequence must contain at least one frame")
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=x.device)
        if mask.shape != (b, t):
            raise ValueError("mask must have shape [B,T]")
        mask = mask.bool()
        x = torch.nan_to_num(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        if self.encoder_type == "conv":
            y = self.net(x.transpose(1, 2)).transpose(1, 2)
        elif self.encoder_type == "gru":
            y, _ = self.net(x)
        elif self.encoder_type == "tcn":
            y = x.transpose(1, 2)
            for conv, norm in zip(self.blocks, self.norms):
                residual = y if y.shape[1] == self.out_dim else None
                y = torch.nn.functional.gelu(norm(conv(y)))
                if residual is not None and residual.shape == y.shape:
                    y = y + residual
            y = y.transpose(1, 2)
        else:
            y = self.input_proj(x)
            positions = torch.arange(t, device=x.device, dtype=y.dtype).view(1, t, 1)
            div = torch.exp(torch.arange(0, self.out_dim, 2, device=x.device, dtype=y.dtype) * (-math.log(10000.0) / self.out_dim))
            pe = torch.zeros((1, t, self.out_dim), device=x.device, dtype=y.dtype)
            pe[..., 0::2] = torch.sin(positions * div)
            pe[..., 1::2] = torch.cos(positions * div[: pe[..., 1::2].shape[-1]])
            # Transformer cannot represent an all-padding row without NaNs.
            key_padding = ~mask
            key_padding = key_padding.clone()
            key_padding[~mask.any(dim=1), 0] = False
            y = self.net(y + pe, src_key_padding_mask=key_padding)
        valid = mask.unsqueeze(-1).to(y.dtype)
        pooled = (y * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.out_norm(pooled)
        pooled = pooled * mask.any(dim=1, keepdim=True).to(pooled.dtype)
        return pooled
