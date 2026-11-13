"""Transparent single-frame and masked temporal-pooling baselines."""

from __future__ import annotations

import torch
from torch import nn


def masked_pool(features: torch.Tensor, mask: torch.Tensor | None = None, mode: str = "mean") -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape [B,T,D]")
    b, t, _ = features.shape
    if mask is None:
        mask = torch.ones((b, t), dtype=torch.bool, device=features.device)
    if mask.shape != (b, t):
        raise ValueError("mask must have shape [B,T]")
    valid = mask.unsqueeze(-1)
    if mode == "mean":
        return (features * valid).sum(1) / valid.sum(1).clamp_min(1)
    if mode == "max":
        masked = features.masked_fill(~valid, -torch.inf)
        pooled = masked.max(1).values
        return torch.where(mask.any(1, keepdim=True), pooled, torch.zeros_like(pooled))
    if mode in {"single", "first"}:
        return features[:, 0]
    raise ValueError(f"unknown pooling mode: {mode}")


class TemporalPoolingClassifier(nn.Module):
    """A reproducible mean/max/single-frame classifier for ablations."""

    def __init__(self, input_dim: int, num_classes: int = 3, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "max", "single", "first"}:
            raise ValueError("mode must be mean, max or single")
        self.mode = mode
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(masked_pool(features, mask, self.mode))


__all__ = ["masked_pool", "TemporalPoolingClassifier"]
