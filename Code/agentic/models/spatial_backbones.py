"""Optional spatial encoders for the F-GMVAE-T ablation ladder."""

from __future__ import annotations

import torch
from torch import nn


class TinyViTEncoder(nn.Module):
    def __init__(self, output_dim: int = 64, patch: int = 16, heads: int = 4):
        super().__init__()
        self.patch = nn.Conv2d(3, output_dim, kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(value).flatten(2).transpose(1, 2)
        return self.norm(self.encoder(tokens)).mean(dim=1)


def build_spatial_encoder(name: str, output_dim: int = 64) -> tuple[nn.Module, int]:
    """Build ``tiny_cnn``, ``convnext_tiny`` or ``vit_tiny`` encoders.

    External pretrained weights are intentionally never downloaded here.  A
    caller must provide and record its own checkpoint if transfer learning is
    desired.
    """
    normalized = str(name or "tiny_cnn").lower()
    if normalized in {"convnext_tiny", "convnext-tiny"}:
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        input_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_dim, output_dim)
        return model, output_dim
    if normalized in {"vit_tiny", "vit-tiny", "lightweight_vit"}:
        return TinyViTEncoder(output_dim=output_dim), output_dim
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.GELU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        nn.Conv2d(32, output_dim, 3, stride=2, padding=1), nn.GELU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    ), output_dim


__all__ = ["TinyViTEncoder", "build_spatial_encoder"]
