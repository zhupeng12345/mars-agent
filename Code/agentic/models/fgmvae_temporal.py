"""Minimal local F-GMVAE-T reimplementation for visual evidence windows.

This is an auditable baseline based on the project plan: spatial ROI encoder,
temporal encoder, feature-guided Gaussian mixture prior, fuzzy-label loss and
uncertainty. It intentionally exposes each intermediate tensor for ablations.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .losses import contrastive_consistency_loss, gaussian_mixture_kl, temporal_consistency_loss, fgmvae_loss
from .temporal import TemporalEncoder
from .visual_types import ROISequence
from .spatial_backbones import build_spatial_encoder


class FeatureLabelGenerator(nn.Module):
    """Map static/dynamic expert descriptors to a fuzzy mixture guide."""

    def __init__(self, feature_dim: int, mixture_components: int):
        super().__init__()
        if feature_dim < 1 or mixture_components < 1:
            raise ValueError("feature_dim and mixture_components must be positive")
        self.projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, mixture_components))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.projection(torch.nan_to_num(features.float())), dim=-1)


class GaussianMixturePrior(nn.Module):
    """Learnable diagonal Gaussian mixture prior used by the latent branch."""

    def __init__(self, mixture_components: int, latent_dim: int):
        super().__init__()
        self.means = nn.Parameter(torch.randn(mixture_components, latent_dim) * 0.05)
        self.logvars = nn.Parameter(torch.zeros(mixture_components, latent_dim))

    def parameters_for(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.means, self.logvars


class FuzzyGuidedVAE(nn.Module):
    """Shared named boundary for the fuzzy-label VAE family.

    The concrete temporal implementation below exposes the full forward and
    loss contract; this base keeps the paper components independently
    discoverable for ablation and downstream adapters.
    """

    pass


class FGMVAETemporal(FuzzyGuidedVAE):
    def __init__(
        self,
        num_classes: int = 3,
        latent_dim: int = 16,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        mixture_components: int = 3,
        temporal_type: str = "gru",
        expert_dim: int = 0,
        spatial_backbone: str = "tiny_cnn",
        model_profile: str = "fgmvae-t-mini-v1",
    ) -> None:
        super().__init__()
        if num_classes < 1 or latent_dim < 1 or mixture_components < 1:
            raise ValueError("num_classes, latent_dim and mixture_components must be positive")
        self.num_classes, self.latent_dim, self.mixture_components = num_classes, latent_dim, mixture_components
        self.model_profile = model_profile
        self.expert_dim = int(expert_dim)
        self.spatial_backbone = str(spatial_backbone)
        self.spatial_encoder, spatial_dim = build_spatial_encoder(self.spatial_backbone, spatial_dim)
        self.temporal_encoder = TemporalEncoder(spatial_dim, temporal_dim, temporal_type)
        self.posterior_mu = nn.Linear(temporal_dim, latent_dim)
        self.posterior_logvar = nn.Linear(temporal_dim, latent_dim)
        self.mixture_head = nn.Linear(temporal_dim, mixture_components)
        self.feature_label_generator = FeatureLabelGenerator(expert_dim, mixture_components) if expert_dim > 0 else None
        self.expert_head = nn.Linear(expert_dim, mixture_components) if expert_dim > 0 else None
        self.gm_prior = GaussianMixturePrior(mixture_components, latent_dim)
        self.prior_means = self.gm_prior.means
        self.prior_logvars = self.gm_prior.logvars
        self.classifier = nn.Linear(temporal_dim + latent_dim, num_classes)
        self.frame_attention = nn.Linear(spatial_dim, 1)

    def _frames_from_input(self, sequence: ROISequence | torch.Tensor, mask: Optional[torch.Tensor]):
        if isinstance(sequence, ROISequence):
            sequence.validate()
            frames = sequence.frames
            mask = sequence.effective_mask if mask is None else mask
        else:
            frames = sequence
            if frames.ndim == 4:
                frames = frames.unsqueeze(0)
            if frames.ndim != 5 or frames.shape[2] != 3:
                raise ValueError("frames must have shape [B,T,3,H,W]")
            if mask is None:
                mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
        if mask.shape != frames.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        return frames.float(), mask.bool()

    def forward(
        self,
        sequence: ROISequence | torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        expert_features: Optional[torch.Tensor] = None,
        feature_guide: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> dict:
        frames, valid_mask = self._frames_from_input(sequence, mask)
        b, t = frames.shape[:2]
        frame_features = self.spatial_encoder(frames.reshape(b * t, *frames.shape[2:])).reshape(b, t, -1)
        frame_features = torch.nan_to_num(frame_features)
        temporal = self.temporal_encoder(frame_features, valid_mask)
        mu, logvar = self.posterior_mu(temporal), self.posterior_logvar(temporal).clamp(-8.0, 8.0)
        do_sample = self.training if sample is None else bool(sample)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if do_sample else mu
        mixture_logits = self.mixture_head(temporal)
        if feature_guide is not None:
            if expert_features is not None:
                raise ValueError("pass only one of expert_features and feature_guide")
            if feature_guide.ndim == 1:
                feature_guide = feature_guide.unsqueeze(0).expand(b, -1)
            if feature_guide.shape != (b, self.mixture_components):
                raise ValueError(f"feature_guide must have shape [{b},{self.mixture_components}]")
            guide = torch.nan_to_num(feature_guide.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            guide_mass = guide.sum(dim=-1, keepdim=True)
            if (guide_mass <= 0).any():
                raise ValueError("feature_guide rows must have positive mass")
            feature_guide = guide / guide_mass
            mixture_logits = mixture_logits + feature_guide.clamp_min(1e-8).log()
        elif self.expert_head is None:
            feature_guide = None
        if self.expert_head is not None:
            # Missing expert/feature-guide input means an unconditioned GM prior;
            # this is the explicit ``prior missing`` path used by runtime smoke.
            if expert_features is not None:
                if expert_features.shape != (b, self.expert_dim):
                    raise ValueError(f"expert_features must have shape [{b},{self.expert_dim}]")
                clean_expert = torch.nan_to_num(expert_features.float())
                # The expert descriptors create a fuzzy mixture guide.  The
                # guide is injected as a log prior so it changes the latent
                # mixture itself, rather than only acting as an auxiliary head.
                feature_guide = self.feature_label_generator(clean_expert)
                mixture_logits = mixture_logits + self.expert_head(clean_expert) + feature_guide.clamp_min(1e-8).log()
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        latent_logits = self.classifier(torch.cat([temporal, z], dim=-1))
        frame_scores = self.frame_attention(frame_features).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~valid_mask, -1e4)
        # Avoid a degenerate all-missing softmax while retaining a zero attention sum.
        if (~valid_mask).all(dim=1).any():
            frame_scores = frame_scores.clone()
            frame_scores[~valid_mask.any(dim=1), 0] = 0.0
        attention = F.softmax(frame_scores, dim=-1) * valid_mask.float()
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        probs = F.softmax(latent_logits, dim=-1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1) / torch.log(torch.tensor(float(self.num_classes), device=probs.device))
        # A small variance term makes uncertainty sensitive to stochastic posterior width.
        uncertainty = (entropy + torch.sigmoid(logvar).mean(-1) * 0.25).clamp(0, 1)
        abnormal = probs[:, : min(2, self.num_classes)].sum(-1)
        risk_level = [
            "unknown" if not bool(valid_mask[i].any()) or float(uncertainty[i].detach()) >= 0.85
            else "high" if float(abnormal[i].detach()) >= 0.75
            else "medium" if float(abnormal[i].detach()) >= 0.45 else "low"
            for i in range(b)
        ]
        key = torch.topk(attention, k=min(3, t), dim=-1).indices
        kl = gaussian_mixture_kl(z, mu, logvar, self.prior_means, self.prior_logvars, mixture_weights)
        temp_loss = temporal_consistency_loss(frame_features, valid_mask)
        contrast_loss = contrastive_consistency_loss(frame_features, valid_mask)
        return {
            "logits": latent_logits,
            "class_probs": probs,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "mixture_logits": mixture_logits,
            "mixture_weights": mixture_weights,
            "feature_guide": feature_guide,
            "frame_features": frame_features,
            "temporal_embedding": temporal,
            "frame_attention": attention,
            "key_frame_ids": key,
            "uncertainty": torch.nan_to_num(uncertainty),
            "risk_score": abnormal,
            "risk_level": risk_level,
            "temporal_consistency": (1.0 - temp_loss.detach()).clamp(0, 1),
            "temporal_consistency_loss": temp_loss,
            "contrastive_loss": contrast_loss,
            "kl": kl,
            "valid_mask": valid_mask,
            "model_profile": self.model_profile,
        }

    def loss(self, outputs: dict, labels: Optional[torch.Tensor] = None, fuzzy_labels: Optional[torch.Tensor] = None, **kwargs):
        return fgmvae_loss(outputs, labels=labels, fuzzy_labels=fuzzy_labels, **kwargs)


__all__ = ["FeatureLabelGenerator", "GaussianMixturePrior", "FuzzyGuidedVAE", "FGMVAETemporal", "fgmvae_loss"]
