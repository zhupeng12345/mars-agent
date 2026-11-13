"""Loss terms for the minimal F-GMVAE-T implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fuzzy_label_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy for hard labels or soft/fuzzy class memberships."""
    if target.ndim == 1:
        return F.cross_entropy(logits, target.long())
    if target.shape != logits.shape:
        raise ValueError("fuzzy target must have the same shape as class logits")
    target = target.float().clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def gaussian_mixture_kl(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_means: torch.Tensor,
    prior_logvars: torch.Tensor,
    mixture_weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monte-Carlo KL estimate ``log q(z)-log p(z)`` for diagonal GM prior."""
    if z.ndim != 2 or mu.shape != z.shape or logvar.shape != z.shape:
        raise ValueError("z, mu and logvar must have shape [B,Z]")
    if prior_means.ndim != 2 or prior_means.shape != prior_logvars.shape:
        raise ValueError("prior parameters must have shape [K,Z]")
    if mixture_weights.shape != (z.shape[0], prior_means.shape[0]):
        raise ValueError("mixture_weights must have shape [B,K]")
    q_log_prob = -0.5 * (((z - mu) ** 2) / logvar.exp().clamp_min(1e-6) + logvar + z.shape[-1] * 1.8378770664093453).sum(-1)
    z_expand = z[:, None, :]
    p_log_prob = -0.5 * (
        ((z_expand - prior_means[None]) ** 2) / prior_logvars.exp().clamp_min(1e-6)
        + prior_logvars[None] + z.shape[-1] * 1.8378770664093453
    ).sum(-1)
    log_mix = torch.log_softmax(mixture_weights.clamp_min(1e-8), dim=-1) if not torch.allclose(mixture_weights.sum(-1), torch.ones_like(mixture_weights[:, 0]), atol=1e-4) else torch.log(mixture_weights.clamp_min(1e-8))
    kl = q_log_prob - torch.logsumexp(log_mix + p_log_prob, dim=-1)
    kl = torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=0.0)
    if mask is not None:
        kl = kl * mask.float()
        return kl.sum() / mask.float().sum().clamp_min(1.0)
    return kl.mean()


def temporal_consistency_loss(frame_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Penalise adjacent feature jumps while respecting missing frames."""
    if frame_features.ndim != 3 or frame_features.shape[1] < 2:
        return frame_features.new_zeros(())
    diff = (frame_features[:, 1:] - frame_features[:, :-1]).pow(2).mean(-1)
    if mask is not None:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        if not pair_mask.any():
            return diff.new_zeros(())
        return diff[pair_mask].mean()
    return diff.mean()


def contrastive_consistency_loss(frame_features: torch.Tensor, mask: torch.Tensor | None = None, temperature: float = 0.2) -> torch.Tensor:
    """Adjacent-frame supervised contrastive proxy with missing-frame masks."""
    if frame_features.ndim != 3 or frame_features.shape[1] < 2:
        return frame_features.new_zeros(())
    pair_mask = torch.ones(frame_features.shape[:2], dtype=torch.bool, device=frame_features.device)
    if mask is not None:
        pair_mask = mask.bool()
    left, right = frame_features[:, :-1], frame_features[:, 1:]
    valid = pair_mask[:, :-1] & pair_mask[:, 1:]
    if not bool(valid.any()):
        return frame_features.new_zeros(())
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    cosine = (left * right).sum(-1)
    # Keep the temperature in the named loss so it can be reported and tuned.
    return ((1.0 - cosine) / max(float(temperature), 1e-6))[valid].mean()


def fgmvae_loss(
    outputs: dict,
    labels: torch.Tensor | None = None,
    fuzzy_labels: torch.Tensor | None = None,
    *,
    lambda_kl: float = 0.1,
    lambda_fuzzy: float = 0.5,
    lambda_temporal: float = 0.1,
    lambda_contrast: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return named loss terms and ``total`` for training or smoke tests."""
    logits = outputs["logits"]
    cls = fuzzy_label_loss(logits, labels) if labels is not None else logits.new_zeros(())
    fuzzy = fuzzy_label_loss(logits, fuzzy_labels) if fuzzy_labels is not None else logits.new_zeros(())
    kl = outputs.get("kl", logits.new_zeros(()))
    temporal = outputs.get("temporal_consistency_loss", logits.new_zeros(()))
    contrast = outputs.get("contrastive_loss")
    if contrast is None and "frame_features" in outputs:
        contrast = contrastive_consistency_loss(outputs["frame_features"], outputs.get("valid_mask"))
    if contrast is None:
        contrast = logits.new_zeros(())
    total = cls + lambda_kl * kl + lambda_fuzzy * fuzzy + lambda_temporal * temporal + lambda_contrast * contrast
    return {"total": total, "classification": cls, "kl": kl, "fuzzy": fuzzy, "temporal": temporal, "contrastive": contrast}


__all__ = ["contrastive_consistency_loss", "fuzzy_label_loss", "gaussian_mixture_kl", "temporal_consistency_loss", "fgmvae_loss"]
