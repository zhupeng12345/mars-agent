"""Adapter from tensor model output to the report-facing VisualEvidence contract."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from agentic.models.fgmvae_temporal import FGMVAETemporal
from agentic.models.visual_types import ROISequence
from agentic.platform.contracts import EvidenceKind, EvidenceRef, QualityStatus, RiskLevel, VisualEvidence

EXPERT_FEATURE_NAMES = (
    "brightness_mean", "brightness_std", "gray_contrast", "saturation_proxy", "hot_pixel_ratio",
    "gray_change_rate", "gray_change_std", "structure_change_rate", "area_change_rate",
)


def derive_expert_features(sequence: ROISequence) -> tuple[dict[str, float], dict[str, float]]:
    """Derive auditable static/dynamic ROI descriptors for F-GMVAE-T.

    These descriptors are weak visual features, not physical temperature or
    chemistry measurements.  Missing frames are excluded through the ROI mask.
    """
    frames = sequence.frames[0].float().clamp(0, 1)
    valid = sequence.effective_mask[0].bool()
    if not bool(valid.any()):
        return {}, {}
    selected = frames[valid]
    gray = selected.mean(dim=1)
    static = {
        "brightness_mean": float(gray.mean().item()),
        "brightness_std": float(gray.std(unbiased=False).item()),
        "gray_contrast": float(gray.std(unbiased=False).item()),
        "saturation_proxy": float((selected.max(dim=1).values - selected.min(dim=1).values).mean().item()),
        "hot_pixel_ratio": float((gray > 0.9).float().mean().item()),
    }
    dynamic: dict[str, float] = {}
    valid_indices = torch.where(valid)[0]
    if len(valid_indices) > 1:
        ordered = frames[valid_indices].mean(dim=1)
        delta = (ordered[1:] - ordered[:-1]).abs()
        dynamic = {
            "gray_change_rate": float(delta.mean().item()),
            "gray_change_std": float(delta.std(unbiased=False).item()),
            "structure_change_rate": float((delta > 0.08).float().mean().item()),
            "area_change_rate": float((ordered.mean(dim=(1, 2))[1:] - ordered.mean(dim=(1, 2))[:-1]).abs().mean().item()),
        }
    else:
        dynamic = {"gray_change_rate": 0.0, "gray_change_std": 0.0, "structure_change_rate": 0.0, "area_change_rate": 0.0}
    return static, dynamic


def _expert_feature_tensor(model: FGMVAETemporal, static: Mapping[str, float], dynamic: Mapping[str, float], device: torch.device) -> Optional[torch.Tensor]:
    if model.expert_dim == 0:
        return None
    if model.expert_dim != len(EXPERT_FEATURE_NAMES):
        raise ValueError(
            f"F-GMVAE-T runtime expert_dim must be {len(EXPERT_FEATURE_NAMES)} "
            f"for the registered static/dynamic descriptors; got {model.expert_dim}"
        )
    values = [float(static.get(name, 0.0)) for name in EXPERT_FEATURE_NAMES[:5]]
    values.extend(float(dynamic.get(name, 0.0)) for name in EXPERT_FEATURE_NAMES[5:])
    return torch.tensor([values], dtype=torch.float32, device=device)


def _risk_level(class_probs: Mapping[str, float], uncertainty: float, has_target: bool) -> str:
    if not has_target or uncertainty >= 0.85:
        return RiskLevel.UNKNOWN.value
    hot = float(class_probs.get("H", 0.0))
    cold = float(class_probs.get("C", 0.0))
    if max(hot, cold) >= 0.75:
        return RiskLevel.HIGH.value
    if max(hot, cold) >= 0.45:
        return RiskLevel.MEDIUM.value
    return RiskLevel.LOW.value


@torch.no_grad()
def run_visual_evidence(
    model: FGMVAETemporal,
    sequence: ROISequence,
    *,
    source_id: Optional[str] = None,
    frame_ids: Optional[Sequence[int]] = None,
    run_id: str = "",
    thread_id: str = "",
    static_features: Optional[Mapping[str, float]] = None,
    dynamic_features: Optional[Mapping[str, float]] = None,
    framewise_pooling: bool = False,
) -> VisualEvidence:
    """Run temporal inference or pool a single-frame supervised model over frames."""
    sequence.validate()
    if static_features is None or dynamic_features is None:
        derived_static, derived_dynamic = derive_expert_features(sequence)
        static_features = dict(static_features or derived_static)
        dynamic_features = dict(dynamic_features or derived_dynamic)
    expert_tensor = _expert_feature_tensor(model, static_features or {}, dynamic_features or {}, sequence.frames.device)
    model.eval()
    if framewise_pooling:
        valid_indices = torch.where(sequence.effective_mask[0])[0]
        if len(valid_indices):
            frames = sequence.frames[0, valid_indices].unsqueeze(1)
            frame_output = model(frames, sample=False)
            frame_probs = frame_output["class_probs"].detach().cpu()
            mean_probs = frame_probs.mean(dim=0)
            probs = mean_probs.tolist()
            frame_predictions = frame_probs.argmax(dim=-1)
            pooled_prediction = int(mean_probs.argmax().item())
            stability = float((frame_predictions == pooled_prediction).float().mean().item())
            confidences = frame_probs.max(dim=-1).values
            key_frame_ids = [int(valid_indices[i].item()) for i in torch.topk(confidences, k=min(3, len(valid_indices))).indices.tolist()]
            uncertainty = float((-(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum() / torch.log(torch.tensor(float(frame_probs.shape[-1])))).item())
            valid = True
            model_profile = str(frame_output["model_profile"])
            risk_level = _risk_level({name: float(value) for name, value in zip(("C", "H", "N"), probs)}, uncertainty, valid)
            temporal_consistency = stability
        else:
            probs = [1.0 / model.num_classes] * model.num_classes
            uncertainty, valid, model_profile = 1.0, False, model.model_profile
            key_frame_ids, risk_level, temporal_consistency = [], RiskLevel.UNKNOWN.value, 0.0
    else:
        output = model(sequence, expert_features=expert_tensor, sample=False)
        probs = output["class_probs"][0].detach().cpu().tolist()
        uncertainty = float(output["uncertainty"][0].item())
        valid = bool(output["valid_mask"][0].any().item())
        ids = list(frame_ids) if frame_ids is not None else list(range(sequence.frames.shape[1]))
        key_local = output["key_frame_ids"][0].detach().cpu().tolist()
        key_frame_ids = [int(ids[i]) for i in key_local if 0 <= int(i) < len(ids)]
        temporal_consistency = float(output["temporal_consistency"].item() if output["temporal_consistency"].ndim == 0 else output["temporal_consistency"][0].item())
        model_profile = str(output["model_profile"])
        risk_level = _risk_level({name: float(value) for name, value in zip(("C", "H", "N"), probs)}, uncertainty, valid)
    names = ["C", "H", "N"][: len(probs)]
    class_probs = {name: float(value) for name, value in zip(names, probs)}
    label = max(class_probs, key=class_probs.get) if class_probs else ""
    risk = risk_level
    ids = list(frame_ids) if frame_ids is not None else list(range(sequence.frames.shape[1]))
    key_ids = [int(ids[i]) for i in key_frame_ids if 0 <= int(i) < len(ids)]
    refs = [EvidenceRef(kind=EvidenceKind.VISUAL_ROI.value, source_id=source_id or sequence.source_ids[0] or "visual", run_id=run_id, thread_id=thread_id, locator="roi_sequence", confidence=1.0 - uncertainty)]
    evidence = VisualEvidence(
        source_id=source_id or (sequence.source_ids[0] if sequence.source_ids else "visual"),
        frame_ids=ids, label=label, score=float(max(class_probs.values(), default=0.0)), uncertainty=uncertainty,
        quality_status=QualityStatus.PASS.value if valid and bool(sequence.effective_mask[0].all()) else QualityStatus.HUMAN.value,
        class_probs=class_probs, risk_level=risk,
        temporal_consistency=temporal_consistency, key_frame_ids=key_ids,
        static_features=dict(static_features or {}), dynamic_features=dict(dynamic_features or {}),
        evidence_refs=refs, run_id=run_id, thread_id=thread_id, model_profile=model_profile,
        metadata={
            "missing_mask": (~sequence.effective_mask[0]).detach().cpu().tolist(),
            "input_mode": "framewise_probability_pooling" if framewise_pooling else "video_window",
            "expert_feature_names": list(EXPERT_FEATURE_NAMES) if expert_tensor is not None else [],
            "expert_guidance": "disabled_for_framewise_baseline" if framewise_pooling else ("static_dynamic_fuzzy_prior" if expert_tensor is not None else "unconditioned_prior"),
        },
    )
    evidence.validate()
    return evidence


__all__ = ["EXPERT_FEATURE_NAMES", "derive_expert_features", "run_visual_evidence"]
