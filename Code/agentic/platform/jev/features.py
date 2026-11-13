"""Build normalized JEV inputs from observable agent outputs."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agentic.platform.contracts import InspectionFeatures, ProcessStateSnapshot


def build_inspection_features(
    *,
    run_id: str,
    thread_id: str = "",
    quality: Optional[Mapping[str, Any]] = None,
    visual: Optional[Mapping[str, Any]] = None,
    knowledge: Optional[list[Mapping[str, Any]]] = None,
    cases: Optional[list[Mapping[str, Any]]] = None,
    process_snapshot: Optional[ProcessStateSnapshot] = None,
    evidence_conflict: float = 0.0,
    novelty_score: Optional[float] = None,
) -> InspectionFeatures:
    quality = dict(quality or {})
    visual = dict(visual or {})
    knowledge = list(knowledge or [])
    cases = list(cases or [])
    class_probs = visual.get("class_probabilities") or visual.get("class_probs") or {}
    confidence = float(visual.get("confidence", visual.get("score", max(class_probs.values()) if class_probs else 0.0)) or 0.0)
    roi_coverage = float(visual.get("roi_coverage", 1.0 if visual.get("has_target") else 0.0) or 0.0)
    cross_frame = visual.get("temporal_consistency", visual.get("cross_frame_consistency"))
    if cross_frame is None:
        cross_frame = quality.get("cross_frame_consistency")
    scores = [float(case.get("similarity", case.get("score", 0.0)) or 0.0) for case in cases]
    case_similarity = max(scores) if scores else 0.0
    coverage = min(1.0, len(knowledge) / 3.0)
    feature_map = {
        "quality_status": quality.get("quality_status", "unknown"),
        "target_present": bool(visual.get("has_target", quality.get("target_present", False))),
        "case_applicable": any(bool(case.get("applicable", False)) for case in cases),
        "case_count": len(cases),
        "knowledge_count": len(knowledge),
        "decoder_error": bool(quality.get("read_error")),
        "risk_level": str(visual.get("risk_level", "unknown")),
    }
    return InspectionFeatures(
        run_id=run_id,
        thread_id=thread_id,
        image_quality=float(quality.get("quality_score", quality.get("brightness_score", 0.0)) or 0.0),
        roi_coverage=roi_coverage,
        visual_confidence=confidence,
        cross_frame_consistency=float(cross_frame) if cross_frame is not None else None,
        novelty_score=float(novelty_score if novelty_score is not None else visual.get("novelty_score", 0.0) or 0.0),
        risk_score=float(visual.get("risk_score", visual.get("risk_probability", 0.0)) or 0.0),
        knowledge_coverage=coverage,
        case_similarity=case_similarity,
        evidence_conflict=float(evidence_conflict),
        process_snapshot=process_snapshot,
        evidence_ids=[str(item.get("source_id")) for item in knowledge if item.get("source_id")],
        features=feature_map,
    )
