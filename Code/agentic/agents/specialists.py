"""Deterministic domain specialists used by the full-loop fan-out."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.knowledge import VersionedKnowledgeBase


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def visual_review(context: Mapping[str, Any]) -> dict[str, Any]:
    visual = context.get("visual") if isinstance(context.get("visual"), Mapping) else {}
    quality = _mapping(context.get("quality"))
    probabilities = visual.get("class_probabilities") or visual.get("class_probs") or {}
    probabilities = {
        str(key): float(value)
        for key, value in probabilities.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    } if isinstance(probabilities, Mapping) else {}
    confidence = _finite_score(visual.get("confidence", visual.get("score", max(probabilities.values(), default=0.0)))) or 0.0
    confidence = min(1.0, max(0.0, confidence))
    uncertainty = visual.get("uncertainty")
    source_id = str(visual.get("source_id") or context.get("media_source_id") or "")
    issues = []
    if not source_id:
        issues.append("visual_source_missing")
    if not visual.get("has_target", quality.get("target_present", False)):
        issues.append("visual_target_missing")
    if confidence < 0.5:
        issues.append("visual_confidence_low")
    uncertainty_score = _finite_score(uncertainty)
    if uncertainty is not None and (uncertainty_score is None or uncertainty_score >= 0.5):
        issues.append("visual_uncertainty_high")
    consistency = visual.get("temporal_consistency")
    if consistency is None:
        issues.append("temporal_consistency_unavailable")
    elif (_finite_score(consistency) is None) or float(consistency) < 0.5:
        issues.append("cross_frame_consistency_low")
    predicted_label = max(probabilities, key=probabilities.get) if probabilities else "unknown"
    stated_label = str(visual.get("label", visual.get("predicted_class", predicted_label)))
    if probabilities and stated_label != predicted_label:
        issues.append("label_probability_mismatch")
    return {
        "analysis_status": "insufficient" if not source_id else "uncertain" if issues else "supported",
        "label": stated_label,
        "probability_top_label": predicted_label,
        "confidence": confidence,
        "class_probabilities": probabilities,
        "uncertainty": uncertainty,
        "temporal_consistency": consistency,
        "frame_ids": list(visual.get("frame_ids", [])),
        "key_frame_ids": list(visual.get("key_frame_ids", [])),
        "source_ids": [source_id] if source_id else [],
        "issues": issues,
        "interpretation": "visual_state_evidence_only",
    }


def evidence_quality(context: Mapping[str, Any]) -> dict[str, Any]:
    quality = _mapping(context.get("quality"))
    visual = context.get("visual") if isinstance(context.get("visual"), Mapping) else {}
    status = str(quality.get("quality_status", "unknown"))
    quality_issues = quality.get("issues", [])
    issues = [str(item) for item in quality_issues] if isinstance(quality_issues, (list, tuple)) else [str(quality_issues)] if quality_issues else []
    missing = []
    if status != "pass":
        missing.append("image_quality_pass")
    if not quality.get("target_present", visual.get("has_target", False)):
        missing.append("visible_target")
    if not (visual.get("source_id") or context.get("media_source_id")):
        missing.append("visual_source_id")
    frame_ids = list(visual.get("frame_ids", []))
    if not frame_ids:
        missing.append("frame_provenance")
    combined = list(dict.fromkeys(issues + missing))
    source_id = str(visual.get("source_id") or context.get("media_source_id") or "")
    return {
        "analysis_status": "complete" if not combined else "incomplete",
        "quality_status": status,
        "quality_issues": issues,
        "missing_requirements": missing,
        "frame_count": len(frame_ids),
        "roi_coverage": visual.get("roi_coverage"),
        "source_ids": [source_id] if source_id else [],
        "issues": combined,
    }


def knowledge_retrieval(context: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(context.get("knowledge_path", ""))) if context.get("knowledge_path") else None
    visual = context.get("visual") if isinstance(context.get("visual"), Mapping) else {}
    process = _mapping(context.get("process_state"))
    terms = ["visual", "fire_eye", "process", "superheat", "report", "review"]
    extra_terms = context.get("rsi_knowledge_query_terms", [])
    if isinstance(extra_terms, (list, tuple)):
        terms.extend(
            " ".join(item.split())
            for item in extra_terms[:4]
            if isinstance(item, str) and item.strip() and len(item) <= 40
        )
    terms.extend((visual.get("label"), visual.get("predicted_class"), visual.get("risk_level"), process.get("process_regime")))
    query = " ".join(str(term) for term in terms if term and term != "unknown")
    if not path or not path.is_file() or not query:
        return {"analysis_status": "abstain", "query": query, "sources": [], "source_ids": [], "issues": ["knowledge_query_or_pack_unavailable"]}
    sources = VersionedKnowledgeBase(path).search_bm25(query, top_k=5, applicability=("visual", "fire_eye", "process", "superheat", "report"))
    valid = [item for item in sources if item.get("source_id") and item.get("source_version") and item.get("locator") and item.get("text")]
    return {
        "analysis_status": "retrieved" if valid else "abstain",
        "query": query,
        "retrieval_mode": "versioned_bm25",
        "sources": valid,
        "source_ids": [str(item["source_id"]) for item in valid],
        "issues": [] if valid else ["no_applicable_knowledge_match"],
    }


def case_comparison(context: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(context.get("case_memory_root", ""))) if context.get("case_memory_root") else None
    features = context.get("case_features") if isinstance(context.get("case_features"), Mapping) else {}
    if not root or not features:
        return {"analysis_status": "abstain", "matches": [], "source_ids": [], "issues": ["case_memory_or_query_features_unavailable"]}
    matches = SpecialCaseMemory(root).search(
        dict(features),
        applicability=("visual", "fire_eye", "process", "report"),
        top_k=5,
    )
    applicable = [item for item in matches if item.get("applicable") and item.get("case_id")]
    return {
        "analysis_status": "matched" if applicable else "abstain",
        "matches": applicable,
        "source_ids": [str(item.get("source_id") or item["case_id"]) for item in applicable],
        "issues": [] if applicable else ["no_approved_applicable_case_match"],
    }


def process_state_analysis(context: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = context.get("process_state") if isinstance(context.get("process_state"), Mapping) else None
    if not snapshot:
        return {
            "analysis_status": "missing",
            "process_missing": ["process_snapshot"],
            "alignment_quality": "unknown",
            "video_relation": "unavailable",
            "process_regime": "unknown",
            "event_risk": None,
            "source_ids": [],
            "issues": ["process_snapshot_unavailable"],
        }
    alignment = str(snapshot.get("alignment_quality", "unknown"))
    missing = snapshot.get("process_missing", [])
    missing = list(missing) if isinstance(missing, list) else [str(missing)]
    source_id = snapshot.get("source_id")
    snapshot_run_id = snapshot.get("run_id") or context.get("run_id")
    if not source_id and snapshot_run_id:
        source_id = f"process-snapshot-{snapshot_run_id}"
    variables = snapshot.get("variables", {})
    names = list(variables.get("by_name", {}).keys()) if isinstance(variables, Mapping) and isinstance(variables.get("by_name"), Mapping) else []
    return {
        "analysis_status": "prior_only" if alignment == "unknown" else "aligned_context",
        "process_missing": missing,
        "alignment_quality": alignment,
        "video_relation": "independent_prior" if alignment == "unknown" else "aligned_window",
        "process_regime": snapshot.get("process_regime", "unknown"),
        "event_risk": snapshot.get("event_risk"),
        "anomaly_score": snapshot.get("anomaly_score"),
        "process_tags": list(snapshot.get("process_tags", [])),
        "variable_count": len(names),
        "evidence_window": snapshot.get("evidence_window"),
        "source_ids": [str(source_id)] if source_id else [],
        "issues": ["video_process_alignment_unknown"] if alignment == "unknown" else [],
    }


def build_full_loop_workers():
    return {
        "visual_review": visual_review,
        "evidence_quality": evidence_quality,
        "knowledge_rag": knowledge_retrieval,
        "case_compare": case_comparison,
        "process_state": process_state_analysis,
    }


__all__ = [
    "build_full_loop_workers", "case_comparison", "evidence_quality",
    "knowledge_retrieval", "process_state_analysis", "visual_review",
]
