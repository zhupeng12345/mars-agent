"""Deterministic JEV safety policy.

The policy chooses a diagnostic loop. It never predicts C/N/H or a measured
superheat temperature.
"""

from __future__ import annotations

from typing import Any, Mapping


def safety_route(features: Any) -> tuple[str, list[str]]:
    values: Mapping[str, Any] = getattr(features, "features", {}) or {}
    quality_status = str(values.get("quality_status", "unknown"))
    if quality_status == "retake" or values.get("decoder_error"):
        return "retake", ["quality_gate_failed"]
    if quality_status == "human":
        return "human", ["quality_requires_human_review"]
    if not bool(values.get("target_present", False)):
        return "retake", ["target_missing"]

    novelty = float(getattr(features, "novelty_score", 0.0) or 0.0)
    confidence = float(getattr(features, "visual_confidence", 0.0) or 0.0)
    risk = float(getattr(features, "risk_score", 0.0) or 0.0)
    conflict = float(getattr(features, "evidence_conflict", 0.0) or 0.0)
    coverage = float(getattr(features, "knowledge_coverage", 0.0) or 0.0)
    case_similarity = float(getattr(features, "case_similarity", 0.0) or 0.0)
    case_applicable = bool(values.get("case_applicable", False))

    if novelty >= 0.85:
        return "human", ["novel_case_requires_human_review"]
    if confidence < 0.45:
        return "human", ["visual_confidence_below_human_threshold"]
    if risk >= 0.75 or conflict >= 0.45:
        return "full_loop", ["high_risk_or_evidence_conflict"]
    if coverage < 0.34:
        return "full_loop", ["knowledge_coverage_insufficient"]
    if case_similarity >= 0.88 and case_applicable and risk < 0.45 and conflict < 0.2:
        return "simple_loop", ["applicable_special_case_match"]
    if confidence >= 0.82 and case_similarity >= 0.7 and risk < 0.45 and conflict < 0.2:
        return "simple_loop", ["stable_visual_and_case_evidence"]
    return "full_loop", ["requires_multi_agent_evidence"]
