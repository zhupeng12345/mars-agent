"""Auditable ablation matrix for the fire-eye multi-agent system.

The matrix is deliberately model-agnostic.  It records the exact comparison
that must be run and accepts externally produced results only when their
provenance is present.  Missing expert labels, GPU runs, or paired video/
process samples are represented as ``structure_only`` or ``not_run`` rather
than replaced with invented numbers.
"""

from __future__ import annotations

import json
import math
import platform
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


ABLATION_SPECS = (
    ("single_frame_vs_multi_frame", "single-frame baseline", "multi-frame temporal model", "visual labels and windows"),
    ("multi_frame_vs_expert_features", "multi-frame model", "multi-frame + static/dynamic expert features", "visual labels and windows"),
    ("expert_features_vs_fgmvae_t", "expert-feature fusion", "F-GMVAE-T fusion", "visual labels, GPU training or checkpoint"),
    ("without_process_vs_with_process", "visual-only diagnosis", "visual + Process State Agent", "process manifest and paired/independent provenance"),
    ("without_jev_vs_with_jev", "always full loop", "JEV conditional routing", "route manifest and trace replay"),
    ("without_case_gate_vs_conditioned_case_gate", "unfiltered case retrieval", "approved-case condition gate", "case audit and route fixture"),
    ("lexical_vs_hybrid_retrieval", "lexical knowledge retrieval", "lexical + vector/reranker retrieval", "knowledge query set and retrieval logs"),
    ("without_hitl_vs_with_hitl", "automatic finalization", "interrupt/resume human review", "HITL audit records"),
    ("local_model_vs_qwen_api", "local/fake model profile", "Qwen API profile", "paired model traces and API metadata"),
)


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    baseline: str
    variant: str
    required_evidence: str


def specs() -> list[AblationSpec]:
    return [AblationSpec(*row) for row in ABLATION_SPECS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_result_overrides(value: Optional[Mapping[str, Any] | Path | str]) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items() if isinstance(item, Mapping)}
    path = Path(value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, Mapping) and isinstance(payload.get("results"), Mapping):
        payload = payload["results"]
    return {str(key): item for key, item in payload.items()} if isinstance(payload, Mapping) else {}


def _artifact_state(root: Path, ablation_id: str) -> tuple[str, list[str]]:
    visual_manifest = root / "data" / "evaluation" / "visual_manifest.jsonl"
    route_manifest = root / "data" / "evaluation" / "route_manifest.jsonl"
    process_manifest = root / "data" / "evaluation" / "process_manifest.json"
    hitl_audit = root / "runtime_data" / "hitl_audit.jsonl"
    knowledge_queries = root / "data" / "evaluation" / "knowledge_queries.jsonl"
    if ablation_id == "without_process_vs_with_process":
        return ("structure_only", [str(process_manifest)]) if process_manifest.exists() else ("not_run", [str(process_manifest)])
    if ablation_id == "without_jev_vs_with_jev":
        return ("structure_only", [str(route_manifest)]) if route_manifest.exists() else ("not_run", [str(route_manifest)])
    if ablation_id == "without_hitl_vs_with_hitl":
        return ("structure_only", [str(hitl_audit)]) if hitl_audit.exists() else ("not_run", [str(hitl_audit)])
    if ablation_id == "lexical_vs_hybrid_retrieval":
        return ("structure_only", [str(knowledge_queries)]) if knowledge_queries.exists() else ("not_run", [str(knowledge_queries)])
    if visual_manifest.exists():
        return "structure_only", [str(visual_manifest)]
    return "not_run", [str(visual_manifest)]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_override(root: Path, supplied: Mapping[str, Any]) -> Optional[tuple[dict[str, float], dict[str, Any]]]:
    raw_metrics = supplied.get("metrics")
    provenance_value = supplied.get("provenance")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics or not isinstance(provenance_value, Mapping):
        return None
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(key, str) or not key.strip() or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        metrics[key] = numeric

    provenance = dict(provenance_value)
    required_text = ("data_version", "split", "model", "generated_at")
    if any(not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in required_text):
        return None
    seed = provenance.get("seed")
    if isinstance(seed, bool) or seed is None or not str(seed).strip():
        return None
    paired_count = provenance.get("paired_sample_count")
    if isinstance(paired_count, bool) or not isinstance(paired_count, int) or paired_count <= 0:
        return None

    digest_fields = (
        ("paired_data_path", "paired_data_sha256"),
        ("baseline_artifact_path", "baseline_artifact_sha256"),
        ("variant_artifact_path", "variant_artifact_sha256"),
    )
    for path_key, digest_key in digest_fields:
        raw_path = provenance.get(path_key)
        expected_digest = provenance.get(digest_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest):
            return None
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = root / artifact
        if not artifact.is_file():
            return None
        try:
            actual_digest = _sha256(artifact)
        except OSError:
            return None
        if actual_digest.casefold() != expected_digest.casefold():
            return None
    return metrics, provenance


def build_ablation_report(
    root: Path,
    *,
    result_overrides: Optional[Mapping[str, Any] | Path | str] = None,
    generated_at: Optional[str] = None,
) -> dict[str, Any]:
    root = Path(root)
    overrides = _load_result_overrides(result_overrides)
    rows: list[dict[str, Any]] = []
    for spec in specs():
        status, artifacts = _artifact_state(root, spec.ablation_id)
        supplied = overrides.get(spec.ablation_id)
        metrics: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        if supplied:
            validated = _validated_override(root, supplied)
            if validated is not None:
                metrics, provenance = validated
                status = "completed"
            else:
                raw_provenance = supplied.get("provenance", {})
                provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
                status = "structure_only"
        rows.append({
            **asdict(spec),
            "status": status,
            "metrics": metrics,
            "provenance": provenance,
            "artifacts": artifacts,
            "claimable": status == "completed",
        })
    return {
        "schema_version": "ablation-report-1",
        "generated_at": generated_at or _now(),
        "root": str(root),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "result_policy": "Numeric comparisons require provenance; missing labels/GPU/pairs remain structure_only or not_run.",
        "comparisons": rows,
        "completed_count": sum(item["status"] == "completed" for item in rows),
        "structure_only_count": sum(item["status"] == "structure_only" for item in rows),
        "not_run_count": sum(item["status"] == "not_run" for item in rows),
    }


__all__ = ["ABLATION_SPECS", "AblationSpec", "build_ablation_report", "specs"]
