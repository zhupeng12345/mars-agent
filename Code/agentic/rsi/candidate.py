"""Candidate diffs and mutation guardrails for RSI.

The candidate format is intentionally declarative.  It can describe a change
to prompts, JEV policy parameters, knowledge query templates, or report
templates.  It cannot contain edits to the frozen evaluator, safety rules,
weights, or human conclusions.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4

from .records import jsonable, stable_hash, utc_now


ALLOWED_MUTABLE_SURFACES = frozenset({
    "prompts", "jev_policies", "knowledge_queries", "report_templates",
})
# Public aliases used by callers that describe the policy as a mutable surface.
MUTABLE_SURFACES = ALLOWED_MUTABLE_SURFACES

# These tokens are checked in paths and recursively in metadata/reference keys.
# A JEV policy candidate may tune thresholds, but never the safety ruleset.
FORBIDDEN_TOKENS = frozenset({
    "frozen", "sealed", "evaluation", "evaluator", "eval", "safety_rule",
    "safety_rules", "security_rule", "weights", "weight", "model_weights",
    "human_conclusion", "human_conclusions", "manual_conclusion", "conclusion",
    "expert_conclusion", "production_rule", "online_rule",
})


class CandidateError(ValueError):
    """Base class for invalid RSI candidates."""


class ForbiddenMutationError(CandidateError):
    """Raised when a candidate attempts to modify a frozen surface."""


class CandidateReferenceError(CandidateError):
    """Raised when a candidate refers to evidence not in its declared inputs."""


_MISSING = object()


def _segments(path: str) -> list[str]:
    return [part for part in re.split(r"[./\\]+", str(path).strip()) if part]


def _is_forbidden_path(path: str) -> bool:
    for token in _segments(path):
        normalized = re.sub(r"[^a-z0-9_]+", "_", token.lower()).strip("_")
        if normalized in FORBIDDEN_TOKENS or normalized in {"safety", "safetyrules", "rules"} or any(
            normalized.startswith(prefix) for prefix in ("frozen_", "sealed_", "safety_rule", "human_conclusion")
        ):
            return True
    return False


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value[key], path))
        return output
    return {prefix: copy.deepcopy(value)}


def diff_snapshot(before: Mapping[str, Any], after: Mapping[str, Any], surface: str) -> list["CandidateDiff"]:
    """Return stable leaf-level diffs for one allowed mutable surface."""
    _validate_surface(surface)
    before_part = before.get(surface, {}) if isinstance(before, Mapping) else {}
    after_part = after.get(surface, {}) if isinstance(after, Mapping) else {}
    left, right = _flatten(before_part), _flatten(after_part)
    diffs: list[CandidateDiff] = []
    for path in sorted(set(left) | set(right)):
        old, new = left.get(path, _MISSING), right.get(path, _MISSING)
        if old == new:
            continue
        operation = "add" if old is _MISSING else "remove" if new is _MISSING else "replace"
        diffs.append(CandidateDiff(
            surface=surface,
            path=path,
            before=None if old is _MISSING else old,
            after=None if new is _MISSING else new,
            operation=operation,
        ))
    return diffs


def _validate_surface(surface: str) -> None:
    if surface not in ALLOWED_MUTABLE_SURFACES:
        raise ForbiddenMutationError(
            f"surface {surface!r} is immutable; allowed surfaces are {sorted(ALLOWED_MUTABLE_SURFACES)}"
        )


@dataclass(frozen=True)
class CandidateDiff:
    surface: str
    path: str
    before: Any
    after: Any
    operation: str = "replace"

    def __post_init__(self) -> None:
        _validate_surface(self.surface)
        if not self.path or _is_forbidden_path(self.path):
            raise ForbiddenMutationError(f"forbidden mutation path: {self.surface}.{self.path}")
        if self.operation not in {"add", "remove", "replace"}:
            raise CandidateError(f"invalid diff operation: {self.operation}")

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


def _scan_references(value: Any, path: str = "") -> list[str]:
    """Collect source/evidence identifiers carried by candidate metadata."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key).lower()
            child = f"{path}.{key}" if path else str(key)
            if key_s in {"source_id", "source_ids", "evidence_id", "evidence_ids", "reference_id", "reference_ids"}:
                if isinstance(item, str):
                    found.append(item)
                elif isinstance(item, (list, tuple, set)):
                    found.extend(str(part) for part in item)
            found.extend(_scan_references(item, child))
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            found.extend(_scan_references(item, f"{path}[{index}]"))
    return found


@dataclass
class Candidate:
    candidate_id: str = field(default_factory=lambda: f"cand-{uuid4().hex[:16]}")
    parent_version: str = "base"
    diffs: list[CandidateDiff] = field(default_factory=list)
    source_run_ids: list[str] = field(default_factory=list)
    generator_model: str = "offline-deterministic"
    created_at: str = field(default_factory=utc_now)
    target_metrics: dict[str, Any] = field(default_factory=dict)
    risk_notes: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    base_snapshot: dict[str, Any] = field(default_factory=dict, repr=False)
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "rsi-candidate-1"

    def __post_init__(self) -> None:
        self.diffs = [item if isinstance(item, CandidateDiff) else CandidateDiff(**item) for item in self.diffs]
        self.validate()

    @property
    def surfaces(self) -> set[str]:
        return {item.surface for item in self.diffs}

    @property
    def diff_hash(self) -> str:
        return stable_hash([item.to_dict() for item in self.diffs])

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        if not self.diffs:
            raise CandidateError("candidate must contain at least one diff")
        for item in self.diffs:
            _validate_surface(item.surface)
            if _is_forbidden_path(item.path):
                raise ForbiddenMutationError(f"forbidden mutation path: {item.surface}.{item.path}")
        if any(surface not in ALLOWED_MUTABLE_SURFACES for surface in self.surfaces):
            raise ForbiddenMutationError("candidate includes an immutable surface")
        refs = set(self.source_refs)
        refs.update(_scan_references(self.metadata))
        refs.update(_scan_references(self.target_metrics))
        refs.update(_scan_references([item.after for item in self.diffs]))
        if known_source_ids is not None:
            known = {str(item) for item in known_source_ids}
            unknown = sorted(ref for ref in refs if ref and ref not in known)
            if unknown:
                raise CandidateReferenceError(f"unknown candidate reference: {unknown[0]}")
        if self.status not in {"candidate", "replayed", "gated", "approved", "published", "rolled_back", "rejected"}:
            raise CandidateError(f"invalid candidate status: {self.status}")
        return True

    def to_dict(self, include_base: bool = True) -> dict[str, Any]:
        value = jsonable(self)
        if not include_base:
            value.pop("base_snapshot", None)
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Candidate":
        value = dict(payload)
        value["diffs"] = [item if isinstance(item, CandidateDiff) else CandidateDiff(**item) for item in value.get("diffs", [])]
        return cls(**value)

    def apply(self, snapshot: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Apply declarative diffs to a copy of a version snapshot."""
        output = copy.deepcopy(dict(snapshot if snapshot is not None else self.base_snapshot))
        for change in self.diffs:
            target = output.setdefault(change.surface, {})
            parts = _segments(change.path)
            for part in parts[:-1]:
                current = target.get(part)
                if not isinstance(current, dict):
                    current = {}
                    target[part] = current
                target = current
            leaf = parts[-1]
            if change.operation == "remove":
                target.pop(leaf, None)
            else:
                target[leaf] = copy.deepcopy(change.after)
        return output


def build_candidate(
    *,
    surface: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    source_run_ids: Iterable[str] = (),
    generator_model: str = "offline-deterministic",
    parent_version: str = "base",
    target_metrics: Optional[Mapping[str, Any]] = None,
    risk_notes: Iterable[str] = (),
    source_refs: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    diffs = diff_snapshot(before, after, surface)
    return Candidate(
        parent_version=parent_version,
        diffs=diffs,
        source_run_ids=list(source_run_ids),
        generator_model=generator_model,
        target_metrics=dict(target_metrics or {}),
        risk_notes=list(risk_notes),
        source_refs=list(source_refs),
        base_snapshot=copy.deepcopy(dict(before)),
        metadata=dict(metadata or {}),
    )


generate_candidate = build_candidate


def build_candidate_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    surface: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    generator_model: str = "offline-deterministic",
    parent_version: str = "base",
    target_metrics: Optional[Mapping[str, Any]] = None,
    risk_notes: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    """Generate a candidate while preserving run IDs and references from records."""
    source_run_ids: list[str] = []
    source_refs: list[str] = []
    for record in records:
        payload = record.get("payload", record) if isinstance(record, Mapping) else {}
        if not isinstance(payload, Mapping):
            continue
        run_id = payload.get("run_id")
        if run_id and str(run_id) not in source_run_ids:
            source_run_ids.append(str(run_id))
        for key in ("source_id", "case_id", "reference_id"):
            value = payload.get(key)
            if value and str(value) not in source_refs:
                source_refs.append(str(value))
    return build_candidate(
        surface=surface,
        before=before,
        after=after,
        source_run_ids=source_run_ids,
        source_refs=source_refs,
        generator_model=generator_model,
        parent_version=parent_version,
        target_metrics=target_metrics,
        risk_notes=risk_notes,
        metadata=metadata,
    )


__all__ = [
    "ALLOWED_MUTABLE_SURFACES", "MUTABLE_SURFACES", "Candidate", "CandidateDiff", "CandidateError",
    "CandidateReferenceError", "ForbiddenMutationError", "build_candidate",
    "build_candidate_from_records", "diff_snapshot", "generate_candidate",
]
