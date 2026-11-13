"""Deterministic offline replay for RSI candidates.

Replay is intentionally separate from the production evaluator.  A caller can
inject a domain evaluator, while the default evaluator computes transparent
metrics from annotated JSON rows and never edits the rows or their manifests.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from .candidate import Candidate
from .records import jsonable, stable_hash, utc_now


SplitRows = Mapping[str, Iterable[Mapping[str, Any]]]
EvaluatorFn = Callable[..., Mapping[str, Any]]


def _read_rows(source: Any) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
            return rows
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        return list(value) if isinstance(value, list) else [value] if isinstance(value, dict) else []
    return [dict(row) for row in source if isinstance(row, Mapping)]


@dataclass(frozen=True)
class ReplayBundle:
    """Immutable train/frozen/sealed row collections."""

    train: tuple[dict[str, Any], ...] = ()
    frozen: tuple[dict[str, Any], ...] = ()
    sealed: tuple[dict[str, Any], ...] = ()
    manifest_hashes: dict[str, str] = field(default_factory=dict)
    schema_version: str = "rsi-replay-bundle-1"

    @classmethod
    def from_mapping(cls, datasets: Mapping[str, Any]) -> "ReplayBundle":
        rows = {name: tuple(_read_rows(datasets.get(name, []))) for name in ("train", "frozen", "sealed")}
        hashes = {name: stable_hash(rows[name]) for name in rows}
        return cls(train=rows["train"], frozen=rows["frozen"], sealed=rows["sealed"], manifest_hashes=hashes)

    @classmethod
    def from_paths(cls, *, train: Any = None, frozen: Any = None, sealed: Any = None) -> "ReplayBundle":
        return cls.from_mapping({"train": train, "frozen": frozen, "sealed": sealed})

    def split(self, name: str) -> tuple[dict[str, Any], ...]:
        if name not in {"train", "frozen", "sealed"}:
            raise ValueError(f"unknown replay split: {name}")
        return tuple(copy.deepcopy(getattr(self, name)))


@dataclass
class ReplayResult:
    candidate_id: str
    split: str
    metrics: dict[str, float]
    row_count: int
    input_hash: str
    reference_errors: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now)
    schema_version: str = "rsi-replay-result-1"

    @property
    def passed(self) -> bool:
        required = {"quality", "safety", "efficiency"}
        scores = [self.metrics.get(name) for name in required]
        valid_scores = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) <= 1.0
            for value in scores
        )
        finite_metrics = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in self.metrics.values()
        )
        return self.row_count > 0 and valid_scores and finite_metrics and not self.errors and not self.reference_errors

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayResult":
        return cls(**dict(payload))


def _number(row: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, bool):
            return float(value)
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score {name} must be in [0,1]")
        return score
    return None


def _default_metrics(candidate: Candidate, rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot evaluate an empty replay split")
    quality_values: list[float] = []
    safety_values: list[float] = []
    efficiency_values: list[float] = []
    route_hits = 0
    route_total = 0
    violations = 0
    for row in rows:
        quality = _number(row, ("quality", "quality_score", "quality_pass", "evidence_quality"))
        if quality is None:
            raise ValueError("required metric missing: quality")
        quality_values.append(quality)
        safety = _number(row, ("safety", "safety_score", "safety_pass", "safe"))
        if safety is None:
            raise ValueError("required metric missing: safety")
        safety_values.append(safety)
        if safety < 0.5:
            violations += 1
        efficiency = _number(row, ("efficiency", "efficiency_score", "latency_score", "token_score"))
        if efficiency is not None:
            efficiency_values.append(efficiency)
        else:
            latency = row.get("latency_ms")
            tokens = row.get("tokens")
            values = []
            for value, label in ((latency, "latency_ms"), (tokens, "tokens")):
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {label}") from exc
                if numeric < 0:
                    raise ValueError(f"{label} must be non-negative")
                values.append(1.0 / (1.0 + numeric / 1000.0))
            if not values:
                raise ValueError("required metric missing: efficiency or latency_ms/tokens")
            efficiency_values.append(sum(values) / len(values))
        expected = row.get("expert_route", row.get("expected_route"))
        predicted = row.get("candidate_route", row.get("predicted_route", row.get("route")))
        if expected is not None and predicted is not None:
            route_total += 1
            route_hits += int(str(expected) == str(predicted))
    metrics = {
        "quality": sum(quality_values) / len(quality_values),
        "safety": sum(safety_values) / len(safety_values),
        "efficiency": sum(efficiency_values) / len(efficiency_values),
        "safety_violations": float(violations),
    }
    if route_total:
        metrics["route_accuracy"] = route_hits / route_total
    return metrics


def _call_evaluator(fn: EvaluatorFn, candidate: Candidate, rows: list[dict[str, Any]], split: str) -> Mapping[str, Any]:
    """Support evaluator callbacks taking (candidate, rows, split), (candidate, rows), or rows."""
    try:
        signature = inspect.signature(fn)
        count = len([p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        count = 3
    if count >= 3:
        return fn(candidate, rows, split)
    if count == 2:
        return fn(candidate, rows)
    return fn(rows)


def evaluate_candidate(
    candidate: Candidate,
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    evaluator: Optional[EvaluatorFn] = None,
) -> ReplayResult:
    values = [dict(row) for row in rows]
    errors: list[str] = []
    reference_errors: list[str] = []
    try:
        raw = _call_evaluator(evaluator, candidate, copy.deepcopy(values), split) if evaluator else _default_metrics(candidate, values)
        metrics = {str(key): float(value) for key, value in dict(raw).items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        missing = sorted({"quality", "safety", "efficiency"}.difference(metrics))
        if missing:
            errors.append(f"missing_required_metrics:{','.join(missing)}")
        nonfinite = sorted(key for key, value in metrics.items() if not math.isfinite(value))
        if nonfinite:
            errors.append(f"nonfinite_metrics:{','.join(nonfinite)}")
        for key in ("quality", "safety", "efficiency"):
            if key in metrics and not 0.0 <= metrics[key] <= 1.0:
                errors.append(f"metric_out_of_range:{key}")
    except Exception as exc:  # replay must produce an auditable failed result
        metrics = {"quality": 0.0, "safety": 0.0, "efficiency": 0.0}
        errors.append(f"{type(exc).__name__}: {exc}")
    return ReplayResult(
        candidate_id=candidate.candidate_id,
        split=split,
        metrics=metrics,
        row_count=len(values),
        input_hash=stable_hash(values),
        reference_errors=reference_errors,
        errors=errors,
    )


class OfflineReplayer:
    def __init__(self, evaluator: Optional[EvaluatorFn] = None):
        self.evaluator = evaluator

    def replay(self, candidate: Candidate, rows: Iterable[Mapping[str, Any]], *, split: str) -> ReplayResult:
        if split not in {"train", "frozen", "sealed"}:
            raise ValueError(f"replay split must be train, frozen, or sealed: {split}")
        # Candidate validation happens before any callback is called.
        candidate.validate()
        return evaluate_candidate(candidate, tuple(rows), split=split, evaluator=self.evaluator)

    def replay_bundle(self, candidate: Candidate, bundle: ReplayBundle) -> dict[str, ReplayResult]:
        return {name: self.replay(candidate, bundle.split(name), split=name) for name in ("train", "frozen", "sealed")}


def replay_candidate(
    candidate: Candidate,
    datasets: Mapping[str, Any],
    *,
    evaluator: Optional[EvaluatorFn] = None,
) -> dict[str, ReplayResult]:
    """Functional wrapper for train/frozen/sealed replay."""
    return OfflineReplayer(evaluator).replay_bundle(candidate, ReplayBundle.from_mapping(datasets))


__all__ = ["OfflineReplayer", "ReplayBundle", "ReplayResult", "evaluate_candidate", "replay_candidate"]
