"""Quality, safety, and efficiency gates for offline candidate promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .candidate import Candidate
from .evaluator import ReplayResult
from .records import jsonable, utc_now


@dataclass(frozen=True)
class GateConfig:
    """Conservative defaults: frozen/sealed safety cannot regress."""

    min_quality: float = 0.50
    min_safety: float = 0.95
    min_efficiency: float = 0.0
    quality_tolerance: float = 0.0
    safety_tolerance: float = 0.0
    efficiency_tolerance: float = 0.05
    require_all_splits: bool = True
    schema_version: str = "rsi-gate-config-1"


@dataclass
class GateResult:
    candidate_id: str
    accepted: bool
    checks: dict[str, bool]
    reasons: list[str]
    metrics: dict[str, Any]
    candidate_hash: str = ""
    generated_at: str = field(default_factory=utc_now)
    gate_version: str = "rsi-gate-1"

    @property
    def passed(self) -> bool:
        return self.accepted

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateResult":
        required = ("candidate_id", "accepted", "checks", "reasons", "metrics", "candidate_hash")
        if not isinstance(payload, Mapping) or any(key not in payload for key in required):
            raise ValueError("gate result is incomplete")
        checks = payload.get("checks")
        metrics = payload.get("metrics")
        reasons = payload.get("reasons")
        if not isinstance(checks, Mapping) or not isinstance(metrics, Mapping) or not isinstance(reasons, list):
            raise ValueError("gate result has invalid checks, metrics, or reasons")
        return cls(
            candidate_id=payload.get("candidate_id"),
            accepted=payload.get("accepted"),
            checks=dict(checks),
            reasons=list(reasons),
            metrics=dict(metrics),
            candidate_hash=payload.get("candidate_hash"),
            generated_at=str(payload.get("generated_at", "")),
            gate_version=str(payload.get("gate_version", "")),
        )

    def validate_for(self, candidate: Candidate) -> bool:
        """Fail closed unless this complete gate belongs to this exact diff."""
        candidate.validate()
        if self.candidate_id != candidate.candidate_id or self.candidate_hash != candidate.diff_hash:
            raise ValueError("gate result does not match the candidate id and diff")
        if self.gate_version != "rsi-gate-1":
            raise ValueError("unsupported RSI gate version")
        if self.accepted is not True or self.reasons:
            raise ValueError("candidate has no accepted gate result")
        required_checks = {"candidate_valid", "all_splits_present", "all_splits_nonempty"}
        for split in ("train", "frozen", "sealed"):
            required_checks.update({
                f"{split}_input_hash_match",
                f"{split}_required_metrics_present",
                f"{split}_quality",
                f"{split}_safety",
                f"{split}_efficiency",
                f"{split}_no_safety_violations",
            })
        if not required_checks.issubset(self.checks) or any(value is not True for value in self.checks.values()):
            raise ValueError("gate result is missing required successful checks")
        for split in ("train", "frozen", "sealed"):
            row = self.metrics.get(split)
            if not isinstance(row, Mapping):
                raise ValueError(f"{split} replay metrics are required")
            for count_name in ("candidate_rows", "baseline_rows"):
                count = row.get(count_name)
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{split} {count_name} must be a positive integer")
            candidate_hash = row.get("candidate_hash")
            baseline_hash = row.get("baseline_hash")
            if not isinstance(candidate_hash, str) or not candidate_hash or candidate_hash != baseline_hash:
                raise ValueError(f"{split} replay data hashes must be present and match")
            for side in ("candidate", "baseline"):
                scores = row.get(side)
                if not isinstance(scores, Mapping):
                    raise ValueError(f"{split} {side} metrics are required")
                for name in ("quality", "safety", "efficiency"):
                    value = scores.get(name)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                        raise ValueError(f"{split} {side} {name} metric is invalid")
        return True


def _metrics(value: Any) -> dict[str, float]:
    if isinstance(value, ReplayResult):
        return dict(value.metrics)
    if isinstance(value, Mapping):
        raw = value.get("metrics", value)
        return {str(key): float(item) for key, item in dict(raw).items() if isinstance(item, (int, float)) and not isinstance(item, bool)}
    return {}


def _result_map(results: Any) -> dict[str, dict[str, Any]]:
    if isinstance(results, Mapping):
        output = {}
        for name, value in results.items():
            metrics = _metrics(value)
            if isinstance(value, ReplayResult):
                row_count = value.row_count
                valid = value.passed
                errors = list(value.errors) + list(value.reference_errors)
                input_hash = value.input_hash
            elif isinstance(value, Mapping):
                row_count = int(value.get("row_count", 0) or 0)
                errors = list(value.get("errors", [])) + list(value.get("reference_errors", []))
                input_hash = str(value.get("input_hash", ""))
                required = ("quality", "safety", "efficiency")
                scores_valid = all(
                    name in metrics and math.isfinite(metrics[name]) and 0.0 <= metrics[name] <= 1.0
                    for name in required
                )
                valid = row_count > 0 and not errors and scores_valid
            else:
                row_count, errors, valid, input_hash = 0, ["missing_replay_metadata"], False, ""
            output[str(name)] = {**metrics, "__row_count__": row_count, "__valid__": valid, "__errors__": errors, "__input_hash__": input_hash}
        return output
    if isinstance(results, (list, tuple)):
        output = {}
        for value in results:
            name = value.split if isinstance(value, ReplayResult) else getattr(value, "split", None)
            if name:
                output[str(name)] = {
                    **_metrics(value),
                    "__row_count__": value.row_count if isinstance(value, ReplayResult) else 0,
                    "__valid__": value.passed if isinstance(value, ReplayResult) else False,
                    "__errors__": (list(value.errors) + list(value.reference_errors)) if isinstance(value, ReplayResult) else ["missing_replay_metadata"],
                    "__input_hash__": value.input_hash if isinstance(value, ReplayResult) else "",
                }
        return output
    return {}


def evaluate_gate(
    candidate: Candidate,
    baseline_results: Any,
    candidate_results: Any,
    *,
    config: Optional[GateConfig] = None,
) -> GateResult:
    """Compare candidate to baseline across train, frozen and sealed replays."""
    cfg = config or GateConfig()
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    try:
        candidate.validate()
    except Exception as exc:
        return GateResult(candidate.candidate_id, False, {"candidate_valid": False}, [str(exc)], {}, candidate_hash=candidate.diff_hash)
    checks["candidate_valid"] = True
    baseline = _result_map(baseline_results)
    proposed = _result_map(candidate_results)
    expected_splits = ("train", "frozen", "sealed")
    if cfg.require_all_splits:
        checks["all_splits_present"] = all(name in baseline and name in proposed for name in expected_splits)
        if not checks["all_splits_present"]:
            reasons.append("train, frozen and sealed results are required")
        checks["all_splits_nonempty"] = all(
            name in baseline and name in proposed
            and baseline[name].get("__valid__", False) and proposed[name].get("__valid__", False)
            and baseline[name].get("__row_count__", 0) > 0 and proposed[name].get("__row_count__", 0) > 0
            for name in expected_splits
        )
        if not checks["all_splits_nonempty"]:
            reasons.append("every baseline and candidate split must contain valid replay rows")
    else:
        checks["all_splits_present"] = True

    metric_report: dict[str, Any] = {}
    for split in expected_splits:
        current, previous = proposed.get(split, {}), baseline.get(split, {})
        if not current:
            checks[f"{split}_required_metrics_present"] = False
            continue
        required = ("quality", "safety", "efficiency")
        candidate_metrics_present = all(
            name in current and math.isfinite(current[name]) and 0.0 <= current[name] <= 1.0
            for name in required
        )
        baseline_metrics_present = all(
            name in previous and math.isfinite(previous[name]) and 0.0 <= previous[name] <= 1.0
            for name in required
        ) if previous else False
        checks[f"{split}_required_metrics_present"] = candidate_metrics_present and baseline_metrics_present
        if not checks[f"{split}_required_metrics_present"]:
            reasons.append(f"{split} baseline and candidate must include finite quality, safety and efficiency scores in [0,1]")
        quality = current.get("quality")
        safety = current.get("safety")
        efficiency = current.get("efficiency")
        metric_report[split] = {
            "candidate": {key: value for key, value in current.items() if not key.startswith("__")},
            "baseline": {key: value for key, value in previous.items() if not key.startswith("__")},
            "candidate_rows": current.get("__row_count__", 0),
            "baseline_rows": previous.get("__row_count__", 0),
        }

        quality_ok = candidate_metrics_present and quality >= cfg.min_quality
        safety_ok = candidate_metrics_present and safety >= cfg.min_safety
        efficiency_ok = candidate_metrics_present and efficiency >= cfg.min_efficiency
        if split in {"frozen", "sealed"} and previous:
            quality_ok = quality_ok and baseline_metrics_present and quality >= previous["quality"] - cfg.quality_tolerance
            safety_ok = safety_ok and baseline_metrics_present and safety >= previous["safety"] - cfg.safety_tolerance
            efficiency_ok = efficiency_ok and baseline_metrics_present and efficiency >= previous["efficiency"] - cfg.efficiency_tolerance
        checks[f"{split}_quality"] = quality_ok
        checks[f"{split}_safety"] = safety_ok
        checks[f"{split}_efficiency"] = efficiency_ok
        if not quality_ok:
            reasons.append(f"{split} quality below gate or regressed")
        if not safety_ok:
            reasons.append(f"{split} safety below gate or regressed")
        if not efficiency_ok:
            reasons.append(f"{split} efficiency below gate or regressed")
        if current.get("safety_violations", 0.0) > 0:
            checks[f"{split}_no_safety_violations"] = False
            reasons.append(f"{split} contains safety violations")
        else:
            checks[f"{split}_no_safety_violations"] = True

        candidate_input_hash = str(current.get("__input_hash__", ""))
        baseline_input_hash = str(previous.get("__input_hash__", ""))
        checks[f"{split}_input_hash_match"] = bool(candidate_input_hash and baseline_input_hash and candidate_input_hash == baseline_input_hash)
        if not checks[f"{split}_input_hash_match"]:
            reasons.append(f"{split} baseline and candidate replay data hashes must match")
        metric_report[split].update({
            "candidate_hash": candidate_input_hash,
            "baseline_hash": baseline_input_hash,
        })

    accepted = all(checks.values()) if checks else False
    result = GateResult(candidate.candidate_id, accepted, checks, reasons, metric_report, candidate_hash=candidate.diff_hash)
    candidate.status = "gated" if accepted else "rejected"
    return result


run_gate = evaluate_gate


class Gate:
    """State-free object wrapper for integrations that keep a gate instance."""

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(self, candidate: Candidate, baseline_results: Any, candidate_results: Any) -> GateResult:
        return evaluate_gate(candidate, baseline_results, candidate_results, config=self.config)


__all__ = ["Gate", "GateConfig", "GateResult", "evaluate_gate", "run_gate"]
