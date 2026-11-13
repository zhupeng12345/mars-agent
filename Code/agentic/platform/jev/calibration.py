"""Dependency-free probability calibration helpers for JEV route scores."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

ROUTES = {"simple_loop", "full_loop", "retake", "human"}
MIN_CALIBRATION_ROWS = 10


def calibrate_confidence(scores: Iterable[float], labels: Iterable[bool]) -> dict[str, float]:
    score_values = list(scores)
    label_values = list(labels)
    if len(score_values) != len(label_values):
        raise ValueError("scores and labels must have the same length")
    pairs = []
    for index, (score, label) in enumerate(zip(score_values, label_values)):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"score at row {index} must be a finite number in [0, 1]")
        if not isinstance(label, bool):
            raise ValueError(f"label at row {index} must be boolean")
        pairs.append((float(score), label))
    if not pairs:
        raise ValueError("calibration requires at least one score/label pair")
    brier = sum((score - float(label)) ** 2 for score, label in pairs) / len(pairs)
    best_temperature, best_loss = 1.0, float("inf")
    # A small deterministic grid is sufficient for the route-confidence
    # adapter and keeps calibration reproducible without sklearn.
    for index in range(1, 81):
        temperature = 0.25 + index * 0.05
        calibrated = [1.0 / (1.0 + math.exp(-(score - 0.5) / temperature)) for score, _ in pairs]
        loss = sum((value - float(label)) ** 2 for value, (_, label) in zip(calibrated, pairs)) / len(pairs)
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    return {
        "temperature": float(best_temperature),
        "brier": float(brier),
        "calibrated_brier": float(best_loss),
        "count": float(len(pairs)),
    }


def _route_calibration_rows(rows: Sequence[Mapping[str, Any]], *, min_rows: int) -> tuple[list[dict[str, Any]], list[float], list[bool]]:
    payload = [dict(row) if isinstance(row, Mapping) else row for row in rows]
    if len(payload) < min_rows:
        raise ValueError(f"route calibration requires at least {min_rows} labeled rows; got {len(payload)}")
    scores: list[float] = []
    labels: list[bool] = []
    expert_routes: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"route row {index} must be an object")
        features = row.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"route row {index} must contain an object at features")
        visual_confidence = features.get("visual_confidence")
        if isinstance(visual_confidence, bool) or not isinstance(visual_confidence, (int, float)) or not math.isfinite(float(visual_confidence)) or not 0.0 <= float(visual_confidence) <= 1.0:
            raise ValueError(f"route row {index} features.visual_confidence must be a finite number in [0, 1]")
        expert_route = row.get("expert_route")
        predicted_route = row.get("predicted_route", row.get("route"))
        if not isinstance(expert_route, str) or expert_route not in ROUTES:
            raise ValueError(f"route row {index} expert_route must be one of {sorted(ROUTES)}")
        if not isinstance(predicted_route, str) or predicted_route not in ROUTES:
            raise ValueError(f"route row {index} requires a valid predicted_route or route")
        confidence = row.get("route_confidence", row.get("confidence", visual_confidence))
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"route row {index} confidence must be a finite number in [0, 1]")
        scores.append(float(confidence))
        labels.append(predicted_route == expert_route)
        expert_routes.add(str(expert_route))
    if len(expert_routes) < 2:
        raise ValueError("route calibration requires at least two expert_route classes")
    if len(set(labels)) < 2:
        raise ValueError("route calibration requires both correct and incorrect route predictions")
    return payload, scores, labels


def fit_route_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_rows: int = MIN_CALIBRATION_ROWS,
) -> dict[str, Any]:
    """Fit confidence calibration from schema-checked nested route records.

    Each row must contain ``features.visual_confidence``, ``expert_route`` and
    an observed ``predicted_route`` (or ``route``). A route score can be
    supplied as ``route_confidence``/``confidence``; visual confidence is only
    used as an explicit proxy when the route score is absent.
    """
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    payload, scores, labels = _route_calibration_rows(rows, min_rows=min_rows)
    result = calibrate_confidence(scores, labels)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    confidence_sources = {
        "route_confidence": sum("route_confidence" in row for row in payload),
        "confidence": sum("route_confidence" not in row and "confidence" in row for row in payload),
        "features.visual_confidence_proxy": sum("route_confidence" not in row and "confidence" not in row for row in payload),
    }
    return {
        "schema_version": "jev-calibration-2",
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "row_count": len(payload),
        "expert_route_classes": sorted({str(row["expert_route"]) for row in payload}),
        "confidence_sources": confidence_sources,
        "target": "observed_route_matches_expert_route",
        **result,
    }


def apply_temperature(score: float, temperature: float) -> float:
    value = max(0.0, min(1.0, float(score)))
    scale = max(1e-6, float(temperature))
    return float(1.0 / (1.0 + math.exp(-(value - 0.5) / scale)))


__all__ = ["MIN_CALIBRATION_ROWS", "ROUTES", "apply_temperature", "calibrate_confidence", "fit_route_calibration"]
