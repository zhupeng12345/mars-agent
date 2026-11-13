"""Validation-only permutation attribution for independent process events."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from agentic.agents.process_state import DATASET_FEATURE_COLUMNS, PROCESS_VARIABLE_GROUPS
from agentic.evaluation.process_state import evaluate_process_state


Predictor = Callable[[np.ndarray], np.ndarray]


def _validate_inputs(
    windows: np.ndarray,
    labels: np.ndarray,
    feature_columns: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, tuple[str, ...]]]:
    x = np.asarray(windows, dtype=np.float32)
    y_raw = np.asarray(labels)
    columns = tuple(str(item) for item in feature_columns)
    normalized_groups = {str(name): tuple(str(item) for item in values) for name, values in groups.items()}
    if x.ndim != 3 or x.shape[0] != len(y_raw) or x.shape[2] != len(columns):
        raise ValueError("windows must have shape [N,T,F] with matching labels and feature_columns")
    if len(columns) != 14 or len(set(columns)) != len(columns) or set(columns) != set(DATASET_FEATURE_COLUMNS):
        raise ValueError("feature_columns must identify each of the 14 recorded process variables exactly once")
    if y_raw.ndim != 1 or not np.isfinite(y_raw.astype(np.float64)).all() or not np.isin(y_raw, (0, 1)).all():
        raise ValueError("labels must be finite binary values in {0,1}")
    grouped = [item for values in normalized_groups.values() for item in values]
    if len(grouped) != len(columns) or len(set(grouped)) != len(grouped) or set(grouped) != set(columns):
        raise ValueError("groups must cover every recorded feature exactly once")
    return x, y_raw.astype(int), columns, normalized_groups


def _scores(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | None]:
    result = evaluate_process_state(labels.tolist(), probabilities.tolist(), threshold=threshold)
    return {
        "roc_auc": result["roc_auc"],
        "pr_auc": result["pr_auc"],
        "brier": result["brier"],
        "event_recall": result["event_recall"],
    }


def _summarize_deltas(
    baseline: Mapping[str, float | None],
    perturbed: Sequence[Mapping[str, float | None]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for metric in ("roc_auc", "pr_auc", "event_recall"):
        base_value = baseline.get(metric)
        values = [base_value - row[metric] for row in perturbed if isinstance(base_value, (int, float)) and isinstance(row.get(metric), (int, float))]
        rows[f"{metric}_drop_mean"] = float(np.mean(values)) if values else None
        rows[f"{metric}_drop_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
    base_brier = baseline.get("brier")
    brier_increases = [row["brier"] - base_brier for row in perturbed if isinstance(base_brier, (int, float)) and isinstance(row.get("brier"), (int, float))]
    rows["brier_increase_mean"] = float(np.mean(brier_increases)) if brier_increases else None
    rows["brier_increase_std"] = float(np.std(brier_increases, ddof=1)) if len(brier_increases) > 1 else (0.0 if brier_increases else None)
    return rows


def permutation_importance(
    predict: Predictor,
    windows: np.ndarray,
    labels: np.ndarray,
    feature_columns: Sequence[str],
    *,
    groups: Mapping[str, Sequence[str]] = PROCESS_VARIABLE_GROUPS,
    repetitions: int = 5,
    seed: int = 7,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Rank validation features by held-out window permutation impact.

    A sample permutation moves the complete temporal trajectory for one
    variable. Group permutation applies one shared sample order to all group
    members, preserving their within-group association.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    x, y, columns, normalized_groups = _validate_inputs(windows, labels, feature_columns, groups)
    if len(np.unique(y)) < 2:
        raise ValueError("validation labels must contain both event classes")
    baseline_probabilities = np.asarray(predict(x.copy()), dtype=float).reshape(-1)
    if len(baseline_probabilities) != len(y) or not np.isfinite(baseline_probabilities).all():
        raise ValueError("predictor must return one finite probability per window")
    if np.any((baseline_probabilities < 0.0) | (baseline_probabilities > 1.0)):
        raise ValueError("predictor probabilities must be in [0,1]")
    baseline = _scores(y, baseline_probabilities, threshold)

    def run_permutations(names: Sequence[str], item_seed: int) -> dict[str, Any]:
        indices = [columns.index(name) for name in names]
        results = []
        for repeat in range(repetitions):
            order = np.random.default_rng(item_seed + repeat).permutation(len(x))
            permuted = x.copy()
            permuted[:, :, indices] = x[order][:, :, indices]
            probabilities = np.asarray(predict(permuted), dtype=float).reshape(-1)
            if len(probabilities) != len(y) or not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError("predictor returned invalid probabilities for a permutation")
            results.append(_scores(y, probabilities, threshold))
        return _summarize_deltas(baseline, results) | {"repetitions": repetitions}

    feature_rows = []
    for index, name in enumerate(columns):
        feature_rows.append({"feature": name, "group": next(group for group, values in normalized_groups.items() if name in values), **run_permutations((name,), seed + index * 101)})
    group_rows = []
    for index, (name, members) in enumerate(normalized_groups.items()):
        group_rows.append({"group": name, "features": list(members), **run_permutations(members, seed + 10000 + index * 101)})
    feature_rows.sort(key=lambda row: (row["pr_auc_drop_mean"] is None, -(row["pr_auc_drop_mean"] or 0.0), row["feature"]))
    group_rows.sort(key=lambda row: (row["pr_auc_drop_mean"] is None, -(row["pr_auc_drop_mean"] or 0.0), row["group"]))
    return {
        "evaluation_kind": "validation_window_permutation_attribution",
        "interpretation": "predictive association under window permutation; not causal contribution",
        "baseline_validation_metrics": baseline,
        "feature_importance": feature_rows,
        "group_importance": group_rows,
        "seed": seed,
        "repetitions": repetitions,
        "threshold": threshold,
    }


def select_top_features(importance: Mapping[str, Any], *, top_k: int = 8) -> list[str]:
    """Select a deterministic compact candidate from validation attribution."""
    rows = importance.get("feature_importance")
    if not isinstance(rows, list) or not rows or not 1 <= top_k <= len(rows):
        raise ValueError("importance must contain feature rows and top_k must be in range")
    ranked = sorted(rows, key=lambda row: (row.get("pr_auc_drop_mean") is None, -(row.get("pr_auc_drop_mean") or 0.0), str(row.get("feature", ""))))
    selected = [str(row["feature"]) for row in ranked[:top_k] if row.get("feature")]
    if len(selected) != top_k:
        raise ValueError("feature attribution contains missing or duplicate feature names")
    return selected


__all__ = ["permutation_importance", "select_top_features"]
