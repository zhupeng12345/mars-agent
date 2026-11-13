"""Label-aware visual metrics with an explicit unlabeled-manifest state."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def classification_metrics(y_true: Sequence[str], probabilities: Sequence[Mapping[str, float]], labels: Sequence[str] = ("C", "H", "N")) -> dict[str, Any]:
    names = list(labels)
    truth = [str(value) for value in y_true]
    if len(truth) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths")
    if not truth:
        return {"metric_status": "empty", "n": 0, "macro_f1": None, "accuracy": None, "confusion_matrix": []}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    correct = []
    brier = []
    normalized_probabilities = []
    for target, raw in zip(truth, probabilities):
        if target not in names:
            raise ValueError(f"unknown visual label: {target}")
        if not isinstance(raw, Mapping):
            raise ValueError("each prediction must be a probability mapping")
        scores = np.asarray([float(raw.get(name, 0.0)) for name in names], dtype=float)
        if not np.isfinite(scores).all() or (scores < 0.0).any():
            raise ValueError("prediction probabilities must be finite and non-negative")
        mass = float(scores.sum())
        if mass <= 0.0:
            raise ValueError("prediction probability mass must be positive")
        scores = scores / mass
        normalized_probabilities.append(scores)
        predicted = names[int(np.argmax(scores))]
        if target in names:
            matrix[names.index(target), names.index(predicted)] += 1
        correct.append(float(predicted == target))
        target_one_hot = np.asarray([float(target == name) for name in names])
        brier.append(float(np.sum((scores - target_one_hot) ** 2)))
    f1_values = []
    for index in range(len(names)):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        f1_values.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    confidences = [float(scores.max()) for scores in normalized_probabilities]
    ece = float(np.mean([abs(confidence - correct_value) for confidence, correct_value in zip(confidences, correct)]))
    return {
        "metric_status": "computed",
        "n": len(truth),
        "labels": names,
        "accuracy": float(np.mean(correct)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_f1": {name: float(value) for name, value in zip(names, f1_values)},
        "confusion_matrix": matrix.tolist(),
        "ece": ece,
        "brier": float(np.mean(brier)),
    }


def evaluate_visual_manifest(rows: Iterable[Mapping[str, Any]], *, predictions: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    records = list(rows)
    labeled = [row for row in records if row.get("human_label") in {"C", "H", "N"}]
    if not labeled:
        return {"metric_status": "structure_validation_only", "records": len(records), "human_labeled_records": 0, "expert_accuracy_available": False}
    predictions = predictions or {}
    missing = []
    for row in labeled:
        window_id = str(row.get("window_id", ""))
        probabilities = predictions.get(window_id)
        if not window_id or not isinstance(probabilities, Mapping) or not set(("C", "H", "N")).issubset(probabilities):
            missing.append(window_id or "<missing-window-id>")
    if missing:
        return {
            "metric_status": "prediction_missing",
            "records": len(records),
            "human_labeled_records": len(labeled),
            "predicted_records": len(labeled) - len(missing),
            "missing_prediction_count": len(missing),
            "missing_prediction_window_ids": missing,
            "expert_accuracy_available": False,
        }
    truth = [str(row["human_label"]) for row in labeled]
    probs = [predictions.get(str(row.get("window_id")), {}) for row in labeled]
    report = classification_metrics(truth, probs)
    report.update({"records": len(records), "human_labeled_records": len(labeled), "expert_accuracy_available": True})
    return report


__all__ = ["classification_metrics", "evaluate_visual_manifest"]
