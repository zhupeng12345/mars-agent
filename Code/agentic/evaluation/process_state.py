"""Metrics for the independent process-event baseline.

Only the checked-in process labels and baseline predictions are evaluated
here.  This module intentionally has no video input or visual-fusion metric.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agentic.process.data import purged_date_splits


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def roc_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> Optional[float]:
    """Compute ROC-AUC with average ranks for ties without sklearn."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    positive = y == 1
    negative = y == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> Optional[float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    positives = int(np.sum(y == 1))
    if not positives:
        return None
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / max(positives, 1)
    precision = tp / np.maximum(tp + fp, 1)
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(np.r_[0.0, precision], np.r_[0.0, recall]))


def calibration_metrics(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if not len(y):
        return {"brier": 0.0, "ece": 0.0}
    ece = 0.0
    for left, right in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if np.any(mask):
            ece += float(np.sum(mask) / len(y)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}


def event_segments(y_true: Sequence[int]) -> List[Tuple[int, int]]:
    y = np.asarray(y_true, dtype=int)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(y.tolist() + [0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index - 1))
            start = None
    return segments


def event_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5, timestamps: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = p >= threshold
    segments = event_segments(y)
    hits = [bool(np.any(pred[start : end + 1])) for start, end in segments]
    leads: List[float] = []
    segment_results: List[Dict[str, Any]] = []
    parsed: List[Optional[datetime]] = []
    if timestamps is not None and len(timestamps) == len(y):
        for value in timestamps:
            try:
                parsed.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
            except (TypeError, ValueError):
                parsed.append(None)
    for ordinal, (start, end) in enumerate(segments, start=1):
        inside = np.flatnonzero(pred[start : end + 1])
        first_detection = int(start + inside[0]) if len(inside) else None
        peak_index = int(start + np.argmax(p[start : end + 1]))
        lead_minutes = None
        # The source labels indicate an event in a future horizon; they do not
        # provide the physical event onset.  Bound this quantity to 15 sample
        # steps and name it as a lead to the labelled risk-window boundary.
        prior_start = max(0, start - 15)
        prior_alarms = np.flatnonzero(pred[prior_start:start]) + prior_start
        if first_detection is None and len(prior_alarms) and parsed and parsed[start] is not None and parsed[int(prior_alarms[-1])] is not None:
            lead_minutes = (parsed[start] - parsed[int(prior_alarms[-1])]).total_seconds() / 60.0
            leads.append(lead_minutes)
        segment_results.append({
            "event_id": ordinal,
            "start_index": int(start),
            "end_index": int(end),
            "start_timestamp": str(timestamps[start]) if timestamps is not None and start < len(timestamps) else None,
            "end_timestamp": str(timestamps[end]) if timestamps is not None and end < len(timestamps) else None,
            "detected_within_risk_window": bool(len(inside)),
            "first_detection_index": first_detection,
            "peak_score": float(np.max(p[start : end + 1])),
            "peak_index": peak_index,
            "pre_window_alert_lead_minutes": float(lead_minutes) if lead_minutes is not None else None,
        })
    false_alarm_segments = []
    for start, end in event_segments(pred.astype(int)):
        if not np.any(y[start : end + 1]):
            false_alarm_segments.append({
                "start_index": int(start), "end_index": int(end),
                "start_timestamp": str(timestamps[start]) if timestamps is not None and start < len(timestamps) else None,
                "end_timestamp": str(timestamps[end]) if timestamps is not None and end < len(timestamps) else None,
                "peak_score": float(np.max(p[start : end + 1])),
            })
    return {
        "event_count": len(segments),
        "event_hits": int(sum(hits)),
        "event_recall": float(sum(hits) / len(hits)) if hits else None,
        "median_pre_window_alert_lead_minutes": float(np.median(leads)) if leads else None,
        "lead_samples": len(leads),
        "lead_time_semantics": "alert before the first positive future-risk label window within 15 samples; not physical event onset",
        "event_segments": segment_results,
        "false_alarm_segment_count": len(false_alarm_segments),
        "false_alarm_segments": false_alarm_segments,
    }


def evaluate_process_state(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
    timestamps: Optional[Sequence[Any]] = None,
    model_name: str = "process_state",
) -> Dict[str, Any]:
    """Evaluate one process-event prediction stream."""
    y = [int(item) for item in y_true]
    p = [_as_float(item) for item in y_prob]
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have equal lengths")
    calibration = calibration_metrics(y, p)
    truth = np.asarray(y, dtype=int)
    predicted = np.asarray(p, dtype=float) >= float(threshold)
    true_positive = int(np.sum(predicted & (truth == 1)))
    false_positive = int(np.sum(predicted & (truth == 0)))
    false_negative = int(np.sum((~predicted) & (truth == 1)))
    true_negative = int(np.sum((~predicted) & (truth == 0)))
    point_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    point_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    point_f1 = 2 * point_precision * point_recall / (point_precision + point_recall) if point_precision + point_recall else 0.0
    result: Dict[str, Any] = {
        "model": model_name,
        "n": len(y),
        "positive_rate": float(np.mean(y)) if y else 0.0,
        "threshold": float(threshold),
        "predicted_positive_rate": float(np.mean(predicted)) if len(predicted) else 0.0,
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": point_f1,
        "point_specificity": specificity,
        "confusion_counts": {"tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative},
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "calibration": calibration,
        "brier": calibration["brier"],
    }
    result.update(event_metrics(y, p, threshold=threshold, timestamps=timestamps))
    return result


def _column(row: Mapping[str, str], names: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in row:
            return row[name]
    return default


def _read_prediction(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows.extend(reader)
    if not rows:
        raise ValueError(f"empty baseline prediction file: {path}")
    index_column = next((name for name in ("sample_idx", "idx", "row_idx") if name in fieldnames), None)
    if index_column is None:
        raise ValueError(f"prediction file must include sample_idx, idx, or row_idx: {path}")
    y: List[int] = []
    p: List[float] = []
    timestamps: List[str] = []
    indices: List[int] = []
    for row in rows:
        y_value = _column(row, ("y_true", "y", "label"))
        p_value = _column(row, ("y_prob", "prob", "probability", "score"))
        if y_value is None or p_value is None:
            raise ValueError(f"prediction file needs y/y_prob columns: {path}")
        try:
            y_numeric = float(y_value)
            p_item = float(p_value)
            index_numeric = float(row[index_column])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid label, probability, or sample index in {path}") from exc
        if not np.isfinite(y_numeric) or y_numeric not in (0.0, 1.0):
            raise ValueError(f"prediction labels must be binary in {path}")
        if not np.isfinite(index_numeric) or not index_numeric.is_integer():
            raise ValueError(f"prediction sample indices must be finite integers in {path}")
        if not np.isfinite(p_item) or not 0.0 <= p_item <= 1.0:
            raise ValueError(f"prediction probabilities must be finite values in [0,1] in {path}")
        y_item = int(y_numeric)
        index_item = int(index_numeric)
        y.append(y_item)
        p.append(p_item)
        ts = _column(row, ("anchor_ts", "ts", "timestamp"), "")
        timestamps.append(ts or "")
        indices.append(index_item)
    if len(indices) != len(set(indices)):
        raise ValueError(f"prediction file contains duplicate sample indices: {path}")
    return {
        "y": y,
        "p": p,
        "timestamps": timestamps,
        "indices": indices,
        "index_column": index_column,
        "rows": len(rows),
    }


def _timestamp_ns(value: Any, *, source: str) -> np.datetime64:
    try:
        timestamp = np.datetime64(str(value).strip().replace(" ", "T"), "ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid timestamp {value!r} in {source}") from exc
    if np.isnat(timestamp):
        raise ValueError(f"missing timestamp in {source}")
    return timestamp


def evaluate_baselines(
    data_root: Path,
    *,
    split: str = "test",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Recompute metrics for all checked-in process baseline predictions."""
    root = Path(data_root).resolve()
    dataset_path = next(root.rglob("dataset.npz"))
    split_path = next(root.rglob("split.json"))
    with np.load(dataset_path, allow_pickle=True) as payload:
        all_y_raw = np.asarray(payload["y"])
        all_ts = np.asarray(payload["ts_anchor"])
        all_days = np.asarray(payload["day_anchor"])
        history_steps = int(payload["L"])
        future_steps = int(payload["H"])
        sample_minutes = int(payload["SAMPLE_MINUTES"])
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    effective_splits, split_report = purged_date_splits(
        split_payload,
        all_ts,
        all_days,
        history_steps=history_steps,
        future_steps=future_steps,
        sample_minutes=sample_minutes,
    )
    split_indices = [int(item) for item in effective_splits[split]]
    if all_y_raw.ndim != 1 or all_ts.ndim != 1 or all_days.ndim != 1:
        raise ValueError("dataset labels, timestamps, and days must be one-dimensional")
    try:
        numeric_labels = all_y_raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset labels must be numeric binary values") from exc
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise ValueError("dataset labels must be binary values in {0,1}")
    all_y = numeric_labels.astype(int)
    if len(all_y) != len(all_ts) or len(all_days) != len(all_y):
        raise ValueError("dataset labels, timestamps, and days have inconsistent sample counts")
    expected_y = all_y[split_indices]
    expected_ts = [str(all_ts[index]) for index in split_indices]
    baseline_root = root / "step3_baseline_202111"
    output: Dict[str, Any] = {
        "schema_version": "process-evaluation-1",
        "status": "experimental_baseline",
        "split": split,
        "split_count": len(split_indices),
        "split_validation": split_report,
        "index_validation": "strict_sample_index_and_label_match",
        "models": {},
        "process_missing": ["video_mapping"],
        "alignment_quality": "unknown",
        "visual_process_fusion_evaluated": False,
        "video_classification_improvement_claimed": False,
    }
    for directory in sorted(item for item in baseline_root.iterdir() if item.is_dir()):
        prediction_paths = sorted(directory.glob("*pred*.csv"))
        if not prediction_paths:
            continue
        prediction = _read_prediction(prediction_paths[0])
        # Prediction rows must exactly cover the requested date-purged split.
        indexed = {int(index): pos for pos, index in enumerate(prediction["indices"])}
        expected_indices = set(split_indices)
        observed_indices = set(indexed)
        missing_indices = sorted(expected_indices - observed_indices)
        extra_indices = sorted(observed_indices - expected_indices)
        if missing_indices or extra_indices:
            raise ValueError(
                f"prediction sample-index mismatch for {split} split in {prediction_paths[0]}: "
                f"missing={missing_indices[:10]}, extra={extra_indices[:10]}"
            )
        order = [indexed[index] for index in split_indices]
        y = [prediction["y"][pos] for pos in order]
        p = [prediction["p"][pos] for pos in order]
        if not np.array_equal(np.asarray(y, dtype=int), expected_y):
            mismatches = [
                split_indices[index]
                for index, (observed, expected) in enumerate(zip(y, expected_y.tolist()))
                if int(observed) != int(expected)
            ]
            raise ValueError(
                f"prediction label mismatch for {split} split in {prediction_paths[0]}: "
                f"sample_indices={mismatches[:10]}"
            )
        ts = []
        for position, sample_index, expected in zip(order, split_indices, expected_ts):
            observed = prediction["timestamps"][position]
            if observed:
                observed_ns = _timestamp_ns(observed, source=str(prediction_paths[0]))
                expected_ns = _timestamp_ns(expected, source=str(dataset_path))
                if observed_ns != expected_ns:
                    raise ValueError(
                        f"prediction timestamp mismatch for sample_idx={sample_index} "
                        f"in {prediction_paths[0]}: {observed!r} != {expected!r}"
                    )
                ts.append(observed)
            else:
                ts.append(expected)
        metrics = evaluate_process_state(y, p, threshold=threshold, timestamps=ts, model_name=directory.name)
        metrics["prediction_source"] = prediction_paths[0].relative_to(root).as_posix()
        metrics["label_match"] = True
        metrics["sample_index_count"] = len(split_indices)
        stored = directory / "metrics.json"
        if stored.exists():
            payload = json.loads(stored.read_text(encoding="utf-8"))
            metrics["reported_metrics"] = payload.get(split, {})
        output["models"][directory.name] = metrics
    return output


__all__ = [
    "calibration_metrics", "event_metrics", "event_segments", "evaluate_baselines",
    "evaluate_process_state", "pr_auc", "roc_auc",
]
