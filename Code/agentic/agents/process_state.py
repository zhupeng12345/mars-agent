"""Deterministic process-state features for the independent process branch.

The process branch consumes the 14 recorded process variables only.  It does
not infer a video label and it never silently aligns a process window to a
video.  The small rule encoder is deliberately auditable: every score is
computed from window statistics, trends, missingness and detected changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from agentic.platform.contracts import AlignmentQuality, ProcessStateSnapshot


# This is the order stored in Code/data/v1/step2_ew_202111/dataset.npz.  The
# names are intentionally kept as recorded; they are proxy process variables,
# not direct measurements of bath chemistry or superheat.
DATASET_FEATURE_COLUMNS: Tuple[str, ...] = (
    "settingVolt", "potVolt", "potCurrent", "settingVoltMax", "settingVoltMin",
    "aeWaitInterval", "fluctDelta", "wavingDelta", "slopeData", "sumSlopeData",
    "baseFeedingInter", "actualFeedingInter", "feedingState", "andoeMoveState",
)

PROCESS_VARIABLE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "electrical_state": (
        "potVolt", "potCurrent", "settingVolt", "settingVoltMin", "settingVoltMax",
    ),
    "fluctuation_dynamics": (
        "fluctDelta", "wavingDelta", "slopeData", "sumSlopeData",
    ),
    "feeding_operation": (
        "baseFeedingInter", "actualFeedingInter", "feedingState",
    ),
    "equipment_state": ("aeWaitInterval", "andoeMoveState"),
}

PROCESS_VARIABLES: Tuple[str, ...] = tuple(
    name for names in PROCESS_VARIABLE_GROUPS.values() for name in names
)
VARIABLE_TO_GROUP = {
    name: group for group, names in PROCESS_VARIABLE_GROUPS.items() for name in names
}


@dataclass(frozen=True)
class ProcessStateConfig:
    """Versioned thresholds for the transparent process-state encoder."""

    version: str = "process-state-rules-1.0"
    expected_steps: int = 30
    sample_minutes: int = 2
    transition_score: float = 0.42
    abnormal_score: float = 0.72
    change_z: float = 3.0
    max_change_points: int = 8


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _clip01(value: float) -> float:
    return float(np.clip(_finite(value), 0.0, 1.0))


def _timestamp(value: Any) -> str:
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[ms]").astype(datetime)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    text = str(value)
    return text if text and text != "None" else ""


def _slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)[valid]
    y = values[valid].astype(float)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    return float(np.dot(x, y - float(np.mean(y))) / denom) if denom else 0.0


class ProcessStateAgent:
    """Extract a ``ProcessStateSnapshot`` from one process history window."""

    agent_id = "process_state"

    def __init__(
        self,
        *,
        config: Optional[ProcessStateConfig] = None,
        feature_columns: Sequence[str] = DATASET_FEATURE_COLUMNS,
    ) -> None:
        self.config = config or ProcessStateConfig()
        self.feature_columns = tuple(feature_columns)
        if len(self.feature_columns) != 14:
            raise ValueError("ProcessStateAgent expects exactly 14 feature columns")
        unknown = set(self.feature_columns) - set(PROCESS_VARIABLES)
        if unknown:
            raise ValueError(f"unknown process variables: {sorted(unknown)}")

    def _coerce_window(
        self, window: Any, *, feature_columns: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Return a float window, its columns and missing variable names."""
        columns = tuple(feature_columns or self.feature_columns)
        if isinstance(window, Mapping):
            values: List[np.ndarray] = []
            missing: List[str] = []
            length = 0
            for name in columns:
                raw = window.get(name)
                if raw is None:
                    missing.append(name)
                    values.append(np.asarray([], dtype=float))
                    continue
                array = np.asarray(raw, dtype=float).reshape(-1)
                length = max(length, array.size)
                values.append(array)
            if not length:
                return np.empty((0, len(columns)), dtype=float), list(columns), list(columns)
            result = np.full((length, len(columns)), np.nan, dtype=float)
            for index, array in enumerate(values):
                if array.size:
                    result[: min(length, array.size), index] = array[:length]
            missing.extend(name for name, col in zip(columns, result.T) if not np.isfinite(col).any())
            return result, list(columns), sorted(set(missing))

        array = np.asarray(window, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError(f"process window must have shape (steps, {len(columns)})")
        missing = [name for name, col in zip(columns, array.T) if not np.isfinite(col).any()]
        return array, list(columns), missing

    def _window_descriptor(
        self, timestamps: Optional[Sequence[Any]], steps: int,
    ) -> Dict[str, Any]:
        values = [_timestamp(item) for item in (timestamps if timestamps is not None else [])]
        values = [item for item in values if item]
        descriptor: Dict[str, Any] = {
            "n_steps": int(steps),
            "sample_minutes": self.config.sample_minutes,
            "start": values[0] if values else "",
            "end": values[-1] if values else "",
        }
        if values:
            descriptor["timestamps_observed"] = len(values)
        return descriptor

    def _change_points(
        self, array: np.ndarray, columns: Sequence[str], timestamps: Optional[Sequence[Any]],
    ) -> List[Dict[str, Any]]:
        if array.shape[0] < 2:
            return []
        candidates: List[Dict[str, Any]] = []
        for col_index, name in enumerate(columns):
            values = array[:, col_index]
            finite = np.where(np.isfinite(values), values, np.nanmedian(values[np.isfinite(values)]) if np.isfinite(values).any() else 0.0)
            diffs = np.abs(np.diff(finite))
            if not diffs.size:
                continue
            baseline = float(np.median(diffs))
            mad = float(np.median(np.abs(diffs - baseline)))
            scale = max(mad * 1.4826, float(np.std(diffs)) * 0.5, 1e-9)
            score = (diffs - baseline) / scale
            for index in np.where(score >= self.config.change_z)[0]:
                candidates.append({
                    "step": int(index + 1),
                    "variable": name,
                    "score": _clip01(float(score[index]) / (self.config.change_z * 2.0)),
                    "delta": _finite(diffs[index]),
                    "timestamp": _timestamp(timestamps[index + 1]) if timestamps is not None and index + 1 < len(timestamps) else "",
                })
        # Keep the strongest points while retaining one record per step where
        # several process variables move together.
        candidates.sort(key=lambda item: (-item["score"], item["step"], item["variable"]))
        selected: List[Dict[str, Any]] = []
        seen_steps: set[int] = set()
        for item in candidates:
            step = int(item["step"])
            if step in seen_steps:
                for existing in selected:
                    if existing["step"] == step:
                        existing.setdefault("variables", []).append(item["variable"])
                        existing["score"] = max(existing["score"], item["score"])
                        break
                continue
            selected.append({**item, "variables": [item["variable"]]})
            seen_steps.add(step)
            if len(selected) >= self.config.max_change_points:
                break
        for item in selected:
            item.pop("variable", None)
        return sorted(selected, key=lambda item: item["step"])

    def _variable_stats(self, array: np.ndarray, columns: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for index, name in enumerate(columns):
            raw = array[:, index]
            finite = raw[np.isfinite(raw)]
            filled = np.where(np.isfinite(raw), raw, np.nanmedian(finite) if finite.size else 0.0)
            mean = float(np.mean(filled)) if filled.size else 0.0
            std = float(np.std(filled)) if filled.size else 0.0
            stats[name] = {
                "group": VARIABLE_TO_GROUP.get(name, "unknown"),
                "mean": mean,
                "std": std,
                "min": float(np.min(filled)) if filled.size else 0.0,
                "max": float(np.max(filled)) if filled.size else 0.0,
                "first": _finite(filled[0]) if filled.size else 0.0,
                "last": _finite(filled[-1]) if filled.size else 0.0,
                "delta": _finite(filled[-1] - filled[0]) if filled.size else 0.0,
                "slope_per_step": _finite(_slope(raw)),
                "missing": bool(not np.isfinite(raw).all()),
                "observed_steps": int(finite.size),
            }
        return stats

    def _group_scores(self, stats: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for group, names in PROCESS_VARIABLE_GROUPS.items():
            values: List[float] = []
            for name in names:
                item = stats.get(name)
                if not item:
                    continue
                scale = abs(float(item["mean"])) + abs(float(item["std"])) + 1e-6
                relative_delta = abs(float(item["delta"])) / scale
                variation = float(item["std"]) / (abs(float(item["mean"])) + 1e-6)
                values.append(_clip01(relative_delta * 2.0 + variation * 2.0))
            scores[group] = float(np.mean(values)) if values else 0.0
        return scores

    def run(
        self,
        window: Any = None,
        *,
        sample_index: int = 0,
        timestamps: Optional[Sequence[Any]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        video_mapping: Optional[Mapping[str, Any]] = None,
        alignment_quality: Optional[str] = None,
        process_missing: Optional[Iterable[str]] = None,
        run_id: str = "",
        thread_id: str = "",
    ) -> ProcessStateSnapshot:
        """Return a validated process snapshot.

        ``video_mapping`` is optional by design.  Without an auditable mapping,
        the snapshot carries ``video_mapping`` in ``process_missing`` and
        ``alignment_quality=unknown``.  This prevents the independent process
        model from being mistaken for a video classifier.
        """
        run_id = run_id or f"process-{uuid4().hex[:12]}"
        if isinstance(window, (str, Path)):
            dataset_path = Path(window)
            if dataset_path.suffix.lower() != ".npz":
                raise ValueError("path input must point to a dataset.npz file")
            with np.load(dataset_path, allow_pickle=True) as payload:
                sequence = np.asarray(payload["X_seq"])
                if sample_index < 0 or sample_index >= len(sequence):
                    raise IndexError(f"sample_index out of range: {sample_index}")
                columns = [str(item) for item in payload["feature_cols"].tolist()]
                return self.run(
                    sequence[sample_index], sample_index=sample_index,
                    timestamps=timestamps, feature_columns=columns,
                    video_mapping=video_mapping, alignment_quality=alignment_quality,
                    process_missing=process_missing, run_id=run_id, thread_id=thread_id,
                )
        if window is None:
            descriptor = self._window_descriptor(timestamps, 0)
            missing = set(process_missing or ())
            missing.update(("process_window", "video_mapping"))
            snapshot = ProcessStateSnapshot(
                process_missing=sorted(missing), alignment_quality=AlignmentQuality.UNKNOWN.value,
                process_window=descriptor, process_regime="unknown", event_risk=None,
                anomaly_score=None, process_tags=["process_missing"], evidence_window=descriptor,
                variables={}, missing_mask={}, run_id=run_id, thread_id=thread_id,
                model_profile=self.config.version, implementation="agentic.agents.process_state",
            )
            snapshot.validate()
            return snapshot

        array, columns, missing_columns = self._coerce_window(window, feature_columns=feature_columns)
        if not array.shape[0]:
            return self.run(None, timestamps=timestamps, video_mapping=video_mapping,
                            alignment_quality=alignment_quality, process_missing=missing_columns,
                            run_id=run_id, thread_id=thread_id)
        stats = self._variable_stats(array, columns)
        group_scores = self._group_scores(stats)
        changes = self._change_points(array, columns, timestamps)
        observed = np.asarray([value for item in stats.values() for value in (item["std"], item["delta"])], dtype=float)
        variability = _clip01(float(np.mean(np.abs(observed))) / (float(np.mean(np.abs([item["mean"] for item in stats.values()]))) + 1e-6) / 2.0)
        anomaly_score = _clip01(0.65 * float(np.mean(list(group_scores.values()))) + 0.25 * variability + 0.10 * min(1.0, len(changes) / 3.0))
        event_risk = anomaly_score
        if anomaly_score >= self.config.abnormal_score:
            regime = "abnormal"
        elif anomaly_score >= self.config.transition_score or changes:
            regime = "transition"
        else:
            regime = "stable"

        tags: List[str] = []
        for group, score in group_scores.items():
            if score >= self.config.transition_score:
                tags.append(f"{group}_unstable")
        if changes:
            tags.append("change_point_detected")
        if missing_columns:
            tags.append("missing_data")
        for name, item in stats.items():
            if abs(float(item["slope_per_step"])) > (abs(float(item["mean"])) + 1e-6) * 0.02:
                tags.append("trend_up" if item["slope_per_step"] > 0 else "trend_down")
                break
        mapping = dict(video_mapping or {})
        quality = alignment_quality or mapping.get("alignment_quality") or AlignmentQuality.UNKNOWN.value
        if quality not in {item.value for item in AlignmentQuality}:
            raise ValueError(f"invalid alignment_quality: {quality}")
        missing = set(process_missing or ())
        missing.update(missing_columns)
        if quality == AlignmentQuality.UNKNOWN.value:
            missing.add("video_mapping")
        descriptor = self._window_descriptor(timestamps, array.shape[0])
        descriptor["feature_columns"] = list(columns)
        # The dataset contract names a 30-step history as 60 minutes; the
        # timestamp span between the first and last anchor is 58 minutes.
        descriptor["history_minutes"] = int(array.shape[0] * self.config.sample_minutes)
        snapshot = ProcessStateSnapshot(
            process_missing=sorted(missing), alignment_quality=quality,
            process_window=descriptor, process_regime=regime,
            event_risk=event_risk, anomaly_score=anomaly_score,
            change_points=changes, process_tags=sorted(set(tags)), evidence_window=descriptor,
            variables={"by_name": stats, "groups": group_scores},
            missing_mask={name: bool(item["missing"]) for name, item in stats.items()},
            run_id=run_id, thread_id=thread_id, model_profile=self.config.version,
            implementation="agentic.agents.process_state",
        )
        snapshot.validate()
        return snapshot

    analyze = run
    predict = run

    def run_from_npz(self, dataset_path: Path | str, *, sample_index: int = 0, **kwargs: Any) -> ProcessStateSnapshot:
        return self.run(dataset_path, sample_index=sample_index, **kwargs)

    def run_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Mapping convenience for JSONL/CLI callers."""
        return self.run(*args, **kwargs).to_dict()


def build_process_state_snapshot(*args: Any, **kwargs: Any) -> ProcessStateSnapshot:
    return ProcessStateAgent().run(*args, **kwargs)


__all__ = [
    "DATASET_FEATURE_COLUMNS", "PROCESS_VARIABLE_GROUPS", "PROCESS_VARIABLES",
    "ProcessStateAgent", "ProcessStateConfig", "build_process_state_snapshot",
]
