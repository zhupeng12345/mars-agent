"""Training-only normalization and purged chronological process splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class ProcessNormalizer:
    """Imputation and scale parameters fitted on training windows only."""

    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray) -> "ProcessNormalizer":
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or not len(values):
            raise ValueError("normalizer fit requires nonempty [N,T,F] training windows")
        flat = values.reshape(-1, values.shape[-1])
        median = np.asarray([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in flat.T], dtype=np.float32)
        filled = np.where(np.isfinite(flat), flat, median)
        mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(median, mean, std)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim not in (2, 3) or values.shape[-1] != len(self.mean):
            raise ValueError("process features do not match normalizer dimensions")
        filled = np.where(np.isfinite(values), values, self.median)
        return np.clip((filled - self.mean) / self.std, -20.0, 20.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist(), "fitted_on": "train_only", "clip_z": 20.0}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcessNormalizer":
        arrays = [np.asarray(payload[name], dtype=np.float32) for name in ("median", "mean", "std")]
        if any(value.ndim != 1 or not np.isfinite(value).all() for value in arrays) or len({value.shape for value in arrays}) != 1 or np.any(arrays[2] <= 0):
            raise ValueError("invalid process normalizer parameters")
        return cls(*arrays)


def purged_date_splits(split: Mapping[str, Sequence[int]], timestamps: np.ndarray, days: np.ndarray, *, history_steps: int, future_steps: int, sample_minutes: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate date splits and purge overlapping input/label support.

    An anchor's support is [anchor-(L-1)*dt, anchor+H*dt]. Purging the tail
    of the earlier split keeps its future label horizon out of the next
    split's history window.
    """
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    day = np.asarray(days).astype(str)
    if len(ts) != len(day) or np.isnat(ts).any():
        raise ValueError("invalid process timestamps/day metadata")
    if history_steps < 1 or future_steps < 0 or sample_minutes < 1:
        raise ValueError("invalid temporal window contract")
    indices: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        values = np.asarray(split.get(name, []), dtype=np.int64)
        if not len(values) or values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} split must contain unique, nonempty indices")
        if np.any(values < 0) or np.any(values >= len(ts)):
            raise ValueError(f"{name} split index outside dataset")
        if not np.all(ts[values][1:] >= ts[values][:-1]):
            raise ValueError(f"{name} split is not chronological")
        indices[name] = values
    for left, right in (("train", "val"), ("val", "test"), ("train", "test")):
        if set(indices[left].tolist()) & set(indices[right].tolist()):
            raise ValueError("process splits share sample indices")
        if set(day[indices[left]]) & set(day[indices[right]]):
            raise ValueError("process splits share dates")
        if ts[indices[left]].max() >= ts[indices[right]].min():
            raise ValueError("process splits are not ordered in time")
    original = {name: len(values) for name, values in indices.items()}
    history_delta = np.timedelta64((history_steps - 1) * sample_minutes, "m")
    future_delta = np.timedelta64(future_steps * sample_minutes, "m")
    for left, right in (("val", "test"), ("train", "val")):
        next_history_start = ts[indices[right]].min() - history_delta
        values = indices[left]
        indices[left] = values[(ts[values] + future_delta) < next_history_start]
        if not len(indices[left]):
            raise ValueError(f"purging overlap emptied {left} split")
    report = {
        "policy": "date_disjoint_and_purged_history_future_support_v1",
        "original_counts": original,
        "effective_counts": {name: len(values) for name, values in indices.items()},
        "purged_counts": {name: original[name] - len(values) for name, values in indices.items()},
        "history_steps": history_steps, "future_steps": future_steps, "sample_minutes": sample_minutes,
        "boundaries": {name: {"first_anchor": str(ts[values].min()), "last_anchor": str(ts[values].max()), "history_start": str(ts[values].min() - history_delta), "label_support_end": str(ts[values].max() + future_delta)} for name, values in indices.items()},
        "normalization_fit_split": "train", "dates_disjoint": True, "support_overlap": False,
    }
    return indices, report


__all__ = ["ProcessNormalizer", "purged_date_splits"]
