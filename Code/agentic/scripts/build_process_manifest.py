"""Audit the independent ``Code/data/v1`` process dataset.

The manifest is an audit artifact, not a training cache.  It records source
hashes, tensor shapes, the temporal contract, date split boundaries and the
checked-in CNN/1DCNN/LSTM baseline files.  No video mapping is inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from agentic.agents.process_state import PROCESS_VARIABLE_GROUPS, PROCESS_VARIABLES


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "v1"
EXPECTED_COLUMNS = tuple(PROCESS_VARIABLES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path, root: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _find_required(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name), key=lambda path: path.as_posix().lower())
    if not candidates:
        raise FileNotFoundError(f"missing process asset: {name} under {root}")
    return candidates[0]


def _split_summary(
    split: Mapping[str, Iterable[int]],
    *,
    labels: np.ndarray,
    timestamps: np.ndarray,
    days: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    ranges: List[tuple[str, int, int]] = []
    for name in ("train", "val", "test"):
        indices = [int(item) for item in split.get(name, [])]
        if any(item < 0 or item >= len(labels) for item in indices):
            raise ValueError(f"{name} split contains an out-of-range sample index")
        dates = [str(days[item]) for item in indices]
        ts = [str(timestamps[item]) for item in indices]
        counts = {str(label): int(np.sum(labels[indices] == label)) for label in sorted(set(labels.tolist()))} if indices else {}
        result[name] = {
            "count": len(indices),
            "first_index": indices[0] if indices else None,
            "last_index": indices[-1] if indices else None,
            "first_timestamp": ts[0] if ts else None,
            "last_timestamp": ts[-1] if ts else None,
            "first_day": dates[0] if dates else None,
            "last_day": dates[-1] if dates else None,
            "label_counts": counts,
            "indices_are_sorted": indices == sorted(indices),
        }
        if indices:
            ranges.append((name, indices[0], indices[-1]))
    result["ordered_without_overlap"] = all(
        left[2] < right[1] for left, right in zip(ranges, ranges[1:])
    )
    return result


def _baseline_records(root: Path) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    baseline_root = root / "step3_baseline_202111"
    if not baseline_root.exists():
        return records
    for directory in sorted(item for item in baseline_root.iterdir() if item.is_dir()):
        metrics_path = directory / "metrics.json"
        prediction_paths = sorted(directory.glob("*pred*.csv"))
        item: Dict[str, Any] = {"path": directory.relative_to(root).as_posix()}
        if metrics_path.exists():
            item["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            item["metrics_source"] = _source_record(metrics_path, root)
        predictions: List[Dict[str, Any]] = []
        for path in prediction_paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            predictions.append({
                "source": _source_record(path, root),
                "columns": header,
                "rows": row_count,
            })
        if predictions:
            item["predictions"] = predictions
        records[directory.name] = item
    return records


def build_process_manifest(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    dataset_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a JSON-safe process-data audit manifest."""
    root = Path(data_root).resolve()
    dataset = Path(dataset_path).resolve() if dataset_path else _find_required(root, "dataset.npz")
    split_file = Path(split_path).resolve() if split_path else _find_required(root, "split.json")
    with np.load(dataset, allow_pickle=True) as payload:
        keys = sorted(payload.files)
        required = {"X_seq", "X_stat", "y", "ts_anchor", "day_anchor", "feature_cols", "L", "H", "SAMPLE_MINUTES"}
        missing_keys = sorted(required - set(keys))
        if missing_keys:
            raise ValueError(f"dataset.npz missing keys: {missing_keys}")
        x_seq = np.asarray(payload["X_seq"])
        x_stat = np.asarray(payload["X_stat"])
        labels = np.asarray(payload["y"]).astype(int)
        timestamps = np.asarray(payload["ts_anchor"])
        days = np.asarray(payload["day_anchor"])
        feature_columns = [str(item) for item in payload["feature_cols"].tolist()]
        temporal = {
            "history_steps": int(payload["L"]),
            "future_steps": int(payload["H"]),
            "history_minutes": int(payload["L_MIN"]),
            "future_minutes": int(payload["H_MIN"]),
            "sample_minutes": int(payload["SAMPLE_MINUTES"]),
        }
        if len(timestamps) > 1:
            deltas = np.diff(timestamps.astype("datetime64[m]").astype("int64"))
            regular = deltas[deltas <= max(10, temporal["sample_minutes"] * 2)]
            gaps = deltas[deltas > max(10, temporal["sample_minutes"] * 2)]
            temporal["observed_sampling_minutes_median"] = int(np.median(deltas))
            temporal["observed_sampling_minutes_unique"] = sorted(set(int(item) for item in regular.tolist()))
            temporal["sampling_gap_count"] = int(len(gaps))
            temporal["sampling_gap_minutes_unique"] = sorted(set(int(item) for item in gaps.tolist()))
        split = json.loads(split_file.read_text(encoding="utf-8"))
    if len(feature_columns) != 14 or set(feature_columns) != set(EXPECTED_COLUMNS):
        raise ValueError(f"unexpected 14-variable feature columns: {feature_columns}")
    if x_seq.ndim != 3 or x_seq.shape[1:] != (temporal["history_steps"], 14):
        raise ValueError(f"unexpected X_seq shape: {x_seq.shape}")
    if len(labels) != x_seq.shape[0] or len(timestamps) != len(labels):
        raise ValueError("dataset arrays have inconsistent sample counts")
    summary = _split_summary(split, labels=labels, timestamps=timestamps, days=days)
    sources = [_source_record(dataset, root), _source_record(split_file, root)]
    for path in sorted(root.glob("*.csv"), key=lambda item: item.name.lower()):
        sources.append(_source_record(path, root))
    manifest: Dict[str, Any] = {
        "schema_version": "process-manifest-1",
        "manifest_id": "process-v1-step2-ew-202111",
        "generated_at": generated_at or _utc_now(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "dataset": {
            "root": root.name,
            "dataset_path": dataset.relative_to(root).as_posix(),
            "split_path": split_file.relative_to(root).as_posix(),
            "keys": keys,
            "sample_count": int(x_seq.shape[0]),
            "x_seq_shape": list(x_seq.shape),
            "x_stat_shape": list(x_stat.shape),
            "label_values": sorted(set(int(item) for item in labels.tolist())),
            "label_counts": {str(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))},
            "feature_columns": feature_columns,
        },
        "temporal_contract": temporal,
        "split": summary,
        "variable_groups": {name: list(values) for name, values in PROCESS_VARIABLE_GROUPS.items()},
        "variable_notes": {
            name: {"source_name": name, "unit": "unspecified_in_source", "is_proxy": True}
            for name in feature_columns
        },
        "baseline": _baseline_records(root),
        "sources": sources,
        "video_mapping": {
            "available": False,
            "alignment_quality": "unknown",
            "process_missing": ["video_mapping"],
            "note": "The process windows are independently audited and are not video labels.",
        },
        "claims": {
            "process_event_baseline_only": True,
            "video_classification_improvement_reported": False,
            "visual_process_fusion_evaluated": False,
        },
    }
    return _json_value(manifest)


def audit_process_data(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return build_process_manifest(*args, **kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Code/data/v1 process windows and baselines")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_process_manifest(args.data_root, dataset_path=args.dataset, split_path=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "sample_count": manifest["dataset"]["sample_count"],
        "splits": {key: value["count"] for key, value in manifest["split"].items() if isinstance(value, dict) and "count" in value},
        "alignment_quality": manifest["video_mapping"]["alignment_quality"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_process_data", "build_process_manifest", "main"]
