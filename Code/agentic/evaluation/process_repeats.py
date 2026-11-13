"""Repeated-seed evaluation for the independent process-state baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Mapping

from agentic.training.train_process import train_process


METRICS = (
    "roc_auc", "pr_auc", "event_recall", "false_alarm_segment_count",
    "predicted_positive_rate", "point_precision", "point_recall", "point_f1",
    "point_specificity", "brier", "ece",
)


def summarize_repeated_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(reports)
    if not rows:
        raise ValueError("at least one process training report is required")
    dataset_hashes = {str(row.get("dataset_sha256", "")) for row in rows}
    split_hashes = {
        str((row.get("split") or {}).get("effective_split_index_sha256", {}).get("test", ""))
        for row in rows
    }
    if len(dataset_hashes) != 1 or not next(iter(dataset_hashes)):
        raise ValueError("repeated reports must use the same identified dataset")
    if len(split_hashes) != 1 or not next(iter(split_hashes)):
        raise ValueError("repeated reports must use the same effective test split")

    summaries: dict[str, Any] = {}
    for metric in METRICS:
        values = []
        for row in rows:
            metric_row = row.get("metrics") or {}
            raw = metric_row.get(metric)
            if metric == "ece" and raw is None:
                raw = (metric_row.get("calibration") or {}).get("ece")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                continue
            values.append(float(raw))
        summaries[metric] = {
            "n": len(values),
            "mean": mean(values) if values else None,
            "std_population": pstdev(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "values": values,
        }
    return {
        "schema_version": "process-repeated-evaluation-1",
        "status": "experimental_baseline",
        "model": rows[0].get("model", "unknown"),
        "seeds": [int(row["seed"]) for row in rows],
        "run_count": len(rows),
        "dataset_sha256": next(iter(dataset_hashes)),
        "split_sha256": rows[0].get("split_sha256", ""),
        "effective_test_split_sha256": next(iter(split_hashes)),
        "test_windows": rows[0].get("test_count"),
        "metrics": summaries,
        "video_mapping": {"available": False, "alignment_quality": "unknown"},
        "claims": {
            "repeatability_only": True,
            "video_classification_improvement": False,
            "video_process_fusion_evaluated": False,
            "real_superheat_measurement": False,
        },
        "runs": [
            {
                "seed": row.get("seed"),
                "checkpoint": row.get("checkpoint"),
                "test_predictions": row.get("test_predictions"),
                "metrics": row.get("metrics"),
            }
            for row in rows
        ],
    }


def run_repeated_process_evaluation(
    dataset: Path,
    split: Path,
    output_dir: Path,
    *,
    seeds: Iterable[int] = (7, 17, 27, 37, 47),
    model_name: str = "cnn",
    epochs: int = 1,
    batch_size: int = 128,
    device: str = "cpu",
    train_fn: Callable[..., dict[str, Any]] = train_process,
) -> dict[str, Any]:
    seed_values = [int(item) for item in seeds]
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be a nonempty list of unique integers")
    output_dir = Path(output_dir)
    reports = [
        train_fn(
            Path(dataset),
            Path(split),
            output_dir / f"seed_{seed}",
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            evaluate=True,
        )
        for seed in seed_values
    ]
    summary = summarize_repeated_reports(reports)
    summary_path = output_dir / "repeated_process_evaluation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "repeated_process_evaluation_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "n", "mean", "std_population", "min", "max"))
        writer.writeheader()
        for metric, values in summary["metrics"].items():
            writer.writerow({"metric": metric, **{key: values[key] for key in ("n", "mean", "std_population", "min", "max")}})
    summary["artifacts"] = {"json": str(summary_path), "csv": str(csv_path)}
    return summary


__all__ = ["METRICS", "run_repeated_process_evaluation", "summarize_repeated_reports"]
