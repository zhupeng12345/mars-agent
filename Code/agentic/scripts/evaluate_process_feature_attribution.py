"""Train process-only baselines and compare 14-variable/compact inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agentic.agents.process_state import DATASET_FEATURE_COLUMNS
from agentic.evaluation.process_feature_attribution import permutation_importance, select_top_features
from agentic.evaluation.process_state import evaluate_process_state
from agentic.process.data import purged_date_splits
from agentic.process.inference import ProcessCheckpointPredictor
from agentic.training.train_process import train_process


PLANNED_COMPACT_8 = (
    "potVolt", "potCurrent", "fluctDelta", "wavingDelta",
    "slopeData", "sumSlopeData", "actualFeedingInter", "feedingState",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dataset(dataset: Path, split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, np.ndarray], dict[str, Any]]:
    with np.load(dataset, allow_pickle=True) as payload:
        x = np.asarray(payload["X_seq"], dtype=np.float32)
        y = np.asarray(payload["y"])
        timestamps = np.asarray(payload["ts_anchor"])
        days = np.asarray(payload["day_anchor"])
        columns = [str(item) for item in payload["feature_cols"].tolist()]
        history_steps = int(payload["L"])
        future_steps = int(payload["H"])
        sample_minutes = int(payload["SAMPLE_MINUTES"])
    if x.ndim != 3 or x.shape[2] != 14 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("dataset X_seq/y must have matching [N,T,14] and [N] shapes")
    if len(columns) != 14 or len(set(columns)) != 14 or set(columns) != set(DATASET_FEATURE_COLUMNS):
        raise ValueError("dataset feature_cols must identify the 14 recorded variables exactly once")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    indices, split_report = purged_date_splits(
        split,
        timestamps,
        days,
        history_steps=history_steps,
        future_steps=future_steps,
        sample_minutes=sample_minutes,
    )
    if len(np.unique(y[indices["val"]])) < 2 or len(np.unique(y[indices["test"]])) < 2:
        raise ValueError("validation and test splits must each contain both event classes")
    contract = {"history_steps": history_steps, "future_steps": future_steps, "sample_minutes": sample_minutes}
    return x, y.astype(int), timestamps, columns, indices, split_report | {"temporal_contract": contract}


def _predict_windows(predictor: ProcessCheckpointPredictor, windows: np.ndarray, input_columns: list[str], *, batch_size: int = 256) -> np.ndarray:
    columns = list(input_columns)
    order = [columns.index(name) for name in predictor.feature_columns]
    selected = np.asarray(windows[:, :, order], dtype=np.float32)
    normalized = predictor.normalizer.transform(selected)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size])
            outputs.extend(torch.sigmoid(predictor.model(batch)).cpu().numpy().tolist())
    return np.asarray(outputs, dtype=float)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_process_feature_attribution(
    dataset: Path,
    split: Path,
    output_dir: Path,
    *,
    model_name: str = "cnn",
    epochs: int = 1,
    batch_size: int = 128,
    seed: int = 7,
    device: str = "cpu",
    repetitions: int = 5,
) -> dict[str, Any]:
    dataset, split = Path(dataset), Path(split)
    output_dir = Path(output_dir)
    x, y, timestamps, columns, indices, split_report = _load_dataset(dataset, split)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dir = output_dir / "full14"
    full_training = train_process(dataset, split, full_dir, model_name=model_name, epochs=epochs, batch_size=batch_size, seed=seed, device=device, feature_names=columns, evaluate=False)
    full_predictor = ProcessCheckpointPredictor(full_training["checkpoint"])
    validation = indices["val"]
    validation_predict = lambda value: _predict_windows(full_predictor, value, columns)
    importance = permutation_importance(
        validation_predict,
        x[validation],
        y[validation],
        columns,
        repetitions=repetitions,
        seed=seed,
        threshold=0.5,
    )
    selected8 = select_top_features(importance, top_k=8)
    candidates = {
        "all_14": columns,
        "planned_compact_8": list(PLANNED_COMPACT_8),
        "validation_selected_8": selected8,
    }
    training: dict[str, dict[str, Any]] = {"all_14": full_training}
    for candidate, features in list(candidates.items())[1:]:
        training[candidate] = train_process(
            dataset,
            split,
            output_dir / candidate,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            feature_names=features,
            evaluate=False,
        )

    test = indices["test"]
    timestamps_test = [str(item) for item in timestamps[test]]
    comparison: list[dict[str, Any]] = []
    for candidate, features in candidates.items():
        predictor = ProcessCheckpointPredictor(training[candidate]["checkpoint"])
        probabilities = _predict_windows(predictor, x[test], columns)
        metrics = evaluate_process_state(y[test].tolist(), probabilities.tolist(), threshold=0.5, timestamps=timestamps_test, model_name=candidate)
        comparison.append({
            "candidate": candidate,
            "feature_count": len(features),
            "features": features,
            "status": "experimental_baseline",
            **metrics,
        })

    report = {
        "schema_version": "process-feature-attribution-1",
        "status": "experimental_baseline",
        "model": model_name,
        "epochs": max(1, int(epochs)),
        "seed": seed,
        "normalization_fit_split": "train_only",
        "selection_split": "validation",
        "evaluation_split": "test",
        "test_evaluations_per_candidate": 1,
        "selection_method": "validation_window_permutation_importance_top_8_by_pr_auc_drop",
        "permutation": importance,
        "planned_compact_features": list(PLANNED_COMPACT_8),
        "validation_selected_features": selected8,
        "test_comparison": comparison,
        "split": split_report,
        "provenance": {
            "dataset_sha256": _sha256(dataset),
            "split_sha256": _sha256(split),
            "effective_split_counts": split_report["effective_counts"],
            "feature_columns": columns,
        },
        "claims": {
            "interpretation": "associative model attribution for the independent process-event label",
            "causal_contribution": False,
            "video_fusion_evaluated": False,
            "real_superheat_measurement": False,
        },
    }
    report_path = output_dir / "feature_attribution_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "feature_importance_validation.csv", importance["feature_importance"])
    _write_csv(output_dir / "group_importance_validation.csv", importance["group_importance"])
    _write_csv(output_dir / "feature_subset_test_comparison.csv", comparison)
    return report


def main(argv: list[str] | None = None) -> int:
    runtime_dir = Path(os.environ.get("MARS_AGENT_RUNTIME_DIR", Path(__file__).resolve().parents[2] / "runtime_data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/v1/step2_ew_202111/dataset.npz"))
    parser.add_argument("--split", type=Path, default=Path("data/v1/step2_ew_202111/split.json"))
    parser.add_argument("--output-dir", type=Path, default=runtime_dir / "process_feature_attribution")
    parser.add_argument("--model", choices=("cnn", "lstm", "tcn"), default="cnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args(argv)
    report = evaluate_process_feature_attribution(
        args.dataset,
        args.split,
        args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        repetitions=args.repetitions,
    )
    print(json.dumps({
        "report": str(args.output_dir / "feature_attribution_report.json"),
        "status": report["status"],
        "selected_features": report["validation_selected_features"],
        "test_split_count": report["provenance"]["effective_split_counts"]["test"],
        "video_fusion_evaluated": report["claims"]["video_fusion_evaluated"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
