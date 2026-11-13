"""Train/evaluate a reproducible independent 14-variable process baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from agentic.evaluation.process_state import evaluate_process_state
from agentic.agents.process_state import DATASET_FEATURE_COLUMNS
from agentic.process.baselines import build_process_baseline
from agentic.process.data import ProcessNormalizer, purged_date_splits


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_hash(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def train_process(
    dataset: Path,
    split: Path,
    output_dir: Path,
    *,
    model_name: str = "cnn",
    epochs: int = 1,
    batch_size: int = 128,
    seed: int = 7,
    device: str = "cpu",
    feature_names: Sequence[str] | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    requested = str(device).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    actual = "cuda" if requested == "cuda" else "cpu"
    with np.load(dataset, allow_pickle=True) as payload:
        x = np.asarray(payload["X_seq"], dtype=np.float32)
        y_raw = np.asarray(payload["y"])
        timestamps = np.asarray(payload["ts_anchor"])
        days = np.asarray(payload["day_anchor"])
        history_steps = int(payload["L"])
        future_steps = int(payload["H"])
        sample_minutes = int(payload["SAMPLE_MINUTES"])
        feature_columns = [str(item) for item in payload["feature_cols"].tolist()]
    if y_raw.ndim != 1 or x.ndim != 3 or x.shape[0] != len(y_raw) or x.shape[2] != 14:
        raise ValueError(f"X_seq/y must have shape [N,T,14] and matching rows; got {x.shape}, {y_raw.shape}")
    if timestamps.ndim != 1 or days.ndim != 1 or x.shape[1] != history_steps or len(timestamps) != len(y_raw) or len(days) != len(y_raw):
        raise ValueError("dataset temporal metadata does not match X_seq/y")
    if len(feature_columns) != x.shape[2] or len(set(feature_columns)) != len(feature_columns) or set(feature_columns) != set(DATASET_FEATURE_COLUMNS):
        raise ValueError("feature_cols length does not match X_seq feature dimension")
    selected_features = tuple(feature_names if feature_names is not None else feature_columns)
    if not selected_features or len(selected_features) != len(set(selected_features)) or set(selected_features) - set(feature_columns):
        raise ValueError("feature_names must be a nonempty, unique subset of dataset feature_cols")
    feature_indices = [feature_columns.index(name) for name in selected_features]
    x = x[:, :, feature_indices]
    if not np.isfinite(y_raw.astype(np.float64)).all() or not np.isin(y_raw, (0, 1)).all():
        raise ValueError("process labels must be finite binary values in {0,1}")
    y = y_raw.astype(np.float32)
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    effective_splits, split_report = purged_date_splits(
        split_payload,
        timestamps,
        days,
        history_steps=history_steps,
        future_steps=future_steps,
        sample_minutes=sample_minutes,
    )
    train_idx = effective_splits["train"]
    val_idx = effective_splits["val"]
    test_idx = effective_splits["test"]
    normalizer = ProcessNormalizer.fit(x[train_idx])
    x = normalizer.transform(x)
    hidden_dim = 32
    model = build_process_baseline(model_name, input_dim=len(selected_features), hidden_dim=hidden_dim).to(actual)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    model.train()
    for _ in range(max(1, int(epochs))):
        order = np.random.default_rng(seed).permutation(train_idx)
        epoch_loss = 0.0
        count = 0
        for start in range(0, len(order), max(1, int(batch_size))):
            batch = order[start : start + max(1, int(batch_size))]
            xb = torch.from_numpy(x[batch]).to(actual)
            yb = torch.from_numpy(y[batch]).to(actual)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(epoch_loss / max(1, count))
    model.eval()
    metrics = None
    prediction_path = None
    if evaluate:
        with torch.no_grad():
            test_prob = torch.sigmoid(model(torch.from_numpy(x[test_idx]).to(actual))).cpu().numpy()
        y_test = y[test_idx].astype(int).tolist()
        probabilities = test_prob.astype(float).tolist()
        test_timestamps = [
            np.datetime_as_string(item, unit="s") if np.issubdtype(timestamps.dtype, np.datetime64) else str(item)
            for item in timestamps[test_idx]
        ]
        metrics = evaluate_process_state(
            y_test,
            probabilities,
            threshold=0.5,
            timestamps=test_timestamps,
            model_name=model_name,
        )
        metrics["n_test"] = len(y_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"process_{model_name}.pth"
    if evaluate:
        prediction_path = output_dir / f"process_{model_name}_test_predictions.csv"
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_idx", "y_true", "y_prob", "anchor_ts"))
            writer.writeheader()
            for sample_index, label, probability, timestamp in zip(test_idx, y_test, probabilities, test_timestamps):
                writer.writerow({
                    "sample_idx": int(sample_index),
                    "y_true": int(label),
                    "y_prob": float(probability),
                    "anchor_ts": timestamp,
                })
    split_index_hashes = {name: _index_hash(indices) for name, indices in effective_splits.items()}
    torch.save({
        "schema_version": "process-checkpoint-2",
        "state_dict": model.state_dict(),
        "model": model_name,
        "model_config": {"input_dim": len(selected_features), "hidden_dim": hidden_dim},
        "model_profile": f"process-{model_name}-{len(selected_features)}f-seed-{seed}",
        "temporal_contract": {
            "history_steps": history_steps,
            "future_steps": future_steps,
            "sample_minutes": sample_minutes,
        },
        "threshold_from_val": 0.5,
        "seed": seed,
        "dataset_sha256": _hash(dataset),
        "split_sha256": _hash(split),
        "normalization": normalizer.to_dict(),
        "effective_split_indices": {name: indices.tolist() for name, indices in effective_splits.items()},
        "effective_split_index_sha256": split_index_hashes,
        "split_report": split_report,
        "feature_columns": list(selected_features),
    }, checkpoint)
    report = {
        "schema_version": "process-training-report-1",
        "status": "experimental_baseline",
        "device": actual,
        "model": model_name,
        "epochs": max(1, int(epochs)),
        "seed": seed,
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "test_count": len(test_idx),
        "evaluation_split": "test" if evaluate else None,
        "split": {
            **split_report,
            "effective_split_index_sha256": split_index_hashes,
        },
        "data_contract": {
            "sample_count": int(len(y)),
            "sequence_shape": list(x.shape),
            "feature_columns": list(selected_features),
            "dataset_feature_columns": feature_columns,
            "history_steps": history_steps,
            "future_steps": future_steps,
            "sample_minutes": sample_minutes,
        },
        "normalization": normalizer.to_dict(),
        "losses": losses,
        "metrics": metrics,
        "test_predictions": str(prediction_path) if prediction_path else None,
        "checkpoint": str(checkpoint),
        "dataset_sha256": _hash(dataset),
        "split_sha256": _hash(split),
        "video_mapping": {"available": False, "alignment_quality": "unknown"},
        "claims": {"video_fusion_evaluated": False, "real_superheat_measurement": False},
    }
    report_path = output_dir / f"process_{model_name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/v1/step2_ew_202111/dataset.npz"))
    parser.add_argument("--split", type=Path, default=Path("data/v1/step2_ew_202111/split.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", "lstm", "tcn"), default="cnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    report = train_process(args.dataset, args.split, args.output_dir, model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device)
    print(json.dumps({"report": str(args.output_dir / f"process_{args.model}_report.json"), "model": args.model, "roc_auc": report["metrics"]["roc_auc"], "status": report["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
