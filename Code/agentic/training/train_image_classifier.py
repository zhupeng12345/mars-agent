"""Train and evaluate the F-GMVAE-T visual branch on labelled fire-eye images.

The repository contains image-level ``cold/hot/normal`` labels but no reliable
mapping from those images to the two inspection videos.  This module therefore
trains an auditable image classification checkpoint and records its split and
scope explicitly.  It is used as a supervised visual initialization for the
video window pipeline; it is not a video accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agentic.evaluation.visual_eval import classification_metrics
from agentic.models.fgmvae_temporal import FGMVAETemporal


LABELS = ("C", "H", "N")
DIRECTORY_LABELS = {"cold": "C", "hot": "H", "normal": "N"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    group_id: str
    provenance_status: str = "unknown"


def _filename_prefix(path: Path) -> str:
    """Return the non-numeric stem prefix without inventing a source id."""
    prefix = re.sub(r"(?:[_-]?\d+)$", "", path.stem).strip("_- ")
    return prefix or path.stem


def _group_id(path: Path) -> str:
    """Build a conservative unknown-provenance group from the filename."""
    return f"unknown:{path.parent.name.lower()}:{_filename_prefix(path).lower()}"


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path)


def _load_provenance(image_dir: Path, metadata_path: Optional[Path]) -> dict[str, tuple[str, str]]:
    """Load optional per-image source/session groups from JSONL metadata.

    A metadata record is accepted only when it points to a concrete frame and
    exposes a source/session/group key. Aggregate directory-level entries are
    deliberately ignored because they do not establish source independence.
    """
    candidates: list[Path] = []
    if metadata_path is not None:
        candidates.append(Path(metadata_path))
    candidates.extend([
        image_dir / "image_provenance.jsonl",
        image_dir.parent / "image_provenance.jsonl",
        image_dir.parent.parent / "visual_manifest.jsonl",
        image_dir.parent.parent / "evaluation" / "visual_manifest.jsonl",
    ])
    manifest = next((item for item in candidates if item.exists() and item.is_file()), None)
    if manifest is None:
        return {}
    result: dict[str, tuple[str, str]] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        value = row.get("path") or row.get("frame_path") or row.get("image_path")
        source = row.get("session_id") or row.get("source_group") or row.get("split_group") or row.get("video_id") or row.get("source_id")
        # A directory-level source is not a provenance mapping for a frame.
        if not value or not source or row.get("source") == "image_directory":
            continue
        resolved = _resolve_path(value, manifest.parent).resolve()
        result[str(resolved)] = (str(source), "metadata")
    return result


def collect_records(image_dir: Path, metadata_path: Optional[Path] = None) -> list[ImageRecord]:
    provenance = _load_provenance(image_dir, metadata_path)
    records: list[ImageRecord] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = DIRECTORY_LABELS.get(path.parent.name.lower())
        if label is None:
            continue
        resolved = str(path.resolve())
        if resolved in provenance:
            source, status = provenance[resolved]
            group_id = f"source:{source}"
        else:
            group_id, status = _group_id(path), "unknown"
        records.append(ImageRecord(str(path), label, group_id, status))
    if not records:
        raise ValueError(f"no labelled images found under {image_dir}")
    return records


def discover_image_dir(root: Path) -> Path:
    """Find the labelled image directory without relying on console encoding."""
    candidates = [root / "data", root / "Code" / "data"]
    for base in candidates:
        if not base.exists():
            continue
        for directory in sorted(base.iterdir()):
            if not directory.is_dir():
                continue
            nested = list(directory.rglob("cold"))
            if any(path.is_dir() and (path / "cold_1.jpg").exists() for path in nested):
                return nested[0].parent
    raise FileNotFoundError("could not discover fire-eye image directory")


def split_records(records: Iterable[ImageRecord], *, seed: int = 7, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), allow_unsafe_fallback: bool = False) -> dict[str, list[ImageRecord]]:
    """Group-aware, class-stratified train/val/test split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to one")
    records = list(records)
    grouped: dict[str, list[ImageRecord]] = {}
    for item in records:
        grouped.setdefault(item.group_id, []).append(item)
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    mixed_source_groups = any(len({item.label for item in items}) > 1 for items in grouped.values())
    if mixed_source_groups:
        # A source/session group is indivisible even when it contains several
        # directory labels. This preserves source independence at the cost of
        # exact per-class ratios, which is preferable for evaluation integrity.
        groups = sorted(grouped)
        rng.shuffle(groups)
        n = len(groups)
        if n < 3 and not allow_unsafe_fallback:
            raise ValueError(f"insufficient independent source groups: {n}; provide more sessions")
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        for group in groups[:n_train]:
            result["train"].extend(grouped[group])
        for group in groups[n_train:n_train + n_val]:
            result["val"].extend(grouped[group])
        for group in groups[n_train + n_val:]:
            result["test"].extend(grouped[group])
    else:
        groups_by_label: dict[str, list[str]] = {label: [] for label in LABELS}
        for group, items in grouped.items():
            groups_by_label[next(iter({item.label for item in items}))].append(group)
    for label in LABELS:
        if mixed_source_groups:
            continue
        groups = sorted(groups_by_label[label])
        rng.shuffle(groups)
        n = len(groups)
        if n < 3 and not allow_unsafe_fallback:
            raise ValueError(f"insufficient independent groups for label {label}: {n}; provide source/session metadata")
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        assignments = [("train", group) for group in groups[:n_train]]
        assignments += [("val", group) for group in groups[n_train:n_train + n_val]]
        assignments += [("test", group) for group in groups[n_train + n_val:]]
        for split, group in assignments:
            result[split].extend(grouped[group])
    if allow_unsafe_fallback and any(not result[name] for name in result):
        # No provenance groups means a source-independent split is impossible.
        # Use deterministic item stratification only as an explicit fallback;
        # the caller records this risk in the report.
        result = {"train": [], "val": [], "test": []}
        for label in LABELS:
            items = [item for item in records if item.label == label]
            rng.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * ratios[0]))
            n_val = max(1, int(n * ratios[1]))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
            result["train"].extend(items[:n_train])
            result["val"].extend(items[n_train:n_train + n_val])
            result["test"].extend(items[n_train + n_val:])
    for values in result.values():
        values.sort(key=lambda item: item.path)
    if not all(result.values()):
        raise ValueError("empty split produced")
    return result


def runtime_dir() -> Path:
    configured = os.environ.get("MARS_AGENT_RUNTIME_DIR")
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / "runtime_data"


def resolve_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def _save_confusion_plot(metrics: dict[str, Any], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=int)
    if matrix.shape != (len(LABELS), len(LABELS)):
        return
    figure, axis = plt.subplots(figsize=(5.4, 4.6), dpi=160)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(LABELS)), LABELS)
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted image class")
    axis.set_ylabel("Directory label")
    axis.set_title("Fire-eye image split test confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


class FireEyeImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], image_size: int = 96) -> None:
        self.records = records
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.records[index]
        # cv2.imread cannot reliably open Unicode paths on Windows; the
        # byte-buffer route keeps the dataset usable with the Chinese source
        # directory name.
        buffer = np.fromfile(item.path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode image: {item.path}")
        image = cv2.cvtColor(cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div(255.0)
        return tensor, LABELS.index(item.label)


def _run_epoch(model: FGMVAETemporal, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], device: torch.device, class_weights: torch.Tensor) -> tuple[float, list[str], list[dict[str, float]]]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    truth: list[str] = []
    probabilities: list[dict[str, float]] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        sequence = images.unsqueeze(1)
        outputs = model(sequence, sample=training)
        cls_loss = torch.nn.functional.cross_entropy(outputs["logits"], labels, weight=class_weights)
        regularizer = 0.01 * outputs["kl"]
        loss = cls_loss + regularizer
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        probs = outputs["class_probs"].detach().cpu().numpy()
        truth.extend(LABELS[int(value)] for value in labels.detach().cpu().tolist())
        probabilities.extend({label: float(row[i]) for i, label in enumerate(LABELS)} for row in probs)
    return float(np.mean(losses)) if losses else float("nan"), truth, probabilities


def train_image_classifier(
    image_dir: Path,
    output_dir: Path,
    *,
    seed: int = 7,
    epochs: int = 2,
    batch_size: int = 32,
    image_size: int = 96,
    learning_rate: float = 2e-3,
    max_images: Optional[int] = None,
    device: str = "auto",
    metadata_path: Optional[Path] = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = collect_records(image_dir, metadata_path=metadata_path)
    if max_images is not None:
        limit = int(max_images)
        if limit < len(LABELS):
            raise ValueError("max_images must be at least the number of classes")
        rng = random.Random(seed)
        by_label = {label: [item for item in records if item.label == label] for label in LABELS}
        selected: list[ImageRecord] = []
        base, remainder = divmod(limit, len(LABELS))
        for index, label in enumerate(LABELS):
            count = min(len(by_label[label]), base + (1 if index < remainder else 0))
            selected.extend(rng.sample(by_label[label], count))
        records = sorted(selected, key=lambda item: item.path)
    provenance_known = bool(records) and all(item.provenance_status == "metadata" for item in records)
    splits = split_records(records, seed=seed, allow_unsafe_fallback=not provenance_known)
    split_groups = {name: {item.group_id for item in values} for name, values in splits.items()}
    group_overlap = {
        "train_val": len(split_groups["train"] & split_groups["val"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "val_test": len(split_groups["val"] & split_groups["test"]),
    }
    unsafe_item_fallback = any(value > 0 for value in group_overlap.values())
    selected_device = resolve_device(device)
    model_config = {"num_classes": 3, "latent_dim": 16, "spatial_dim": 64, "temporal_dim": 64, "mixture_components": 3, "temporal_type": "gru", "expert_dim": 0, "model_profile": "fgmvae-t-image-supervised-v1"}
    model = FGMVAETemporal(**model_config).to(selected_device)
    counts = np.bincount([LABELS.index(item.label) for item in splits["train"]], minlength=3).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=selected_device)
    class_weights = class_weights / class_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loaders = {name: DataLoader(FireEyeImageDataset(values, image_size), batch_size=batch_size, shuffle=name == "train", num_workers=0) for name, values in splits.items()}
    history: list[dict[str, Any]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_macro_f1 = -1.0
    for epoch in range(max(1, int(epochs))):
        train_loss, _, _ = _run_epoch(model, loaders["train"], optimizer, selected_device, class_weights)
        val_loss, val_truth, val_probs = _run_epoch(model, loaders["val"], None, selected_device, class_weights)
        val_metrics = classification_metrics(val_truth, val_probs)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_metrics": val_metrics})
        score = float(val_metrics.get("macro_f1") or 0.0)
        if score > best_macro_f1:
            best_macro_f1 = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_truth, test_probs = _run_epoch(model, loaders["test"], None, selected_device, class_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "fgmvae_t_image_supervised.pth"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "seed": seed, "image_size": image_size, "labels": LABELS, "scope": "image_label_supervised; video_windows_unlabeled"}, checkpoint)
    test_metrics = classification_metrics(test_truth, test_probs)
    confusion_plot = output_dir / "image_test_confusion_matrix.png"
    _save_confusion_plot(test_metrics, confusion_plot)
    report = {
        "schema_version": "visual-image-training-1", "status": "completed", "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_dir": str(image_dir), "dataset_count": len(records), "class_counts": {label: sum(item.label == label for item in records) for label in LABELS},
        "split_counts": {name: len(values) for name, values in splits.items()}, "split_group_counts": {name: len({item.group_id for item in values}) for name, values in splits.items()},
        "seed": seed, "epochs": epochs, "best_epoch": best_epoch, "selection_metric": "validation_macro_f1", "batch_size": batch_size, "image_size": image_size, "model_config": model_config,
        "device_request": device, "device": str(selected_device),
        "provenance": {"metadata_path": str(metadata_path) if metadata_path else None, "known_for_all_records": provenance_known, "leakage_risk": "low" if provenance_known and not unsafe_item_fallback else "high", "split_scope": "source_or_session" if provenance_known else "filename_prefix_with_item_fallback", "unsafe_item_fallback": unsafe_item_fallback, "split_group_overlap": group_overlap},
        "history": history, "test_loss": test_loss, "test_metrics": test_metrics, "confusion_plot": str(confusion_plot),
        "scope": "image_level_supervised_checkpoint; video_windows_remain_unlabeled; no_video_superheat_measurement",
        "checkpoint": str(checkpoint), "source_paths_sha256": hashlib.sha256("\n".join(item.path for item in records).encode()).hexdigest(),
    }
    (output_dir / "image_training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "image_split_manifest.jsonl").write_text("".join(json.dumps({"path": item.path, "label": item.label, "group_id": item.group_id, "split": split}, ensure_ascii=False) + "\n" for split, values in splits.items() for item in values), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=None, help="Labelled image root; omitted means discover it under data/")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metadata-path", type=Path, default=None)
    args = parser.parse_args(argv)
    image_dir = args.image_dir or discover_image_dir(Path.cwd())
    output_dir = args.output_dir or (runtime_dir() / "visual_training")
    report = train_image_classifier(image_dir, output_dir, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size, max_images=args.max_images, device=args.device, metadata_path=args.metadata_path)
    print(json.dumps({"status": report["status"], "checkpoint": report["checkpoint"], "test_metrics": report["test_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
