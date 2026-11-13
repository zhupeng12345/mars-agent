"""Train/smoke-test the F-GMVAE-T visual branch without fabricating labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from agentic.models.fgmvae_temporal import FGMVAETemporal
from agentic.models.visual_types import ROISequence


def run_smoke_training(*, seed: int = 7, batch_size: int = 2, steps: int = 2) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = FGMVAETemporal(model_profile="fgmvae-t-cpu-smoke-v1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(max(1, int(steps))):
        frames = torch.rand(batch_size, 8, 3, 64, 64)
        sequence = ROISequence.from_frame_tensor(frames, source_ids=[f"synthetic-{index}" for index in range(batch_size)])
        labels = torch.tensor([index % 3 for index in range(batch_size)], dtype=torch.long)
        output = model(sequence, sample=True)
        loss_terms = model.loss(output, labels=labels, lambda_kl=0.01, lambda_temporal=0.05)
        loss = loss_terms["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite F-GMVAE-T smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"schema_version": "visual-training-smoke-1", "status": "smoke_passed", "seed": seed, "steps": len(losses), "losses": losses, "trained_on_expert_video_labels": False, "model_profile": model.model_profile}


def train_manifest(manifest: Path | str, *, output: Optional[Path | str] = None, seed: int = 7) -> dict[str, Any]:
    path = Path(manifest)
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labeled = [row for row in rows if row.get("human_label") in {"C", "H", "N"}]
    smoke = run_smoke_training(seed=seed)
    result = {**smoke, "manifest": str(path), "manifest_records": len(rows), "human_labeled_windows": len(labeled), "status": "blocked_no_human_video_labels" if not labeled else "ready_for_gpu_training"}
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/visual_manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    result = train_manifest(args.manifest, output=args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "status": result["status"], "loss_last": result["losses"][-1]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
