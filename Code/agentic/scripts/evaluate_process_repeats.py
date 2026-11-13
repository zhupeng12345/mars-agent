"""Run repeated seeded train-only-fitted process baseline experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.evaluation.process_repeats import run_repeated_process_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/v1/step2_ew_202111/dataset.npz"))
    parser.add_argument("--split", type=Path, default=Path("data/v1/step2_ew_202111/split.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--model", choices=("cnn", "lstm", "tcn"), default="cnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    report = run_repeated_process_evaluation(
        args.dataset,
        args.split,
        args.output_dir,
        seeds=args.seeds,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({
        "report": report["artifacts"]["json"],
        "csv": report["artifacts"]["csv"],
        "run_count": report["run_count"],
        "metrics": {key: {"mean": value["mean"], "std_population": value["std_population"]} for key, value in report["metrics"].items()},
        "claims": report["claims"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
