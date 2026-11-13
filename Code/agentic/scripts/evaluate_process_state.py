"""CLI wrapper for the frozen process-state baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from agentic.evaluation.process_state import evaluate_baselines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate checked-in process-event baselines")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "v1")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_baselines(args.data_root, split=args.split, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": sorted(result["models"]), "split": args.split}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
