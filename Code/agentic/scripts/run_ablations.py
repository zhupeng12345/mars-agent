"""Generate the reproducible ablation matrix without fabricating metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.evaluation.ablation import build_ablation_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None, help="optional JSON with provenance-backed result overrides")
    args = parser.parse_args(argv)
    report = build_ablation_report(args.root, result_overrides=args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(report["comparisons"]), "completed": report["completed_count"], "structure_only": report["structure_only_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
