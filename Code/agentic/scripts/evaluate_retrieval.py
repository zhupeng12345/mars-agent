"""Run local knowledge and case retrieval behavior fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.evaluation.retrieval import evaluate_retrieval, write_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    report = evaluate_retrieval(args.root, top_k=args.top_k)
    output = args.output or (args.root / "runtime_data" / "retrieval_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = write_artifacts(report, output)
    passed = report["complete_fixture_coverage"] and all(
        report[name]["fixture_pass_rate"] == 1.0 for name in ("knowledge", "cases")
    )
    print(json.dumps({
        "output": str(output),
        "knowledge_queries": report["knowledge"]["query_count"],
        "knowledge_fixture_pass_rate": report["knowledge"]["fixture_pass_rate"],
        "case_queries": report["cases"]["query_count"],
        "case_fixture_pass_rate": report["cases"]["fixture_pass_rate"],
        "claimable_domain_accuracy": False,
        "artifacts": artifacts,
    }, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
