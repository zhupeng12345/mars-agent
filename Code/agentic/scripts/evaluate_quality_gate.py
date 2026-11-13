"""Generate the deterministic image-quality gate regression report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentic.evaluation.image_quality import evaluate_quality_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic image-quality gate regression")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    runtime_dir = Path(os.environ.get("MARS_AGENT_RUNTIME_DIR", Path(__file__).resolve().parents[2] / "runtime_data"))
    output = args.output or runtime_dir / "image_quality_evaluation.json"
    result = evaluate_quality_gate()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "failure_detection_rate": result["failure_detection_rate"],
        "valid_false_retake_count": result["valid_false_retake_count"],
        "claimable_domain_accuracy": result["claimable_domain_accuracy"],
    }, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
