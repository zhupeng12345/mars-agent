"""Fit an auditable route confidence calibrator from route-manifest records.

This command calibrates confidence; it does not train the Laya router model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.platform.jev.calibration import fit_route_calibration


def _load_route_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("data/evaluation/route_manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=10)
    args = parser.parse_args(argv)
    rows = []
    try:
        if args.min_rows < 1:
            raise ValueError("--min-rows must be positive")
        rows = _load_route_rows(args.rows)
        result = fit_route_calibration(rows, min_rows=args.min_rows)
        result.update({
            "input": str(args.rows),
            "row_count": len(rows),
            "status": "calibrated",
            "model_training": "not_performed",
        })
        exit_code = 0
    except (OSError, ValueError) as exc:
        result = {
            "schema_version": "jev-calibration-2",
            "input": str(args.rows),
            "row_count": len(rows),
            "status": "rejected",
            "model_training": "not_performed",
            "errors": [str(exc)],
        }
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"output": str(args.output), "row_count": len(rows), "status": result["status"]}
    if exit_code:
        summary["errors"] = result["errors"]
    else:
        summary.update({"temperature": result["temperature"], "dataset_sha256": result["dataset_sha256"]})
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
