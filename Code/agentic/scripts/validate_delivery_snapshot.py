"""Fail-closed audit for the evidence bundle used in a local demo/PPT delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _jsonl(path: Path, errors: list[str], *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            errors.append(f"missing:{path.name}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid_json:{path.name}:{line_number}")
            continue
        if not isinstance(value, dict):
            errors.append(f"row_not_object:{path.name}:{line_number}")
            continue
        rows.append(value)
    return rows


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing:{path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid_json:{path.name}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"not_object:{path.name}")
        return {}
    return value


def validate_delivery_snapshot(
    runtime_dir: Path,
    run_ids: Iterable[str],
    *,
    route_report: Path | None = None,
    evaluation_report: Path | None = None,
    require_hitl: bool = False,
    require_rsi: bool = False,
) -> dict[str, Any]:
    """Validate existing trace evidence without creating or inferring records."""
    runtime_dir = Path(runtime_dir)
    selected = sorted({str(item).strip() for item in run_ids if str(item).strip()})
    errors: list[str] = []
    runs = _jsonl(runtime_dir / "runs.jsonl", errors)
    reviews = _jsonl(runtime_dir / "hitl_audit.jsonl", errors, required=require_hitl)
    finished_by_id: dict[str, dict[str, Any]] = {}
    for row in runs:
        run_id = str(row.get("run_id", ""))
        if run_id in selected and row.get("finished_at") and isinstance(row.get("result"), dict):
            finished_by_id[run_id] = row
    missing_runs = sorted(set(selected) - set(finished_by_id))
    if missing_runs:
        errors.append("missing_finished_runs:" + ",".join(missing_runs))

    route = _load_json(route_report, errors) if route_report else {}
    if route_report and (route.get("passed") is not True or int(route.get("fixture_count", 0) or 0) < 4):
        errors.append("route_fixture_regression_incomplete")

    evaluation = _load_json(evaluation_report, errors) if evaluation_report else {}
    if evaluation_report:
        scope = evaluation.get("system", {}).get("runtime_scope", {}) if isinstance(evaluation.get("system"), dict) else {}
        if scope.get("claimable") is not True:
            errors.append("evaluation_scope_not_claimable")
        integrity = evaluation.get("input_integrity", {})
        if integrity and integrity.get("status") != "passed":
            errors.append("evaluation_input_integrity_failed")

    review_by_id: dict[str, list[dict[str, Any]]] = {run_id: [] for run_id in selected}
    for row in reviews:
        run_id = str(row.get("run_id", ""))
        if run_id in review_by_id:
            review_by_id[run_id].append(row)
    if require_hitl:
        for run_id, rows in review_by_id.items():
            actions = {str(row.get("action", "")) for row in rows if row.get("action")}
            if len(actions) != 1 or not actions.intersection({"approve", "amend", "reject", "request_retake"}):
                errors.append(f"hitl_decision_missing_or_ambiguous:{run_id}")

    rsi_rows: dict[str, dict[str, Any]] = {}
    for run_id, row in finished_by_id.items():
        result = row.get("result", {})
        runtime = result.get("rsi_runtime") if isinstance(result, dict) else None
        if isinstance(runtime, dict):
            rsi_rows[run_id] = runtime
        if require_rsi and (not isinstance(runtime, dict) or runtime.get("status") != "active" or not runtime.get("version_id") or not runtime.get("policy_hash")):
            errors.append(f"rsi_runtime_missing_or_inactive:{run_id}")

    return {
        "schema_version": "delivery-snapshot-1",
        "runtime_dir": str(runtime_dir),
        "selected_run_ids": selected,
        "finished_run_count": len(finished_by_id),
        "missing_finished_run_ids": missing_runs,
        "route_fixture_count": route.get("fixture_count") if route_report else None,
        "route_fixture_passed": route.get("passed") if route_report else None,
        "hitl_actions": {run_id: sorted({str(row.get("action")) for row in rows if row.get("action")}) for run_id, rows in review_by_id.items()},
        "rsi_runtime": rsi_rows,
        "errors": errors,
        "status": "passed" if selected and not errors else "failed",
        "claims": {"evidence_audit_only": True, "accuracy_claimable": False, "behavior_gain_claimable": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--route-report", type=Path, default=None)
    parser.add_argument("--evaluation-report", type=Path, default=None)
    parser.add_argument("--require-hitl", action="store_true")
    parser.add_argument("--require-rsi", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_delivery_snapshot(
        args.runtime_dir,
        args.run_id,
        route_report=args.route_report,
        evaluation_report=args.evaluation_report,
        require_hitl=args.require_hitl,
        require_rsi=args.require_rsi,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
