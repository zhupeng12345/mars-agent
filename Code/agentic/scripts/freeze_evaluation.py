"""Validate research evaluation manifests before freezing their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MINIMUM_RECORDS = {
    "visual_manifest.jsonl": 100,
    "process_manifest.jsonl": 100,
    "route_manifest.jsonl": 20,
    "knowledge_queries.jsonl": 10,
    "case_queries.jsonl": 10,
    "hitl_cases.jsonl": 10,
}
REQUIRED_MANIFESTS = tuple(MINIMUM_RECORDS)
VALID_ROUTES = {"simple_loop", "full_loop", "retake", "human"}
VALID_HITL_ACTIONS = {"approve", "amend", "reject", "request_retake"}
IDENTITY_FIELDS = {
    "visual_manifest.jsonl": ("window_id",),
    "process_manifest.jsonl": ("sample_id", "sample_idx", "window_id"),
    "route_manifest.jsonl": ("case_id",),
    "knowledge_queries.jsonl": ("query_id",),
    "case_queries.jsonl": ("query_id",),
    "hitl_cases.jsonl": ("case_id",),
}


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, ["file_missing"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid_json_line:{line_number}")
            continue
        if not isinstance(value, dict):
            errors.append(f"row_not_object:{line_number}")
            continue
        rows.append(value)
    return rows, errors


def _reviewed(row: dict[str, Any]) -> bool:
    status = str(row.get("annotation_status", row.get("label_status", ""))).lower()
    return status in {"expert_reviewed", "human_reviewed", "reviewed", "verified"} and row.get("fixture_type") != "demo_fixture" and not row.get("is_template", False)


def _evaluation_split(value: Any) -> bool:
    return isinstance(value, str) and value in {"test", "frozen", "sealed"}


def _validate_row(name: str, row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _reviewed(row):
        errors.append("annotation_not_expert_reviewed")
    if name == "visual_manifest.jsonl":
        if row.get("human_label") not in {"C", "H", "N"}:
            errors.append("human_label_missing_or_invalid")
        if not row.get("window_id") or not (row.get("source_group") or row.get("video_id") or row.get("source_id")):
            errors.append("visual_provenance_missing")
        if not _evaluation_split(row.get("split")):
            errors.append("visual_eval_split_missing")
    elif name == "process_manifest.jsonl":
        if not any(key in row for key in ("y_true", "label", "target")):
            errors.append("process_label_missing")
        if not any(key in row for key in ("sample_idx", "sample_id", "window_id")):
            errors.append("process_sample_id_missing")
        if not any(key in row for key in ("anchor_ts", "timestamp", "ts_anchor")):
            errors.append("process_timestamp_missing")
        if not _evaluation_split(row.get("split")):
            errors.append("process_eval_split_missing")
    elif name == "route_manifest.jsonl":
        if row.get("expert_route") not in VALID_ROUTES:
            errors.append("expert_route_missing_or_invalid")
        if not isinstance(row.get("features"), dict) or not row.get("case_id"):
            errors.append("route_features_or_case_id_missing")
        if not row.get("annotation_id") and not row.get("reviewer_id"):
            errors.append("route_annotation_provenance_missing")
    elif name in {"knowledge_queries.jsonl", "case_queries.jsonl"}:
        if not row.get("query_id") or not str(row.get("query", "")).strip():
            errors.append("query_id_or_text_missing")
        if name == "knowledge_queries.jsonl" and not isinstance(row.get("expected_source_ids"), list):
            errors.append("expected_source_ids_missing")
        if name == "case_queries.jsonl" and not any(key in row for key in ("expected_case_ids", "expected_outcome")):
            errors.append("expected_case_ids_or_outcome_missing")
    elif name == "hitl_cases.jsonl":
        if row.get("expected_action") not in VALID_HITL_ACTIONS:
            errors.append("expected_action_missing_or_invalid")
        if not row.get("case_id") or not (row.get("reviewer_id") or row.get("annotation_id")):
            errors.append("hitl_case_provenance_missing")
    return errors


def _identity(name: str, row: dict[str, Any]) -> tuple[str, str] | None:
    for field in IDENTITY_FIELDS[name]:
        value = row.get(field)
        if value is not None and str(value).strip():
            return field, str(value).strip()
    return None


def _file_record(path: Path, name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "required_name": name,
        "minimum_records": MINIMUM_RECORDS[name],
        "bytes": 0,
        "sha256": None,
        "records": 0,
        "valid_records": 0,
        "status": "missing",
        "errors": [],
    }
    if not path.exists():
        row["errors"] = ["file_missing"]
        return row
    raw = path.read_bytes()
    row["bytes"] = len(raw)
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    rows, parse_errors = _read_jsonl(path)
    row["records"] = len(rows)
    invalid_rows = []
    invalid_error_counts: dict[str, int] = {}
    seen_identities: set[tuple[str, str]] = set()
    identity_duplicates = 0
    for index, item in enumerate(rows, start=1):
        issues = _validate_row(name, item)
        identity = _identity(name, item)
        if identity is not None:
            if identity in seen_identities:
                identity_duplicates += 1
                issues.append(f"duplicate_identity:{identity[0]}")
            seen_identities.add(identity)
        if issues:
            invalid_rows.append({"row": index, "errors": issues})
            for issue in issues:
                invalid_error_counts[issue] = invalid_error_counts.get(issue, 0) + 1
    row["valid_records"] = len(rows) - len(invalid_rows)
    row["unique_identity_count"] = len(seen_identities)
    row["duplicate_identity_count"] = identity_duplicates
    row["invalid_error_counts"] = dict(sorted(invalid_error_counts.items()))
    errors = list(parse_errors)
    if len(rows) < MINIMUM_RECORDS[name]:
        errors.append(f"insufficient_records:{len(rows)}<{MINIMUM_RECORDS[name]}")
    if invalid_rows:
        errors.append(f"invalid_rows:{len(invalid_rows)}")
    if identity_duplicates:
        errors.append(f"duplicate_identities:{identity_duplicates}")
    row["invalid_row_examples"] = invalid_rows[:20]
    row["errors"] = errors
    row["status"] = "valid" if not errors and row["valid_records"] >= MINIMUM_RECORDS[name] else "invalid"
    return row


def freeze(root: Path) -> dict[str, Any]:
    evaluation = Path(root) / "data" / "evaluation"
    files = {name: _file_record(evaluation / name, name) for name in REQUIRED_MANIFESTS}
    invalid = [name for name, record in files.items() if record["status"] != "valid"]
    return {
        "schema_version": "evaluation-freeze-3",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "missing_or_invalid_manifests": invalid,
        "frozen": not invalid,
        "case_memory_excluded": True,
        "rsi_excluded": True,
        "policy": {
            "reviewed_labels_required": True,
            "demo_fixtures_excluded": True,
            "minimum_records": MINIMUM_RECORDS,
            "freeze_is_content_validation_not_file_presence": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = freeze(args.root)
    output = args.output or (args.root / "runtime_data" / "evaluation_freeze.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "frozen": payload["frozen"],
        "invalid_manifests": payload["missing_or_invalid_manifests"],
    }, ensure_ascii=False))
    return 0 if payload["frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
