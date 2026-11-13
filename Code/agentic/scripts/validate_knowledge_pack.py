"""Fail-closed validation for the versioned domain knowledge JSONL package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


REQUIRED_TEXT_FIELDS = (
    "rule_id",
    "title",
    "source",
    "source_version",
    "effective_date",
    "risk_level",
    "text",
)
REQUIRED_LIST_FIELDS = ("preconditions", "applicability", "allowed_output")
VALID_RISK_LEVELS = {"low", "medium", "high"}


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty_text(item) for item in value)


def validate_knowledge_pack(path: Path) -> dict[str, Any]:
    """Inspect every raw JSONL row and report errors instead of silently skipping them."""
    path = Path(path)
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    parse_error_count = 0

    if not path.is_file():
        errors.append({"row": None, "errors": ["file_missing"]})
        lines: list[str] = []
    else:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            lines = []
            errors.append({"row": None, "errors": [f"file_read_error:{type(exc).__name__}"]})

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row_errors: list[str] = []
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_error_count += 1
            errors.append({"row": line_number, "errors": [f"invalid_json:{exc.msg}"]})
            continue
        if not isinstance(value, dict):
            errors.append({"row": line_number, "errors": ["row_not_object"]})
            continue

        rows.append(value)
        for field in REQUIRED_TEXT_FIELDS:
            if not _nonempty_text(value.get(field)):
                row_errors.append(f"missing_or_invalid:{field}")
        for field in REQUIRED_LIST_FIELDS:
            if not _string_list(value.get(field)):
                row_errors.append(f"missing_or_invalid_list:{field}")

        rule_id = value.get("rule_id")
        if _nonempty_text(rule_id):
            normalized_id = rule_id.strip()
            previous = seen_ids.get(normalized_id)
            if previous is not None:
                row_errors.append(f"duplicate_rule_id:first_seen_row_{previous}")
            else:
                seen_ids[normalized_id] = line_number

        effective_date = value.get("effective_date")
        if _nonempty_text(effective_date):
            try:
                parsed_date = date.fromisoformat(effective_date)
                if parsed_date.isoformat() != effective_date:
                    row_errors.append("effective_date_not_canonical_iso")
            except ValueError:
                row_errors.append("effective_date_invalid_iso")

        risk_level = value.get("risk_level")
        if _nonempty_text(risk_level) and risk_level not in VALID_RISK_LEVELS:
            row_errors.append("risk_level_invalid")

        if row_errors:
            errors.append({"row": line_number, "rule_id": rule_id if _nonempty_text(rule_id) else None, "errors": row_errors})

    nonempty_line_count = sum(1 for line in lines if line.strip())
    if path.is_file() and not rows and parse_error_count == 0 and not errors:
        errors.append({"row": None, "errors": ["package_empty"]})

    invalid_row_count = len({item["row"] for item in errors if isinstance(item.get("row"), int)})
    valid_count = max(0, nonempty_line_count - invalid_row_count)
    risk_counts = Counter(
        str(row.get("risk_level")) for row in rows if row.get("risk_level") in VALID_RISK_LEVELS
    )
    applicability_counts = Counter(
        tag.strip()
        for row in rows
        for tag in (row.get("applicability") if isinstance(row.get("applicability"), list) else [])
        if _nonempty_text(tag)
    )

    return {
        "schema_version": "knowledge-pack-validation-1",
        "path": str(path),
        "status": "pass" if not errors and rows else "fail",
        "claimable": False,
        "record_count": nonempty_line_count,
        "parsed_record_count": len(rows),
        "nonempty_line_count": nonempty_line_count,
        "valid_record_count": valid_count,
        "invalid_record_count": invalid_row_count,
        "parse_error_count": parse_error_count,
        "risk_level_coverage": dict(sorted(risk_counts.items())),
        "applicability_coverage": dict(sorted(applicability_counts.items())),
        "required_fields": list(REQUIRED_TEXT_FIELDS + REQUIRED_LIST_FIELDS),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "knowledge" / "rules_v1.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate_knowledge_pack(args.input)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
