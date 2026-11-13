import json
import tempfile
import unittest
from pathlib import Path

from agentic.scripts.validate_knowledge_pack import validate_knowledge_pack


def _rule(**overrides):
    value = {
        "rule_id": "rule-1",
        "title": "领域规则",
        "source": "manual",
        "source_version": "v1",
        "effective_date": "2026-01-01",
        "preconditions": ["quality_pass"],
        "applicability": ["visual"],
        "risk_level": "medium",
        "allowed_output": ["inspection_item"],
        "text": "记录可追溯的检查证据。",
    }
    value.update(overrides)
    return value


class KnowledgePackValidationTests(unittest.TestCase):
    def test_existing_domain_pack_passes_and_reports_coverage(self):
        path = Path(__file__).resolve().parents[1] / "knowledge" / "rules_v1.jsonl"
        report = validate_knowledge_pack(path)

        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["claimable"])
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["valid_record_count"], 3)
        self.assertEqual(report["risk_level_coverage"], {"high": 2, "medium": 1})
        self.assertIn("visual", report["applicability_coverage"])

    def test_duplicate_ids_and_invalid_rows_fail_closed_with_line_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps(_rule(), ensure_ascii=False),
                    "{invalid",
                    json.dumps(_rule(effective_date="2026-02-30", risk_level="urgent", preconditions=[]), ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )

            report = validate_knowledge_pack(path)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["parsed_record_count"], 2)
        self.assertEqual(report["parse_error_count"], 1)
        by_line = {item["row"]: item["errors"] for item in report["errors"] if item["row"] is not None}
        self.assertIn("invalid_json:Expecting property name enclosed in double quotes", by_line[2])
        self.assertIn("duplicate_rule_id:first_seen_row_1", by_line[3])
        self.assertIn("effective_date_invalid_iso", by_line[3])
        self.assertIn("risk_level_invalid", by_line[3])
        self.assertIn("missing_or_invalid_list:preconditions", by_line[3])

    def test_required_fields_and_list_members_must_be_present_and_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.jsonl"
            path.write_text(
                json.dumps(_rule(source_version=" ", allowed_output=["", "ok"], applicability=[]), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report = validate_knowledge_pack(path)

        self.assertEqual(report["status"], "fail")
        errors = report["errors"][0]["errors"]
        self.assertIn("missing_or_invalid:source_version", errors)
        self.assertIn("missing_or_invalid_list:applicability", errors)
        self.assertIn("missing_or_invalid_list:allowed_output", errors)

    def test_missing_or_empty_package_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = validate_knowledge_pack(Path(directory) / "missing.jsonl")
            empty_path = Path(directory) / "empty.jsonl"
            empty_path.write_text("\n  \n", encoding="utf-8")
            empty = validate_knowledge_pack(empty_path)

        self.assertEqual(missing["status"], "fail")
        self.assertEqual(missing["errors"][0]["errors"], ["file_missing"])
        self.assertEqual(empty["status"], "fail")
        self.assertEqual(empty["errors"][0]["errors"], ["package_empty"])


if __name__ == "__main__":
    unittest.main()
