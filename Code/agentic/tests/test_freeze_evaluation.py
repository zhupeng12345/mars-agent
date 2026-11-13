from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic.scripts.freeze_evaluation import MINIMUM_RECORDS, REQUIRED_MANIFESTS, freeze


class FreezeEvaluationTests(unittest.TestCase):
    def test_missing_manifests_are_not_created_or_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = freeze(root)
            evaluation = root / "data" / "evaluation"
            self.assertFalse(report["frozen"])
            self.assertEqual(set(report["missing_or_invalid_manifests"]), set(REQUIRED_MANIFESTS))
            self.assertFalse((evaluation / "knowledge_queries.jsonl").exists())
            self.assertFalse((evaluation / "hitl_cases.jsonl").exists())

    def test_template_rows_and_demo_fixtures_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "knowledge_queries.jsonl").write_text(
                json.dumps({"query_id": "template", "query": "query", "expected_source_ids": [], "status": "template"}) + "\n",
                encoding="utf-8",
            )
            (evaluation / "hitl_cases.jsonl").write_text(
                json.dumps({"case_id": "demo", "expected_action": "approve", "reviewer_id": "fixture", "annotation_status": "reviewed", "fixture_type": "demo_fixture"}) + "\n",
                encoding="utf-8",
            )
            report = freeze(Path(directory))
        self.assertFalse(report["frozen"])
        self.assertIn("invalid_rows:1", report["files"]["knowledge_queries.jsonl"]["errors"])
        self.assertEqual(
            report["files"]["knowledge_queries.jsonl"]["invalid_error_counts"],
            {"annotation_not_expert_reviewed": 1},
        )
        self.assertIn("insufficient_records:1<10", report["files"]["hitl_cases.jsonl"]["errors"])

    def test_complete_reviewed_manifests_freeze_with_hashes(self):
        row_builders = {
            "visual_manifest.jsonl": lambda i: {"window_id": f"w{i}", "source_group": f"video-{i}", "human_label": "N", "split": "test", "annotation_status": "expert_reviewed"},
            "process_manifest.jsonl": lambda i: {"sample_idx": i, "anchor_ts": f"2026-01-{(i % 28) + 1:02d}T00:00", "y_true": i % 2, "split": "test", "annotation_status": "expert_reviewed"},
            "route_manifest.jsonl": lambda i: {"case_id": f"route-{i}", "features": {"visual_confidence": 0.8}, "expert_route": "full_loop", "annotation_id": f"ann-{i}", "annotation_status": "expert_reviewed"},
            "knowledge_queries.jsonl": lambda i: {"query_id": f"k-{i}", "query": f"query {i}", "expected_source_ids": ["rule-1"], "annotation_status": "expert_reviewed"},
            "case_queries.jsonl": lambda i: {"query_id": f"c-{i}", "query": f"query {i}", "expected_outcome": "match", "annotation_status": "expert_reviewed"},
            "hitl_cases.jsonl": lambda i: {"case_id": f"h-{i}", "expected_action": "approve", "reviewer_id": f"reviewer-{i}", "annotation_status": "expert_reviewed"},
        }
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            for name, builder in row_builders.items():
                rows = [builder(index) for index in range(MINIMUM_RECORDS[name])]
                (evaluation / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = freeze(Path(directory))
        self.assertTrue(report["frozen"])
        self.assertTrue(all(item["status"] == "valid" and item["sha256"] for item in report["files"].values()))

    def test_duplicate_manifest_ids_do_not_count_toward_freeze_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            rows = [
                {
                    "case_id": "same-case",
                    "features": {"visual_confidence": 0.8},
                    "expert_route": "full_loop",
                    "annotation_id": f"review-{index}",
                    "annotation_status": "expert_reviewed",
                }
                for index in range(MINIMUM_RECORDS["route_manifest.jsonl"])
            ]
            (evaluation / "route_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = freeze(Path(directory))
        route = report["files"]["route_manifest.jsonl"]
        self.assertEqual(route["records"], MINIMUM_RECORDS["route_manifest.jsonl"])
        self.assertEqual(route["unique_identity_count"], 1)
        self.assertEqual(route["duplicate_identity_count"], MINIMUM_RECORDS["route_manifest.jsonl"] - 1)
        self.assertEqual(route["valid_records"], 1)
        self.assertIn("duplicate_identities:19", route["errors"])
        self.assertEqual(route["invalid_error_counts"], {"duplicate_identity:case_id": 19})
        self.assertEqual(route["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
