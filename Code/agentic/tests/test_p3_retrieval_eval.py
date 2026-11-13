import csv
import json
from pathlib import Path
import tempfile
import unittest

from agentic.evaluation.retrieval import citation_integrity_issues, evaluate_retrieval, write_artifacts
from agentic.platform.knowledge import VersionedKnowledgeBase


class RetrievalFixtureEvaluationTests(unittest.TestCase):
    def test_ten_knowledge_and_case_behavior_fixtures_replay_without_accuracy_claims(self):
        root = Path(__file__).resolve().parents[2]
        report = evaluate_retrieval(root)
        self.assertTrue(report["complete_fixture_coverage"])
        self.assertFalse(report["claimable_domain_accuracy"])
        self.assertEqual(report["evaluation_kind"], "demo_fixture_behavioral_regression")
        self.assertEqual(report["knowledge"]["query_count"], 10)
        self.assertEqual(report["knowledge"]["fixture_pass_rate"], 1.0)
        citations = report["knowledge"]["citation_integrity"]
        self.assertEqual(citations["status"], "evaluated")
        self.assertEqual(citations["pass_rate"], 1.0)
        self.assertFalse(citations["claimable"])
        self.assertEqual(citations["semantic_support_status"], "not_evaluated_without_human_entailment_labels")
        self.assertEqual(report["cases"]["query_count"], 10)
        self.assertEqual(report["cases"]["fixture_pass_rate"], 1.0)
        self.assertEqual(report["cases"]["expected_no_match_correct_rate"], 1.0)

    def test_citation_integrity_rejects_unresolved_or_tampered_provenance(self):
        root = Path(__file__).resolve().parents[2]
        knowledge = VersionedKnowledgeBase(root / "knowledge" / "rules_v1.jsonl")
        hit = knowledge.search_bm25("过热度初晶温度实测依据", applicability=["process", "superheat"])[0]
        self.assertEqual(citation_integrity_issues(hit, knowledge), [])

        tampered = dict(hit, source_version="unverified", locator="missing-locator")
        issues = citation_integrity_issues(tampered, knowledge)
        self.assertIn("source_version_mismatch_or_missing", issues)
        self.assertIn("locator_mismatch_or_missing", issues)
        self.assertEqual(citation_integrity_issues(dict(hit, source_id="unknown-rule"), knowledge), ["source_unresolved"])

    def test_csv_artifact_round_trips_nested_metrics_as_three_columns(self):
        root = Path(__file__).resolve().parents[2]
        report = evaluate_retrieval(root)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_artifacts(report, Path(directory) / "retrieval.json")
            with Path(artifacts["csv"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(rows[0], ["area", "metric", "value"])
        self.assertTrue(all(len(row) == 3 for row in rows))
        metrics = {(area, metric): value for area, metric, value in rows[1:]}
        citation_integrity = json.loads(metrics[("knowledge", "citation_integrity")])
        self.assertEqual(citation_integrity["pass_rate"], 1.0)
        self.assertEqual(len(citation_integrity["rows"]), 10)


if __name__ == "__main__":
    unittest.main()
