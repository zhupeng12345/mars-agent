from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic.scripts.validate_delivery_snapshot import validate_delivery_snapshot


class DeliverySnapshotTests(unittest.TestCase):
    def test_snapshot_requires_finished_runs_and_explicit_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs.jsonl").write_text(
                json.dumps({
                    "run_id": "r1",
                    "finished_at": "now",
                    "result": {"rsi_runtime": {"status": "active", "version_id": "v1", "policy_hash": "h1"}},
                }) + "\n", encoding="utf-8"
            )
            (root / "hitl_audit.jsonl").write_text(json.dumps({"run_id": "r1", "action": "approve"}) + "\n", encoding="utf-8")
            route = root / "route.json"
            route.write_text(json.dumps({"passed": True, "fixture_count": 5}), encoding="utf-8")
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({"system": {"runtime_scope": {"claimable": True}}, "input_integrity": {"status": "passed"}}), encoding="utf-8")
            report = validate_delivery_snapshot(root, ["r1"], route_report=route, evaluation_report=evaluation, require_hitl=True, require_rsi=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["hitl_actions"]["r1"], ["approve"])
        self.assertFalse(report["claims"]["accuracy_claimable"])

    def test_snapshot_fail_closes_missing_hitl_and_rsi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs.jsonl").write_text(json.dumps({"run_id": "r1", "finished_at": "now", "result": {}}) + "\n", encoding="utf-8")
            (root / "hitl_audit.jsonl").write_text("", encoding="utf-8")
            report = validate_delivery_snapshot(root, ["r1"], require_hitl=True, require_rsi=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("hitl_decision_missing_or_ambiguous:r1", report["errors"])
        self.assertIn("rsi_runtime_missing_or_inactive:r1", report["errors"])


if __name__ == "__main__":
    unittest.main()
