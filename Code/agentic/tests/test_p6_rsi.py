from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.rsi.candidate import CandidateDiff, CandidateReferenceError, ForbiddenMutationError, build_candidate
from agentic.rsi.cli import build_parser
from agentic.rsi.evaluator import OfflineReplayer, ReplayBundle
from agentic.rsi.gate import evaluate_gate
from agentic.rsi.publisher import Publisher
from agentic.scripts.rsi_defect_demo import run as run_rsi_defect_demo


class P6RsiTests(unittest.TestCase):
    def _candidate(self):
        return build_candidate(
            surface="prompts",
            before={"prompts": {"diagnosis": "old"}},
            after={"prompts": {"diagnosis": "new"}},
            source_run_ids=["run-1"],
            source_refs=["route-stable"],
        )

    def _passing_gate(self, candidate):
        valid_row = {"quality": 0.9, "safety": 1.0, "efficiency": 0.8}
        bundle = ReplayBundle.from_mapping({"train": [valid_row], "frozen": [valid_row], "sealed": [valid_row]})
        results = OfflineReplayer().replay_bundle(candidate, bundle)
        return evaluate_gate(candidate, results, results)

    def test_only_mutable_surfaces_and_known_references(self):
        candidate = self._candidate()
        self.assertEqual(candidate.diffs[0].operation, "replace")
        candidate.validate(["route-stable"])
        with self.assertRaises(CandidateReferenceError):
            candidate.validate(["other-source"])
        with self.assertRaises(ForbiddenMutationError):
            build_candidate(surface="weights", before={}, after={"weights": {"x": 1}})
        with self.assertRaises(ForbiddenMutationError):
            build_candidate(surface="jev_policies", before={"jev_policies": {}}, after={"jev_policies": {"safety_rules": {"x": 1}}})

    def test_train_frozen_sealed_replay_and_gate(self):
        candidate = self._candidate()
        bundle = ReplayBundle.from_mapping({
            "train": [{"quality": 0.9, "safety": 1.0, "efficiency": 0.8}],
            "frozen": [{"quality": 0.9, "safety": 1.0, "efficiency": 0.8}],
            "sealed": [{"quality": 0.9, "safety": 1.0, "efficiency": 0.8}],
        })
        baseline = OfflineReplayer().replay_bundle(candidate, bundle)
        proposed = OfflineReplayer().replay_bundle(candidate, bundle)
        gate = evaluate_gate(candidate, baseline, proposed)
        self.assertTrue(gate.accepted)
        self.assertTrue(all(item.passed for item in proposed.values()))
        self.assertEqual(bundle.manifest_hashes["sealed"], ReplayBundle.from_mapping({"sealed": bundle.sealed}).manifest_hashes["sealed"])

    def test_manual_approval_publish_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publisher = Publisher(root)
            first = self._candidate()
            publisher.register(first)
            valid_row = {"quality": 0.9, "safety": 1.0, "efficiency": 0.8}
            bundle = ReplayBundle.from_mapping({"train": [valid_row], "frozen": [valid_row], "sealed": [valid_row]})
            gate = evaluate_gate(first, OfflineReplayer().replay_bundle(first, bundle), OfflineReplayer().replay_bundle(first, bundle))
            publisher.approve(first, "expert-1", "reviewed", gate.to_dict())
            version_one = publisher.publish(first, gate)
            self.assertEqual(publisher.current().version_id, version_one.version_id)

            second = build_candidate(
                surface="prompts",
                before=version_one.surfaces,
                after={"prompts": {"diagnosis": "third"}},
                parent_version=version_one.version_id,
            )
            publisher.register(second)
            gate_two = evaluate_gate(second, OfflineReplayer().replay_bundle(second, bundle), OfflineReplayer().replay_bundle(second, bundle))
            publisher.approve(second, "expert-1", "reviewed", gate_two)
            version_two = publisher.publish(second, gate_two)
            self.assertEqual(publisher.current().version_id, version_two.version_id)
            self.assertEqual(publisher.rollback().version_id, version_one.version_id)

    def test_approval_and_publish_require_complete_candidate_bound_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = Publisher(Path(tmp))
            candidate = self._candidate()
            publisher.register(candidate)
            accepted_only = {"candidate_id": candidate.candidate_id, "accepted": True}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                publisher.approve(candidate, "reviewer", "approve", accepted_only)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                publisher.publish(candidate, accepted_only)

            gate = self._passing_gate(candidate)
            partial = gate.to_dict()
            partial["checks"] = {"candidate_valid": True}
            with self.assertRaisesRegex(ValueError, "required successful checks"):
                publisher.approve(candidate, "reviewer", "approve", partial)

            foreign_gate = self._passing_gate(self._candidate())
            with self.assertRaisesRegex(ValueError, "does not match"):
                publisher.approve(candidate, "reviewer", "approve", foreign_gate)

    def test_approval_is_bound_to_the_approved_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = Publisher(Path(tmp))
            candidate = self._candidate()
            publisher.register(candidate)
            gate = self._passing_gate(candidate)
            publisher.approve(candidate, "reviewer", "approved diff", gate)
            candidate.diffs[0] = CandidateDiff(
                surface="prompts", path="diagnosis", before="old", after="tampered"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                publisher.publish(candidate, gate)

    def test_gate_requires_baseline_and_candidate_to_use_identical_replay_rows(self):
        candidate = self._candidate()
        baseline_rows = [{"quality": 0.9, "safety": 1.0, "efficiency": 0.8, "sample": "baseline"}]
        candidate_rows = [{"quality": 0.9, "safety": 1.0, "efficiency": 0.8, "sample": "different"}]
        baseline_bundle = ReplayBundle.from_mapping({name: baseline_rows for name in ("train", "frozen", "sealed")})
        candidate_bundle = ReplayBundle.from_mapping({name: candidate_rows for name in ("train", "frozen", "sealed")})
        baseline = OfflineReplayer().replay_bundle(candidate, baseline_bundle)
        proposed = OfflineReplayer().replay_bundle(candidate, candidate_bundle)
        gate = evaluate_gate(candidate, baseline, proposed)
        self.assertFalse(gate.accepted)
        self.assertFalse(gate.checks["frozen_input_hash_match"])
        self.assertFalse(gate.checks["sealed_input_hash_match"])

    def test_cli_approval_requires_a_gate_report(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "approve", "--root", "runtime/rsi", "--candidate", "candidate-1",
                "--reviewer-id", "reviewer-1",
            ])

    def test_rsi_demo_replays_baseline_and_candidate_on_identical_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_rsi_defect_demo(Path(tmp))
        self.assertTrue(report["gate"]["accepted"])
        self.assertTrue(report["rollback_verified"])
        for split in ("train", "frozen", "sealed"):
            metrics = report["gate"]["metrics"][split]
            self.assertEqual(metrics["baseline_hash"], metrics["candidate_hash"])

    def test_empty_replay_split_cannot_pass_gate(self):
        candidate = self._candidate()
        bundle = ReplayBundle.from_mapping({"train": [{"quality": 0.9}], "frozen": [], "sealed": [{"quality": 0.9}]})
        results = OfflineReplayer().replay_bundle(candidate, bundle)
        self.assertFalse(results["frozen"].passed)
        gate = evaluate_gate(candidate, results, results)
        self.assertFalse(gate.accepted)
        self.assertIn("all_splits_nonempty", gate.checks)

    def test_missing_required_metrics_fail_closed(self):
        candidate = self._candidate()
        bundle = ReplayBundle.from_mapping({
            "train": [{"quality": 1.0}],
            "frozen": [{"quality": 1.0, "safety": 1.0}],
            "sealed": [{"safety": 1.0, "efficiency": 1.0}],
        })
        results = OfflineReplayer().replay_bundle(candidate, bundle)
        self.assertFalse(any(result.passed for result in results.values()))
        self.assertTrue(all(result.errors for result in results.values()))
        gate = evaluate_gate(candidate, results, results)
        self.assertFalse(gate.accepted)
        self.assertFalse(gate.checks["all_splits_nonempty"])

    def test_empty_metrics_cannot_promote_rsi_candidate(self):
        candidate = self._candidate()
        empty = {name: {"row_count": 1, "metrics": {}} for name in ("train", "frozen", "sealed")}
        gate = evaluate_gate(candidate, empty, empty)
        self.assertFalse(gate.accepted)
        self.assertFalse(gate.checks["all_splits_nonempty"])
        self.assertFalse(gate.checks["frozen_required_metrics_present"])

    def test_nonfinite_or_out_of_range_scores_are_rejected(self):
        candidate = self._candidate()
        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(invalid=invalid):
                result = OfflineReplayer().replay(candidate, [{"quality": invalid, "safety": 1.0, "efficiency": 0.8}], split="train")
                self.assertFalse(result.passed)
                self.assertTrue(result.errors)


if __name__ == "__main__":
    unittest.main()
