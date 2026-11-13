from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.platform.contracts import InspectionFeatures
from agentic.platform.jev.router import JevRouter
from agentic.rsi.candidate import Candidate, CandidateDiff, build_candidate
from agentic.rsi.evaluator import OfflineReplayer, ReplayBundle
from agentic.rsi.gate import evaluate_gate
from agentic.rsi.publisher import Publisher
from agentic.rsi.runtime_policy import resolve_runtime_jev_policy


class Proposal:
    def __init__(self, confidence: float = 0.8):
        self.confidence = confidence

    def predict(self, _features):
        return {"route": "simple_loop", "confidence": self.confidence, "reason": "fixture proposal"}


def feature_fixture() -> InspectionFeatures:
    return InspectionFeatures(
        run_id="rsi-runtime-test",
        visual_confidence=0.9,
        risk_score=0.1,
        knowledge_coverage=1.0,
        case_similarity=0.2,
        novelty_score=0.1,
        evidence_conflict=0.0,
        features={"quality_status": "pass", "target_present": True, "case_applicable": False},
    )


def passing_gate(candidate):
    row = {"quality": 0.9, "safety": 1.0, "efficiency": 0.8}
    bundle = ReplayBundle.from_mapping({split: [row] for split in ("train", "frozen", "sealed")})
    replay = OfflineReplayer().replay_bundle(candidate, bundle)
    return evaluate_gate(candidate, replay, replay)


def publish_policy(publisher: Publisher, value: float, parent=None):
    before = parent.surfaces if parent else {"jev_policies": {"minimum_laya_confidence": 0.7}}
    candidate = build_candidate(
        surface="jev_policies",
        before=before,
        after={"jev_policies": {"minimum_laya_confidence": value}},
        parent_version=parent.version_id if parent else "base",
    )
    publisher.register(candidate)
    gate = passing_gate(candidate)
    publisher.approve(candidate, "reviewer-1", "runtime policy regression", gate)
    return publisher.publish(candidate, gate)


class RsiRuntimePolicyTests(unittest.TestCase):
    def test_publish_and_rollback_change_next_laya_decision_and_keep_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "explicit-registry"
            publisher = Publisher(registry)
            first = publish_policy(publisher, 0.95)
            second = publish_policy(publisher, 0.75, first)

            active = resolve_runtime_jev_policy(registry)
            self.assertEqual(active.status, "active")
            self.assertEqual(active.version_id, second.version_id)
            self.assertEqual(active.policy.minimum_laya_confidence, 0.75)
            accepted = JevRouter(
                adapter=Proposal(),
                policy=active.policy,
                policy_version=active.version_id,
                policy_hash=active.policy_hash,
            ).route(feature_fixture())
            self.assertEqual(accepted.route, "simple_loop")
            self.assertEqual(accepted.policy_version, second.version_id)

            rolled_back = publisher.rollback()
            restored = resolve_runtime_jev_policy(registry)
            self.assertEqual(rolled_back.version_id, first.version_id)
            self.assertEqual(restored.version_id, first.version_id)
            rejected = JevRouter(
                adapter=Proposal(),
                policy=restored.policy,
                policy_version=restored.version_id,
                policy_hash=restored.policy_hash,
            ).route(feature_fixture())
            self.assertEqual(rejected.route, "full_loop")
            self.assertEqual(rejected.router_backend, "laya_adapter_fallback")
            self.assertEqual(rejected.policy_version, first.version_id)

    def test_unconfigured_or_unpublished_registry_uses_builtin_policy(self):
        self.assertEqual(resolve_runtime_jev_policy(None).status, "disabled")
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_runtime_jev_policy(Path(tmp) / "not-created")
        self.assertEqual(resolved.status, "disabled")
        self.assertEqual(resolved.policy.minimum_laya_confidence, 0.7)

    def test_tampered_or_unsupported_registry_fails_back_to_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry"
            publisher = Publisher(registry)
            published = publish_policy(publisher, 0.95)
            publisher.approvals_path.write_text("", encoding="utf-8")
            no_approval = resolve_runtime_jev_policy(registry)
            self.assertEqual(no_approval.status, "fallback")
            self.assertIn("invalid_registry", no_approval.reason)
            self.assertEqual(no_approval.policy.minimum_laya_confidence, 0.7)

            publisher.approve(published.candidate_id, "reviewer-1", "restore approval", published.metadata["gate"])
            altered = publisher.lineage.get(published.version_id)
            altered.surfaces["prompts"] = {"diagnosis": "untrusted"}
            publisher.lineage.path.write_text(
                __import__("json").dumps(altered.to_dict(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            mixed_surface = resolve_runtime_jev_policy(registry)
            self.assertEqual(mixed_surface.status, "fallback")
            self.assertIn("invalid_registry", mixed_surface.reason)

    def test_publisher_rejects_out_of_range_jev_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = Publisher(Path(tmp))
            candidate = build_candidate(
                surface="jev_policies",
                before={"jev_policies": {"minimum_laya_confidence": 0.7}},
                after={"jev_policies": {"minimum_laya_confidence": 0.6}},
            )
            publisher.register(candidate)
            gate = passing_gate(candidate)
            publisher.approve(candidate, "reviewer-1", "invalid policy probe", gate)
            with self.assertRaisesRegex(ValueError, r"\[0.7, 1.0\]"):
                publisher.publish(candidate, gate)
            self.assertIsNone(publisher.current())

    def test_runtime_policy_never_relaxes_deterministic_hard_safety_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry"
            publisher = Publisher(registry)
            active = publish_policy(publisher, 0.95)
            resolved = resolve_runtime_jev_policy(registry)

            decision = JevRouter(
                adapter=Proposal(),
                policy=resolved.policy,
                policy_version=resolved.version_id,
                policy_hash=resolved.policy_hash,
            ).route(feature_fixture())
            risky = feature_fixture()
            risky.risk_score = 0.9
            blocked = JevRouter(
                adapter=Proposal(confidence=0.99),
                policy=resolved.policy,
                policy_version=resolved.version_id,
                policy_hash=resolved.policy_hash,
            ).route(risky)
        self.assertEqual(active.version_id, resolved.version_id)
        self.assertEqual(decision.route, "full_loop")
        self.assertEqual(blocked.route, "full_loop")
        self.assertEqual(blocked.router_backend, "deterministic_rule")
        self.assertIn("adapter_rejected_hard_gate", blocked.reasons)

    def test_approved_bounded_prompt_and_query_surfaces_are_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry"
            publisher = Publisher(registry)
            candidate = Candidate(
                base_snapshot={"jev_policies": {"minimum_laya_confidence": 0.7}},
                diffs=[
                    CandidateDiff("jev_policies", "minimum_laya_confidence", 0.7, 0.8),
                    CandidateDiff("knowledge_queries", "extra_terms", None, ["thermal trend", "surface evidence"]),
                    CandidateDiff("prompts", "variant", None, "diagnosis-evidence-first-v1"),
                ],
            )
            publisher.register(candidate)
            gate = passing_gate(candidate)
            publisher.approve(candidate, "reviewer-1", "bounded runtime surfaces", gate)
            published = publisher.publish(candidate, gate)

            resolved = resolve_runtime_jev_policy(registry)
            self.assertEqual(resolved.status, "active")
            self.assertEqual(resolved.version_id, published.version_id)
            self.assertEqual(resolved.knowledge_query_terms, ("thermal trend", "surface evidence"))
            self.assertEqual(resolved.prompt_variant, "diagnosis-evidence-first-v1")
            self.assertNotEqual(resolved.policy_hash, "builtin")

    def test_invalid_runtime_prompt_or_query_surface_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry"
            publisher = Publisher(registry)
            bad_prompt = Candidate(
                base_snapshot={},
                diffs=[CandidateDiff("prompts", "variant", None, "arbitrary-user-prompt")],
            )
            publisher.register(bad_prompt)
            gate = passing_gate(bad_prompt)
            publisher.approve(bad_prompt, "reviewer-1", "invalid prompt probe", gate)
            publisher.publish(bad_prompt, gate)
            self.assertEqual(resolve_runtime_jev_policy(registry).status, "fallback")

            too_many_terms = Candidate(
                base_snapshot={},
                diffs=[CandidateDiff("knowledge_queries", "extra_terms", None, ["a", "b", "c", "d", "e"])],
            )
            publisher.register(too_many_terms)
            gate = passing_gate(too_many_terms)
            publisher.approve(too_many_terms, "reviewer-1", "invalid query probe", gate)
            publisher.publish(too_many_terms, gate)
            self.assertEqual(resolve_runtime_jev_policy(registry).status, "fallback")


if __name__ == "__main__":
    unittest.main()
