import unittest

from agentic.platform.contracts import InspectionFeatures
from agentic.platform.jev.router import JevRouter, LayaJevRouter, create_jev_router


def features(**kwargs):
    values = {
        "run_id": "run-jev",
        "visual_confidence": 0.9,
        "risk_score": 0.1,
        "knowledge_coverage": 1.0,
        "case_similarity": 0.9,
        "novelty_score": 0.1,
        "evidence_conflict": 0.0,
        "features": {"quality_status": "pass", "target_present": True, "case_applicable": True},
    }
    values.update(kwargs)
    return InspectionFeatures(**values)


class JevRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = JevRouter()

    def test_runtime_identity_is_laya_jev_with_compatibility_class(self):
        self.assertEqual(self.router.model_profile, "deterministic_alu_router_v1")
        self.assertEqual(self.router.gate_version, "alu-jev-safety-0.1")
        self.assertIsInstance(LayaJevRouter(), JevRouter)

    def test_factory_selects_lazy_laya_adapter_without_loading_model(self):
        router = create_jev_router("laya_adapter")
        self.assertIsInstance(router, LayaJevRouter)
        self.assertEqual(router.model_profile, "laya_typed_decisions_alu_adapter_v1")
        self.assertIsNone(router.adapter.router)
        with self.assertRaises(ValueError):
            create_jev_router("unknown")

    def test_laya_proposal_backend_is_reported_only_when_it_made_the_decision(self):
        class FullLoopProposal:
            def predict(self, _features):
                return {"route": "full_loop", "confidence": 0.9}

        class SimpleLoopProposal:
            def predict(self, _features):
                return {"route": "simple_loop", "confidence": 0.9}

        simple_decision = JevRouter(adapter=FullLoopProposal()).route(features())
        self.assertEqual(simple_decision.route, "simple_loop")
        self.assertEqual(simple_decision.router_backend, "deterministic_rule")

        complex_features = features(case_similarity=0.2, features={"quality_status": "pass", "target_present": True, "case_applicable": False})
        adapted = JevRouter(adapter=SimpleLoopProposal()).route(complex_features)
        self.assertEqual(adapted.route, "simple_loop")
        self.assertEqual(adapted.router_backend, "laya_adapter")

    def test_stable_case_simple(self):
        decision = self.router.route(features())
        self.assertEqual(decision.route, "simple_loop")
        self.assertEqual(decision.router_backend, "deterministic_rule")

    def test_high_risk_full(self):
        self.assertEqual(self.router.route(features(risk_score=0.9)).route, "full_loop")

    def test_bad_quality_retake(self):
        self.assertEqual(self.router.route(features(features={"quality_status": "retake", "target_present": True})).route, "retake")

    def test_quality_uncertainty_routes_to_human(self):
        decision = self.router.route(features(features={"quality_status": "human", "target_present": True}))
        self.assertEqual(decision.route, "human")
        self.assertIn("quality_requires_human_review", decision.reasons)

    def test_ood_human(self):
        self.assertEqual(self.router.route(features(novelty_score=0.95)).route, "human")

    def test_low_confidence_human(self):
        self.assertEqual(self.router.route(features(visual_confidence=0.2)).route, "human")

    def test_same_input_is_deterministic(self):
        first = self.router.route(features()).to_dict()
        second = self.router.route(features()).to_dict()
        self.assertEqual(first["route"], second["route"])
        self.assertEqual(first["reasons"], second["reasons"])

    def test_adapter_cannot_downgrade_high_risk_to_simple(self):
        class UnsafeAdapter:
            def predict(self, _features):
                return {"route": "simple_loop", "confidence": 0.99}

        decision = JevRouter(adapter=UnsafeAdapter()).route(features(risk_score=0.9))
        self.assertEqual(decision.route, "full_loop")
        self.assertEqual(decision.router_backend, "deterministic_rule")
        self.assertIn("adapter_rejected_hard_gate", decision.reasons)
        self.assertIn("risk_high", decision.reasons)

    def test_adapter_can_optimize_unblocked_full_route(self):
        class FastAdapter:
            def predict(self, _features):
                return {"route": "simple_loop", "confidence": 0.99, "reason": "stable_retrieval"}

        decision = JevRouter(adapter=FastAdapter()).route(features(case_similarity=0.2))
        self.assertEqual(decision.route, "simple_loop")
        self.assertEqual(decision.router_backend, "laya_adapter")
        self.assertIn("adapter_proposal", decision.reasons)

    def test_adapter_exception_and_low_confidence_record_actual_backend(self):
        class BrokenAdapter:
            def predict(self, _features):
                raise TimeoutError("router timeout")

        class UncertainAdapter:
            def predict(self, _features):
                return {"route": "simple_loop", "confidence": 0.4}

        failed = JevRouter(adapter=BrokenAdapter()).route(features(risk_score=0.9))
        self.assertEqual(failed.route, "human")
        self.assertEqual(failed.router_backend, "safety_fallback")

        uncertain = JevRouter(adapter=UncertainAdapter()).route(features(risk_score=0.9))
        self.assertEqual(uncertain.route, "full_loop")
        self.assertEqual(uncertain.router_backend, "laya_adapter_fallback")
        self.assertIn("adapter_abstain_or_low_confidence", uncertain.reasons)

    def test_case_ids_are_preserved_for_audit(self):
        decision = self.router.route(features(features={"quality_status": "pass", "target_present": True, "similar_case_ids": ["case-17", "case-17", "case-42"]}))
        self.assertEqual(decision.similar_case_ids, ["case-17", "case-42"])


if __name__ == "__main__":
    unittest.main()
