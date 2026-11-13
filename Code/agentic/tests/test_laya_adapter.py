import unittest

from agentic.platform.contracts import InspectionFeatures
from agentic.platform.jev.laya_adapter import LayaAdapter


class _FakeLayaRouter:
    def __init__(self):
        self.questions = None
        self.model = None

    def predict(self, state, *, questions, model=None):
        self.questions = questions
        self.model = model
        return {
            "answers": {
                "route": {
                    "choice": "full_loop",
                    "answer_confidence": 0.81,
                }
            },
            "routing": {"model": "fake-laya"},
        }


class LayaAdapterTests(unittest.TestCase):
    def test_laya_typed_question_contract_and_answer_projection(self):
        backend = _FakeLayaRouter()
        adapter = LayaAdapter(backend)
        features = InspectionFeatures(
            run_id="laya-adapter-test",
            visual_confidence=0.8,
            risk_score=0.2,
            knowledge_coverage=0.8,
            case_similarity=0.5,
            novelty_score=0.1,
        )
        result = adapter.predict(features)
        self.assertEqual(result["route"], "full_loop")
        self.assertGreater(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(backend.questions["route"]["type"], "choice")
        self.assertIn("simple_loop", backend.questions["route"]["criteria"])
        self.assertEqual(backend.model, "typed-decisions")


if __name__ == "__main__":
    unittest.main()
