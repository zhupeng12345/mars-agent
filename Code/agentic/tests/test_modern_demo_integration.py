import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from agentic.platform.case_memory import SpecialCaseMemory
from agentic.rsi.candidate import Candidate, CandidateDiff, build_candidate
from agentic.rsi.evaluator import OfflineReplayer, ReplayBundle
from agentic.rsi.gate import evaluate_gate
from agentic.rsi.publisher import Publisher
from agentic.scripts.demo_run import run_demo


class ModernDemoIntegrationTests(unittest.TestCase):
    def test_demo_wires_knowledge_case_memory_and_fake_gateway(self) -> None:
        code_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "inspection.png"
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            image[48:208, 48:208] = (90, 140, 210)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            runtime = root / "runtime"
            registry = runtime / "approved-rsi-registry"
            publisher = Publisher(registry)
            candidate = Candidate(
                base_snapshot={"jev_policies": {"minimum_laya_confidence": 0.7}},
                diffs=[
                    CandidateDiff("jev_policies", "minimum_laya_confidence", 0.7, 0.9),
                    CandidateDiff("knowledge_queries", "extra_terms", None, ["visual trend"]),
                    CandidateDiff("prompts", "variant", None, "diagnosis-evidence-first-v1"),
                ],
            )
            publisher.register(candidate)
            row = {"quality": 0.9, "safety": 1.0, "efficiency": 0.8}
            bundle = ReplayBundle.from_mapping({split: [row] for split in ("train", "frozen", "sealed")})
            replay = OfflineReplayer().replay_bundle(candidate, bundle)
            gate = evaluate_gate(candidate, replay, replay)
            publisher.approve(candidate, "reviewer-1", "approved integration fixture", gate)
            active_policy = publisher.publish(candidate, gate)

            first = run_demo(
                code_root,
                image_path,
                output=root / "first.json",
                no_model=True,
                runtime_dir=runtime,
                case_memory_root=runtime,
                llm_profile="fake_json",
                jev_policy_registry_root=registry,
            )
            self.assertEqual(first["rsi_runtime"]["status"], "active")
            self.assertEqual(first["rsi_runtime"]["version_id"], active_policy.version_id)
            self.assertEqual(first["pre_route_decision"]["policy_version"], active_policy.version_id)
            self.assertEqual(first["jev_router"]["minimum_laya_confidence"], 0.9)
            self.assertEqual(first["rsi_runtime"]["knowledge_query_terms"], ["visual trend"])
            self.assertEqual(first["rsi_runtime"]["prompt_variant"], "diagnosis-evidence-first-v1")
            self.assertEqual(first["prompt_version"], "diagnosis-evidence-first-v1")
            self.assertEqual(first["llm_gateway"].get("prompt_variant"), "diagnosis-evidence-first-v1")
            self.assertEqual(first["llm_gateway"]["provider"], "fake")
            self.assertEqual(first["llm_gateway"]["status"], "completed")
            self.assertTrue(first["llm_gateway"]["content"]["summary"])
            self.assertIn("risk_level", first["llm_gateway"]["content"])
            self.assertIn("requires_human_review", first["llm_gateway"]["content"])
            self.assertGreaterEqual(len(first["knowledge"]), 3)
            self.assertEqual(first["case_candidate"]["state"], "candidate")
            self.assertTrue((runtime / "special_cases.jsonl").exists())

            memory = SpecialCaseMemory(runtime)
            case_id = first["case_candidate"]["case_id"]
            memory.transition(case_id, "eval_only", reviewer_id="test", reason="evaluate")
            memory.transition(case_id, "approved_special_case", reviewer_id="test", reason="approve")
            memory.transition(case_id, "active", reviewer_id="test", reason="publish", regression_passed=True)

            second = run_demo(
                code_root,
                image_path,
                output=root / "second.json",
                no_model=True,
                runtime_dir=runtime,
                case_memory_root=runtime,
                llm_profile="fake_json",
            )
            self.assertTrue(any(item["case_id"] == case_id for item in second["cases"]))
            self.assertEqual(second["features"]["features"]["case_count"], len(second["cases"]))
            self.assertIn("llm_gateway", second["human_review"]["model_metadata"])

    def test_quality_gate_skips_diagnostic_provider(self) -> None:
        code_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "black.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8)))
            with patch("agentic.orchestration.diagnosis_graph.LLMGateway.chat", side_effect=AssertionError("diagnostic provider must not be called")) as chat:
                result = run_demo(code_root, image_path, output=root / "bad.json", runtime_dir=root / "runtime")
            self.assertIn(result["quality"]["quality_status"], {"retake", "human"})
            self.assertEqual(result["visual"]["weights_status"], "not_called")
            self.assertEqual(result["llm_gateway"]["status"], "skipped_by_jev")
            self.assertIsNone(result["llm_gateway"]["error_type"])
            self.assertIsNone(result["case_candidate"])
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
