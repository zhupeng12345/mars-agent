from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.agents.specialists import (
    case_comparison,
    evidence_quality,
    knowledge_retrieval,
    process_state_analysis,
    visual_review,
)
from agentic.orchestration.diagnosis_graph import run_diagnosis_graph
from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.contracts import DiagnosisDraftPayload, InspectionFeatures
from agentic.platform.llm_gateway import GatewayResponse


def _rule(rule_id: str, title: str, text: str) -> dict:
    return {
        "rule_id": rule_id,
        "title": title,
        "source": "domain-manual",
        "source_version": "v1",
        "effective_date": "2026-01-01",
        "preconditions": ["visual_evidence"],
        "applicability": ["visual", "fire_eye"],
        "risk_level": "medium",
        "allowed_output": ["inspection_suggestion"],
        "text": text,
    }


class FullLoopSpecialistTests(unittest.TestCase):
    def test_specialists_derive_role_specific_findings_and_abstain_on_missing(self):
        context = {
            "visual": {
                "source_id": "media-1",
                "has_target": True,
                "label": "H",
                "confidence": 0.82,
                "uncertainty": 0.2,
                "class_probabilities": {"H": 0.82, "N": 0.12, "C": 0.06},
                "frame_ids": [2, 4, 6],
                "key_frame_ids": [4],
                "temporal_consistency": 0.91,
                "roi_coverage": 0.75,
            },
            "quality": {"quality_status": "pass", "target_present": True, "issues": []},
            "process_state": {"process_regime": "stable", "event_risk": 0.2, "alignment_quality": "unknown", "process_missing": ["video_mapping"], "process_tags": ["stable"], "run_id": "run-process-snapshot"},
        }
        visual = visual_review(context)
        quality = evidence_quality(context)
        process = process_state_analysis(context)
        self.assertEqual(visual["analysis_status"], "supported")
        self.assertEqual(visual["source_ids"], ["media-1"])
        self.assertEqual(quality["analysis_status"], "complete")
        self.assertEqual(process["video_relation"], "independent_prior")
        self.assertEqual(process["alignment_quality"], "unknown")
        self.assertEqual(process["source_ids"], ["process-snapshot-run-process-snapshot"])

        self.assertEqual(visual_review({})["analysis_status"], "insufficient")
        self.assertEqual(knowledge_retrieval({})["analysis_status"], "abstain")
        self.assertEqual(case_comparison({})["analysis_status"], "abstain")
        self.assertEqual(process_state_analysis({})["process_missing"], ["process_snapshot"])
        self.assertEqual(evidence_quality({})["analysis_status"], "incomplete")

    def test_knowledge_and_case_specialists_retrieve_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "rules.jsonl"
            knowledge_path.write_text(json.dumps(_rule("fire-eye-H", "火眼偏高征兆", "偏高状态应复核火眼视觉证据。"), ensure_ascii=False) + "\n", encoding="utf-8")
            case_root = root / "cases"
            memory = SpecialCaseMemory(case_root)
            features = {"confidence": 0.82, "quality_score": 0.8, "target_present": 1.0, "temporal_consistency": 0.9}
            candidate = memory.add_candidate(source_id="media-approved", features=features, conclusion={"state": "H"}, run_id="seed", applicable_when=("visual", "fire_eye"))
            memory.transition(candidate["case_id"], "eval_only", reviewer_id="expert", reason="screen")
            memory.transition(candidate["case_id"], "approved_special_case", reviewer_id="expert", reason="approve")
            memory.transition(candidate["case_id"], "active", reviewer_id="expert", reason="publish", regression_passed=True)
            context = {
                "knowledge_path": str(knowledge_path),
                "case_memory_root": str(case_root),
                "case_features": features,
                "visual": {"label": "H", "risk_level": "high"},
                "process_state": {"process_regime": "stable"},
            }
            knowledge = knowledge_retrieval(context)
            cases = case_comparison(context)
            self.assertEqual(knowledge["analysis_status"], "retrieved")
            self.assertEqual(knowledge["sources"][0]["source_id"], "fire-eye-H")
            self.assertTrue(knowledge["sources"][0]["locator"])
            self.assertEqual(cases["analysis_status"], "matched")
            self.assertEqual(cases["matches"][0]["case_id"], candidate["case_id"])
            self.assertEqual(cases["source_ids"], ["media-approved"])

    def test_default_full_loop_fans_in_five_distinct_domain_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "rules.jsonl"
            knowledge_path.write_text(json.dumps(_rule("rule-H", "火眼状态", "H 状态需要复核。"), ensure_ascii=False) + "\n", encoding="utf-8")
            case_root = root / "cases"
            memory = SpecialCaseMemory(case_root)
            case_features = {"confidence": 0.8, "target_present": 1.0, "quality_score": 0.8, "temporal_consistency": 0.9}
            case = memory.add_candidate(source_id="case-source", features=case_features, conclusion={"visual_state": "H"}, run_id="seed", applicable_when=("visual", "fire_eye"))
            for state, reason, passed in (("eval_only", "screen", False), ("approved_special_case", "approved", False), ("active", "published", True)):
                memory.transition(case["case_id"], state, reviewer_id="expert", reason=reason, regression_passed=passed)
            visual = {"source_id": "media-full", "has_target": True, "label": "H", "confidence": 0.8, "uncertainty": 0.2, "temporal_consistency": 0.9, "frame_ids": [1, 3]}
            context = {
                "llm_profile": "fake_json",
                "visual": visual,
                "quality": {"quality_status": "pass", "target_present": True, "issues": []},
                "knowledge_path": str(knowledge_path),
                "case_memory_root": str(case_root),
                "case_features": case_features,
                "process_state": {"process_regime": "stable", "event_risk": 0.2, "alignment_quality": "unknown", "process_missing": ["video_mapping"]},
                "media_source_id": "media-full",
            }
            features = InspectionFeatures(run_id="run-specialists", risk_score=0.95, visual_confidence=0.8, features={"quality_status": "pass", "target_present": True})
            captured = {}

            def aggregate_call(_self, _profile, messages, **_kwargs):
                captured["prompt"] = messages[0]["content"]
                captured["response_schema"] = _kwargs.get("response_schema")
                return GatewayResponse(
                    status="completed",
                    profile="fake_json",
                    provider="fake",
                    model="test-model",
                    request_id="specialist-test",
                    content={
                        "diagnosis_class": "visual_state",
                        "risk_level": "high",
                        "confidence": 0.8,
                        "summary": "aggregated specialist evidence",
                        "findings": ["Evidence sources reviewed."],
                        "source_ids": ["media-full", "rule-H", "case-source"],
                        "requires_human_review": True,
                    },
                )

            with patch("agentic.orchestration.diagnosis_graph.LLMGateway.chat", aggregate_call):
                result = run_diagnosis_graph(features, context=context)
            bundle = {item["agent_id"]: item for item in result["result"]["evidence_bundle"]}
            expected = {"visual_review", "evidence_quality", "knowledge_rag", "case_compare", "process_state"}
            self.assertTrue(expected.issubset(bundle))
            self.assertEqual(bundle["visual_review"]["output"]["source_ids"], ["media-full"])
            self.assertEqual(bundle["visual_review"]["output"]["evidence_ids"], ["media-full"])
            self.assertEqual(bundle["visual_review"]["evidence"][0]["source_id"], "media-full")
            self.assertEqual(bundle["evidence_quality"]["output"]["analysis_status"], "complete")
            self.assertEqual(bundle["knowledge_rag"]["output"]["source_ids"], ["rule-H"])
            self.assertEqual(bundle["knowledge_rag"]["evidence"][0]["kind"], "knowledge")
            self.assertEqual(bundle["case_compare"]["output"]["source_ids"], ["case-source"])
            self.assertEqual(bundle["process_state"]["output"]["video_relation"], "independent_prior")
            self.assertEqual(bundle["process_state"]["output"]["source_ids"], ["process-snapshot-run-specialists"])
            self.assertEqual(bundle["process_state"]["evidence"][0]["source_id"], "process-snapshot-run-specialists")
            summaries = result["result"]["diagnosis"].get("findings", [])
            self.assertTrue(summaries)
            self.assertIn('"video_relation": "independent_prior"', captured["prompt"])
            self.assertIn('"analysis_status": "retrieved"', captured["prompt"])
            self.assertIs(captured["response_schema"], DiagnosisDraftPayload)
            statuses = [item for item in result["trace"] if item.get("node") == "full_loop_fanout"]
            self.assertTrue(statuses)
            self.assertEqual(statuses[-1]["child_count"], 5)

    def test_one_specialist_failure_keeps_other_four_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "rules.jsonl"
            knowledge_path.write_text(json.dumps(_rule("rule-H", "火眼状态", "H 状态需要复核。"), ensure_ascii=False) + "\n", encoding="utf-8")
            context = {
                "llm_profile": "fake_json",
                "visual": {"source_id": "media-1", "has_target": True, "label": "H", "confidence": 0.8, "temporal_consistency": 0.9, "frame_ids": [1]},
                "quality": {"quality_status": "pass", "target_present": True, "issues": []},
                "knowledge_path": str(knowledge_path),
                "process_state": {"process_regime": "stable", "alignment_quality": "unknown"},
            }
            from agentic.agents.specialists import build_full_loop_workers

            workers = build_full_loop_workers()
            workers["knowledge_rag"] = lambda _: (_ for _ in ()).throw(RuntimeError("injected specialist failure"))
            context["workers"] = workers
            context["worker_retries"] = 0
            features = InspectionFeatures(run_id="run-specialist-failure", risk_score=0.95, visual_confidence=0.8, features={"quality_status": "pass", "target_present": True})
            result = run_diagnosis_graph(features, context=context)
            bundle = {item["agent_id"]: item for item in result["result"]["evidence_bundle"]}
            self.assertEqual(bundle["knowledge_rag"]["status"], "failed")
            self.assertEqual(bundle["knowledge_rag"]["error_code"], "RuntimeError")
            completed = [agent for agent, item in bundle.items() if agent != "llm_gateway" and item["status"] == "completed"]
            self.assertEqual(len(completed), 4)


if __name__ == "__main__":
    unittest.main()
