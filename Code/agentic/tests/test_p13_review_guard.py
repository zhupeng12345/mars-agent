import unittest

from agentic.agents.review_guard import review_draft
from agentic.platform.contracts import DiagnosisDraft, EvidenceRef


class ReviewGuardTests(unittest.TestCase):
    def test_unknown_source_is_rejected(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            findings=["Observed visual pattern"],
            evidence=[EvidenceRef(source_id="missing", run_id="run-1")],
        )
        result = review_draft(draft, known_source_ids={"media-1"})
        self.assertEqual(result["action"], "request_more_evidence")
        self.assertTrue(result["issues"])

    def test_numeric_superheat_without_measurements_is_blocked(self) -> None:
        for claim in ("superheat 12 C", "temperature is 12°F", "ΔT=12", "过热度12"):
            with self.subTest(claim=claim):
                draft = DiagnosisDraft(
                    run_id="run-1",
                    findings=[claim],
                    evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
                )
                result = review_draft(draft, known_source_ids={"media-1"})
                self.assertEqual(result["action"], "request_more_evidence")
                self.assertIn("数值过热度表达缺少温度、成分和初晶温度实测依据", result["issues"])

    def test_placeholder_measurements_do_not_unlock_numeric_superheat(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            findings=["过热度 12℃"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        result = review_draft(
            draft,
            known_source_ids={"media-1"},
            measurements={
                "temperature": "unknown",
                "composition": "N/A",
                "initial_crystallization_temperature": "?",
            },
        )
        self.assertFalse(result["measurements_used"])
        self.assertIn("数值过热度表达缺少温度、成分和初晶温度实测依据", result["issues"])

    def test_non_finite_measurements_do_not_unlock_numeric_superheat(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            findings=["Superheat is 12 C"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        valid = {
            "temperature": 960.0,
            "composition": 4.2,
            "initial_crystallization_temperature": 948.0,
        }
        for field, value in (("temperature", float("nan")), ("composition", float("inf"))):
            with self.subTest(field=field, value=value):
                measurements = {**valid, field: value}
                result = review_draft(draft, known_source_ids={"media-1"}, measurements=measurements)
                self.assertFalse(result["measurements_used"])
                self.assertIn("数值过热度表达缺少温度、成分和初晶温度实测依据", result["issues"])

    def test_explicit_numeric_measurements_allow_numeric_interpretation(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            requires_human_review=False,
            findings=["过热度 12℃"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        result = review_draft(
            draft,
            known_source_ids={"media-1"},
            measurements={
                "temperature": 960.0,
                "composition": 4.2,
                "initial_crystallization_temperature": 948.0,
            },
        )
        self.assertEqual(result["action"], "pass")
        self.assertTrue(result["measurements_used"])

    def test_empty_source_registry_cannot_validate_forged_citation(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            requires_human_review=False,
            findings=["Observed visual pattern"],
            source_ids=["forged-source"],
        )
        result = review_draft(draft, known_source_ids=set())
        self.assertEqual(result["action"], "request_more_evidence")
        self.assertIn("证据来源注册表为空，无法验证引用来源", result["issues"])

    def test_english_control_instruction_requires_human(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            recommendations=["Increase cell voltage and increase feed"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        result = review_draft(draft, known_source_ids={"media-1"})
        self.assertEqual(result["action"], "human_review")

    def test_fallback_result_requires_human_review(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            findings=["Observed visual pattern"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
            metadata={"llm_status": "fallback_fake_json"},
        )
        result = review_draft(draft, known_source_ids={"media-1"})
        self.assertEqual(result["action"], "human_review")

    def test_control_instruction_requires_human(self) -> None:
        draft = DiagnosisDraft(run_id="run-1", recommendations=["adjust voltage and add feed"])
        result = review_draft(draft)
        self.assertEqual(result["action"], "human_review")

    def test_stop_operation_language_requires_human_review(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            recommendations=["停止操作并等待现场复核"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        result = review_draft(draft, known_source_ids={"media-1"})
        self.assertEqual(result["action"], "human_review")

    def test_source_less_findings_request_more_evidence(self) -> None:
        draft = DiagnosisDraft(run_id="run-1", findings=["Observed visual pattern"])
        result = review_draft(draft)
        self.assertEqual(result["action"], "request_more_evidence")
        self.assertIn("missing_traceable_source_citation", result["issues"])

    def test_supported_visual_draft_can_pass_guard(self) -> None:
        draft = DiagnosisDraft(
            run_id="run-1",
            requires_human_review=False,
            findings=["Visible brightness is stable"],
            evidence=[EvidenceRef(source_id="media-1", run_id="run-1")],
        )
        result = review_draft(draft, known_source_ids={"media-1"})
        self.assertEqual(result["action"], "pass")
        self.assertTrue(result["approved"])


if __name__ == "__main__":
    unittest.main()
