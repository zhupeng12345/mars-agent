import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.orchestration.diagnosis_graph import run_diagnosis_graph
from agentic.platform.contracts import InspectionFeatures
from agentic.platform.hitl import AlreadyResumedError, HumanReviewService, InvalidActionError
from agentic.platform.checkpoint import build_sqlite_checkpointer

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    InMemorySaver = None


class HitlPersistenceTests(unittest.TestCase):
    def _service(self, root: str) -> HumanReviewService:
        return HumanReviewService(Path(root))

    def test_pause_survives_service_restart_and_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._service(root)
            first.pause("run-1", thread_id="tenant-session-run-1", draft={"risk_level": "low"}, evidence=[{"source_id": "frame-1"}], route="human")
            restarted = self._service(root)
            self.assertEqual(restarted.pending("run-1")["status"], "waiting_review")
            outcome = restarted.resume("run-1", action="approve", reviewer_id="operator", reason="checked")
            duplicate = self._service(root).resume("run-1", action="approve")
            self.assertEqual(outcome["result"]["status"], "approved")
            self.assertEqual(duplicate["decision"]["decision_id"], outcome["decision"]["decision_id"])
            self.assertEqual(len(self._service(root).history(run_id="run-1")), 1)
            with self.assertRaises(AlreadyResumedError):
                restarted.resume("run-1", action="reject")

    def test_invalid_action_does_not_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            service = self._service(root)
            service.pause("run-invalid", draft={})
            with self.assertRaises(InvalidActionError):
                service.resume("run-invalid", action="escalate")
            self.assertEqual(service.pending("run-invalid")["status"], "waiting_review")

    def test_all_four_actions_are_audited(self):
        for action in ("approve", "amend", "reject", "request_retake"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as root:
                service = self._service(root)
                service.pause(f"run-{action}", draft={"summary": "draft"}, route="full_loop")
                outcome = service.resume(
                    f"run-{action}",
                    action=action,
                    reviewer_id="operator",
                    reason=f"reason-{action}",
                    amendments={"summary": "amended"} if action == "amend" else {},
                )
                self.assertEqual(outcome["decision"]["action"], action)
                self.assertEqual(len(service.history(run_id=f"run-{action}")), 1)
                self.assertEqual(service.report(f"run-{action}")["audit"]["action"], action)

    def test_graph_can_pause_through_service_api(self):
        with tempfile.TemporaryDirectory() as root:
            features = InspectionFeatures(
                run_id="run-graph-hitl",
                visual_confidence=0.9,
                risk_score=0.1,
                knowledge_coverage=1.0,
                case_similarity=0.9,
                novelty_score=0.1,
                features={"quality_status": "pass", "target_present": True, "case_applicable": True},
            )
            result = run_diagnosis_graph(features, hitl_service=self._service(root))
            self.assertEqual(result["status"], "waiting_review")
            self.assertEqual(self._service(root).pending("run-graph-hitl")["thread_id"], features.thread_id)

    def test_report_exports_json_markdown_and_html(self):
        with tempfile.TemporaryDirectory() as root:
            service = self._service(root)
            service.pause("run-export", draft={"summary": "original draft"})
            service.resume("run-export", action="amend", amendments={"summary": "human amended diagnosis"})
            report = service.report("run-export")
            self.assertEqual(report["draft"]["summary"], "original draft")
            self.assertEqual(report["final_diagnosis"]["summary"], "human amended diagnosis")
            for fmt, marker in (("json", '"run_id"'), ("markdown", "# Inspection report"), ("html", "<!doctype html>")):
                with self.subTest(fmt=fmt):
                    rendered = service.export_report("run-export", fmt)
                    self.assertIn(marker, rendered)
                    if fmt == "markdown":
                        self.assertIn("human amended diagnosis", rendered)
                        self.assertIn("original draft", rendered)
                    elif fmt in {"json", "html"}:
                        self.assertIn("human amended diagnosis", rendered)

            from matplotlib.axes import Axes

            captured = []
            original_text = Axes.text

            def capture_text(axis, *args, **kwargs):
                if len(args) >= 3:
                    captured.append(str(args[2]))
                return original_text(axis, *args, **kwargs)

            with patch.object(Axes, "text", autospec=True, side_effect=capture_text):
                png_path = Path(root) / "amended-report.png"
                service.export_report("run-export", "png", destination=png_path)
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(any("human amended diagnosis" in value for value in captured))

    @unittest.skipIf(InMemorySaver is None, "langgraph checkpointer is not installed")
    def test_native_langgraph_interrupt_and_command_resume(self):
        features = InspectionFeatures(
            run_id="run-native-interrupt",
            visual_confidence=0.9,
            risk_score=0.1,
            knowledge_coverage=1.0,
            case_similarity=0.9,
            novelty_score=0.1,
            features={"quality_status": "pass", "target_present": True, "case_applicable": True},
        )
        checkpointer = InMemorySaver()
        paused = run_diagnosis_graph(features, checkpointer=checkpointer, human_review=True)
        self.assertTrue(paused.get("__interrupt__"))
        resumed = run_diagnosis_graph(
            features,
            checkpointer=checkpointer,
            human_review=True,
            resume={"action": "approve", "reviewer_id": "native-operator", "reason": "checked"},
        )
        self.assertEqual(resumed.get("result", {}).get("status"), "approved")
        self.assertEqual(resumed.get("human_decision", {}).get("action"), "approve")

    @unittest.skipUnless(build_sqlite_checkpointer is not None, "SQLite checkpointer factory unavailable")
    def test_native_sqlite_interrupt_survives_reopen(self):
        with tempfile.TemporaryDirectory() as root:
            features = InspectionFeatures(
                run_id="run-native-sqlite",
                visual_confidence=0.9,
                risk_score=0.1,
                knowledge_coverage=1.0,
                case_similarity=0.9,
                novelty_score=0.1,
                features={"quality_status": "pass", "target_present": True, "case_applicable": True},
            )
            path = Path(root) / "checkpoints.sqlite"
            saver = build_sqlite_checkpointer(path)
            paused = run_diagnosis_graph(features, checkpointer=saver, human_review=True)
            self.assertTrue(paused.get("__interrupt__"))
            saver.conn.close()
            reopened = build_sqlite_checkpointer(path)
            resumed = run_diagnosis_graph(
                features,
                checkpointer=reopened,
                human_review=True,
                resume={"action": "approve", "reviewer_id": "sqlite-operator", "reason": "reopened"},
            )
            self.assertEqual(resumed.get("result", {}).get("status"), "approved")
            reopened.conn.close()


if __name__ == "__main__":
    unittest.main()
