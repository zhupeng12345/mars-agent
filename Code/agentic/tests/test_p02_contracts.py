from __future__ import annotations

import unittest

from agentic.platform.contracts import (
    AgentAssignment,
    AgentResult,
    DiagnosisDraft,
    DiagnosisTicket,
    EvidenceRef,
    InspectionFeatures,
    InspectionTask,
    JevRouteDecision,
    MediaManifest,
    ProcessStateSnapshot,
    ReviewDecision,
    TaskEvent,
    TraceEvent,
    VisualEvidence,
)


class P02ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = InspectionTask("sample.jpg")

    def test_task_and_result_round_trip_preserves_audit_fields(self) -> None:
        evidence = EvidenceRef(
            source_id="media-1",
            run_id=self.task.run_id,
            thread_id=self.task.thread_id,
            kind="visual_roi",
        )
        result = AgentResult(
            agent_id="vision",
            run_id=self.task.run_id,
            thread_id=self.task.thread_id,
            evidence=[evidence],
            output={"quality_status": "pass"},
        )
        payload = result.to_dict()
        restored = AgentResult.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)
        for key in ("schema_version", "run_id", "thread_id", "created_at", "implementation"):
            self.assertIn(key, payload)
        self.assertEqual(payload["evidence"][0]["source_id"], "media-1")

    def test_enums_and_source_references_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            JevRouteDecision(run_id=self.task.run_id, route="not-a-route").validate()
        with self.assertRaises(ValueError):
            JevRouteDecision(run_id=self.task.run_id, router_backend="unreported-backend").validate()
        with self.assertRaises(ValueError):
            ReviewDecision(self.task.run_id, "not-an-action", "reviewer").validate()
        evidence = EvidenceRef(
            source_id="missing", run_id=self.task.run_id, thread_id=self.task.thread_id
        )
        with self.assertRaises(ValueError):
            evidence.validate(known_source_ids={"media-1"})

    def test_process_missing_and_alignment_are_explicit(self) -> None:
        snapshot = ProcessStateSnapshot(
            run_id=self.task.run_id,
            process_missing=["potVolt"],
            alignment_quality="unknown",
        )
        features = InspectionFeatures(
            run_id=self.task.run_id,
            process_snapshot=snapshot,
            image_quality=0.9,
        )
        self.assertTrue(features.validate())
        payload = features.to_dict()
        self.assertEqual(payload["process_snapshot"]["process_missing"], ["potVolt"])
        self.assertEqual(payload["process_snapshot"]["alignment_quality"], "unknown")

    def test_process_missing_rejects_boolean_and_uses_list_boundary(self) -> None:
        snapshot = ProcessStateSnapshot(
            run_id=self.task.run_id,
            process_missing=True,
            alignment_quality="unknown",
        )
        with self.assertRaises(ValueError):
            snapshot.validate()

    def test_remaining_contracts_serialize(self) -> None:
        visual = VisualEvidence(
            evidence_id="ev-1", source_id="media-1", run_id=self.task.run_id,
            class_probs={"cold": 0.8}, risk_level="low", temporal_consistency=0.7,
        )
        manifest = MediaManifest(media_id="m-1", source_id="media-1", run_id=self.task.run_id)
        draft = DiagnosisDraft(
            run_id=self.task.run_id,
            evidence=[EvidenceRef(source_id="media-1", run_id=self.task.run_id)],
        )
        assignment = AgentAssignment(
            ticket_id="ticket-1", agent_id="vision", role="perception", run_id=self.task.run_id
        )
        ticket = DiagnosisTicket(
            ticket_id="ticket-1", run_id=self.task.run_id,
            diagnosis_draft=draft, assignments=[assignment],
        )
        event = TraceEvent(self.task.run_id, "vision", "completed")
        for item in (visual, manifest, draft, ticket, assignment, event):
            self.assertTrue(item.validate())
            self.assertIsInstance(item.to_dict(), dict)
        self.assertEqual(visual.to_dict()["class_probs"], {"cold": 0.8})
        ticket.stage = "fan_in"
        ticket.route = "full_loop"
        ticket.worker_id = "diagnosis_aggregator"
        ticket.retry_count = 1
        ticket.report_id = "report-1"
        ticket.human_decision = {"action": "approve"}
        restored_ticket = DiagnosisTicket.from_dict(ticket.to_dict())
        self.assertEqual(restored_ticket.to_dict()["report_id"], "report-1")
        task_event = TaskEvent(run_id=self.task.run_id, thread_id=self.task.thread_id, ticket_id=ticket.ticket_id, event_type="assignment.created")
        self.assertTrue(task_event.validate())

    def test_private_reasoning_is_rejected(self) -> None:
        event = TraceEvent(self.task.run_id, "vision", "completed", summary={"chain_of_thought": "secret"})
        with self.assertRaises(ValueError):
            event.to_dict()


if __name__ == "__main__":
    unittest.main()
