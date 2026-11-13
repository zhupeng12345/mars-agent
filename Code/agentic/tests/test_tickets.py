import tempfile
import unittest
from pathlib import Path

from agentic.platform.contracts import InspectionTask
from agentic.platform.tickets import TicketStore


class TicketStoreTests(unittest.TestCase):
    def test_ticket_assignment_and_worker_lifecycle_are_replayable(self):
        with tempfile.TemporaryDirectory() as root:
            store = TicketStore(Path(root))
            task = InspectionTask("fire-eye.mp4")
            ticket = store.create(task, route="full_loop", model_profile="fake_json")
            assignment = store.assign(ticket, agent_id="visual_review", role="visual evidence")
            stored = store.get(ticket.ticket_id)
            self.assertIsNotNone(stored)
            stored = store.worker_result(stored, assignment.assignment_id, status="completed", attempt=2)
            self.assertEqual(stored.stage, "fan_in")
            self.assertEqual(stored.retry_count, 1)
            self.assertEqual(store.by_run(task.run_id).ticket_id, ticket.ticket_id)
            self.assertGreaterEqual(len(store.events(run_id=task.run_id)), 3)


if __name__ == "__main__":
    unittest.main()
