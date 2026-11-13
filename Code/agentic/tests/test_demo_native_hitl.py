from __future__ import annotations

import tempfile
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import cv2
import numpy as np

from agentic.platform.hitl import AlreadyResumedError, HumanReviewService
from agentic.platform.tickets import TicketStore
from agentic.scripts.demo_run import load_demo_run, resume_demo_review, run_demo


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class NativeDemoHumanReviewTests(unittest.TestCase):
    def _run(self, root: Path, action: str):
        code_root = Path(__file__).resolve().parents[2]
        image_path = root / "inspection.png"
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[48:208, 48:208] = (90, 140, 210)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        return run_demo(
            code_root,
            image_path,
            output=root / f"{action}.json",
            no_model=True,
            runtime_dir=root / "runtime",
            run_id=f"native-{action}",
            llm_profile="fake_json",
            native_hitl=True,
        )

    def test_native_pause_reload_resume_all_actions_and_idempotency(self):
        for action in ("approve", "amend", "reject", "request_retake"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                demo = self._run(root, action)
                runtime = root / "runtime"
                run_id = demo["run_id"]

                self.assertEqual(demo["human_review_backend"], "langgraph_sqlite_interrupt")
                self.assertTrue(demo["graph"].get("__interrupt__"))
                self.assertEqual(HumanReviewService(runtime).pending(run_id)["status"], "waiting_review")

                # Simulate a page refresh: reconstruct all state from disk.
                restored = load_demo_run(runtime, run_id)
                self.assertEqual(restored["thread_id"], demo["thread_id"])
                amended = {"operator_note": "reviewed"} if action == "amend" else {}
                completed = resume_demo_review(
                    runtime,
                    run_id,
                    action=action,
                    reviewer_id="test_operator",
                    reason="checked",
                    amendments=amended,
                )
                expected = {"approve": "approved", "amend": "approved", "reject": "rejected", "request_retake": "retake_requested"}[action]
                self.assertEqual(completed["graph"]["result"]["status"], expected)
                self.assertEqual(completed["human_review"]["decision"]["action"], action)
                if action == "amend":
                    final_diagnosis = HumanReviewService(runtime).report(run_id)["final_diagnosis"]
                    self.assertEqual(completed["case_candidate"]["conclusion"]["diagnosis"], final_diagnosis)
                if action in {"reject", "request_retake"}:
                    self.assertEqual(completed["case_candidate"]["state"], "deprecated")
                self.assertEqual(len(HumanReviewService(runtime).history(run_id=run_id)), 1)
                self.assertEqual(TicketStore(runtime).by_run(run_id).status, "completed")
                run_records = [item for item in jsonl(runtime / "runs.jsonl") if item.get("run_id") == run_id]
                self.assertEqual(run_records[-1]["status"], expected)
                event_nodes = [item.get("node") for item in jsonl(runtime / "events.jsonl") if item.get("run_id") == run_id]
                self.assertEqual(event_nodes.count("human_review"), 1)
                self.assertEqual(event_nodes.count("finalize"), 1)

                duplicate = resume_demo_review(runtime, run_id, action=action, reviewer_id="retry")
                self.assertEqual(duplicate["human_review"]["decision"]["decision_id"], completed["human_review"]["decision"]["decision_id"])
                self.assertEqual(len(HumanReviewService(runtime).history(run_id=run_id)), 1)
                with self.assertRaises(AlreadyResumedError):
                    resume_demo_review(runtime, run_id, action="reject" if action != "reject" else "approve")

    def test_native_checkpoint_resumes_from_a_new_python_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = self._run(root, "process-restart")
            runtime = root / "runtime"
            code_root = Path(__file__).resolve().parents[2]
            script = (
                "import sys; from agentic.scripts.demo_run import resume_demo_review; "
                "result=resume_demo_review(sys.argv[1], sys.argv[2], action='approve', "
                "reviewer_id='restart_operator', reason='process restart'); "
                "print(result['graph']['status'])"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script, str(runtime), demo["run_id"]],
                cwd=code_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stdout.strip(), "approved")
            restored = load_demo_run(runtime, demo["run_id"])
            self.assertEqual(restored["graph"]["result"]["status"], "approved")
            self.assertEqual(len(HumanReviewService(runtime).history(run_id=demo["run_id"])), 1)

    def test_conflicting_concurrent_reviewers_commit_exactly_one_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = self._run(root, "concurrent-review")
            runtime = root / "runtime"
            start = Barrier(2)

            def submit(action):
                start.wait(timeout=5)
                try:
                    result = resume_demo_review(
                        runtime,
                        demo["run_id"],
                        action=action,
                        reviewer_id=f"reviewer-{action}",
                        reason="concurrent review test",
                    )
                    return ("completed", result["human_review"]["decision"]["action"])
                except AlreadyResumedError:
                    return ("conflict", action)

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(submit, ("approve", "reject")))

            self.assertEqual(sum(kind == "completed" for kind, _ in outcomes), 1)
            self.assertEqual(sum(kind == "conflict" for kind, _ in outcomes), 1)
            audit = HumanReviewService(runtime).history(run_id=demo["run_id"])
            self.assertEqual(len(audit), 1)
            self.assertIn(audit[0]["action"], {"approve", "reject"})


if __name__ == "__main__":
    unittest.main()
