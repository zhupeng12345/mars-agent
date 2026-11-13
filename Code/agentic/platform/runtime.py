"""Durable, local run records and controlled human-feedback memory.

The reference EduAgent keeps state behind an orchestrator and a checkpointer.
This smaller local runtime applies the same idea without requiring PostgreSQL,
Redis or Milvus before the visual workflow is ready for them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional

from .contracts import InspectionTask, ReviewDecision, jsonable, utc_now
from .hitl import HumanReviewService


def _default_runtime_dir() -> Path:
    configured = os.getenv("MARS_AGENT_RUNTIME_DIR", "")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "runtime_data"


class JsonlStore:
    """Append-only local store; replaceable by PostgreSQL later."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_all(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows


class CaseMemory:
    """Candidate memory with an explicit human-review promotion boundary."""

    def __init__(self, root: Optional[Path] = None):
        root = root or _default_runtime_dir()
        self.candidates = JsonlStore(root / "case_candidates.jsonl")
        self.decisions = JsonlStore(root / "review_decisions.jsonl")
        self.approved = JsonlStore(root / "approved_cases.jsonl")

    def add_candidate(self, run_id: str, result: Dict[str, Any]) -> None:
        self.candidates.append({
            "run_id": run_id,
            "status": "candidate_pending_review",
            "created_at": utc_now(),
            "result": result,
        })

    def record_decision(self, decision: ReviewDecision) -> None:
        payload = jsonable(decision)
        existing = self.decisions.read_all()
        fingerprint = (
            payload.get("run_id"), payload.get("thread_id"), payload.get("action"),
            payload.get("reviewer_id"), payload.get("decided_at"),
        )
        if any((row.get("run_id"), row.get("thread_id"), row.get("action"), row.get("reviewer_id"), row.get("decided_at")) == fingerprint for row in existing):
            return
        self.decisions.append(payload)
        if decision.action in {"approve", "amend"}:
            self.approved.append({
                "run_id": decision.run_id,
                "status": "approved_case",
                "approved_at": decision.decided_at,
                "reviewer_id": decision.reviewer_id,
                "reason": decision.reason,
                "amendments": decision.amendments,
            })

    def list_candidates(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self.candidates.read_all()[-max(1, limit):]


class InspectionRuntime:
    """Orchestration shell for traces, run summaries, and case feedback."""

    def __init__(self, root: Optional[Path] = None):
        root = root or _default_runtime_dir()
        self.root = root
        self.runs = JsonlStore(root / "runs.jsonl")
        self.events = JsonlStore(root / "events.jsonl")
        self.case_memory = CaseMemory(root)
        self.hitl = HumanReviewService(root)

    def start(self, task: InspectionTask) -> None:
        self.runs.append({
            "run_id": task.run_id,
            "status": "running",
            "started_at": utc_now(),
            "task": jsonable(task),
        })

    def record_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def finish(
        self,
        task: InspectionTask,
        result: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        status = "error" if error else str((result or {}).get("status", "completed"))
        self.runs.append({
            "run_id": task.run_id,
            "status": status,
            "finished_at": utc_now(),
            "error": error,
            "result": result or {},
        })
        if result is not None:
            self.case_memory.add_candidate(task.run_id, result)

    def record_review(self, decision: ReviewDecision) -> None:
        self.case_memory.record_decision(decision)

    def pause_review(self, run_id: str, **payload: Any) -> Dict[str, Any]:
        """Persist a review interruption for UI and service callers."""
        return self.hitl.pause(run_id, **payload)

    def resume_review(self, run_id: str, decision: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Apply an allowlisted human action and return its durable outcome."""
        outcome = self.hitl.resume(run_id, decision, **kwargs)
        decision_payload = outcome.get("decision") if isinstance(outcome, dict) else None
        if decision_payload:
            try:
                self.case_memory.record_decision(ReviewDecision(
                    run_id=str(decision_payload.get("run_id", run_id)),
                    thread_id=str(decision_payload.get("thread_id", "")),
                    action=str(decision_payload.get("action", "")),
                    reviewer_id=str(decision_payload.get("reviewer_id", "")),
                    reason=str(decision_payload.get("reason", "")),
                    amendments=dict(decision_payload.get("amendments", {})),
                    evidence_ids=list(decision_payload.get("evidence_ids", [])),
                    decided_at=str(decision_payload.get("decided_at", utc_now())),
                ))
            except Exception:
                # The HITL checkpoint/audit is authoritative; legacy case
                # memory is best-effort for backwards compatibility.
                pass
        return outcome

    def pending_reviews(self) -> list[Dict[str, Any]]:
        return self.hitl.pending_reviews()

    def review_history(self, **filters: Any) -> list[Dict[str, Any]]:
        return self.hitl.history(**filters)

    def export_report(self, run_id: str, fmt: str = "json", destination: Optional[Path] = None, **kwargs: Any) -> str:
        return self.hitl.export_report(run_id, fmt, destination, **kwargs)

    def recent_runs(self, limit: int = 50) -> Iterable[Dict[str, Any]]:
        return self.runs.read_all()[-max(1, limit):]
