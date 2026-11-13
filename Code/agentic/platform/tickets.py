"""Local Ticket/Worker execution ledger.

The ledger borrows the Source -> Thread -> Ticket -> Assignment -> Worker
semantics used by modern super-agent systems, while keeping the payload
domain-specific and JSONL based for local replay.  It is an audit boundary,
not a second orchestration framework: LangGraph remains the executor.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .contracts import (
    AgentAssignment,
    AssignmentStatus,
    DiagnosisTicket,
    InspectionTask,
    TaskEvent,
    TicketStatus,
    utc_now,
)


class TicketStore:
    """Append-only local Ticket and TaskEvent store with latest-state reads."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tickets_path = self.root / "tickets.jsonl"
        self.events_path = self.root / "task_events.jsonl"
        self._lock = threading.RLock()

    @staticmethod
    def _write(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _append_ticket(self, ticket: DiagnosisTicket) -> DiagnosisTicket:
        ticket.updated_at = utc_now()
        ticket.validate()
        self._write(self.tickets_path, ticket.to_dict())
        return ticket

    def emit(self, event_type: str, *, task: Optional[InspectionTask] = None,
             ticket_id: str = "", run_id: str = "", thread_id: str = "",
             payload: Optional[Mapping[str, Any]] = None,
             agent_id: str = "orchestrator", model_profile: str = "") -> TaskEvent:
        event = TaskEvent(
            event_type=event_type,
            run_id=run_id or (task.run_id if task else ""),
            thread_id=thread_id or (task.thread_id if task else ""),
            ticket_id=ticket_id,
            payload=dict(payload or {}),
            agent_id=agent_id,
            model_profile=model_profile,
        )
        event.validate()
        with self._lock:
            self._write(self.events_path, event.to_dict())
        return event

    def create(self, task: InspectionTask, *, title: str = "fire-eye diagnosis",
               route: str = "human", priority: str = "normal",
               model_profile: str = "") -> DiagnosisTicket:
        ticket = DiagnosisTicket(
            run_id=task.run_id,
            thread_id=task.thread_id,
            title=title,
            priority=priority,
            route=route,
            stage="created",
            worker_id="orchestrator",
            model_profile=model_profile,
            implementation="agentic.tickets",
        )
        with self._lock:
            self._append_ticket(ticket)
            self.emit("task.created", task=task, ticket_id=ticket.ticket_id,
                      payload={"input_path": task.input_path, "input_mode": task.input_mode},
                      model_profile=model_profile)
        return ticket

    def get(self, ticket_id: str) -> Optional[DiagnosisTicket]:
        with self._lock:
            rows = [row for row in self._read(self.tickets_path) if row.get("ticket_id") == ticket_id]
        return DiagnosisTicket.from_dict(rows[-1]) if rows else None

    def by_run(self, run_id: str) -> Optional[DiagnosisTicket]:
        with self._lock:
            rows = [row for row in self._read(self.tickets_path) if row.get("run_id") == run_id]
        return DiagnosisTicket.from_dict(rows[-1]) if rows else None

    def assign(self, ticket: DiagnosisTicket, *, agent_id: str, role: str,
               model_profile: str = "") -> AgentAssignment:
        assignment = AgentAssignment(
            ticket_id=ticket.ticket_id,
            agent_id=agent_id,
            role=role,
            status=AssignmentStatus.ASSIGNED.value,
            model_profile=model_profile,
            run_id=ticket.run_id,
            thread_id=ticket.thread_id,
        )
        with self._lock:
            ticket.assignments.append(assignment)
            ticket.stage = "assigned"
            ticket.status = TicketStatus.ASSIGNED.value
            ticket.worker_id = agent_id
            self._append_ticket(ticket)
            self.emit("assignment.created", ticket_id=ticket.ticket_id,
                      run_id=ticket.run_id, thread_id=ticket.thread_id,
                      agent_id=agent_id, model_profile=model_profile,
                      payload={"assignment_id": assignment.assignment_id, "role": role})
        return assignment

    def update(self, ticket: DiagnosisTicket, *, stage: Optional[str] = None,
               status: Optional[str] = None, health: Optional[str] = None,
               worker_id: Optional[str] = None, attempt: Optional[int] = None,
               retry_count: Optional[int] = None, report_id: Optional[str] = None,
               human_decision: Optional[Mapping[str, Any]] = None,
               payload: Optional[Mapping[str, Any]] = None) -> DiagnosisTicket:
        with self._lock:
            if stage is not None:
                ticket.stage = str(stage)
            if status is not None:
                ticket.status = str(status)
            if health is not None:
                ticket.health = str(health)
            if worker_id is not None:
                ticket.worker_id = str(worker_id)
            if attempt is not None:
                ticket.attempt = int(attempt)
            if retry_count is not None:
                ticket.retry_count = int(retry_count)
            if report_id is not None:
                ticket.report_id = str(report_id)
            if human_decision is not None:
                ticket.human_decision = dict(human_decision)
            result = self._append_ticket(ticket)
            self.emit("ticket.updated", ticket_id=ticket.ticket_id,
                      run_id=ticket.run_id, thread_id=ticket.thread_id,
                      agent_id=ticket.worker_id,
                      payload=dict(payload or {"stage": ticket.stage, "status": ticket.status}))
            return result

    def worker_result(self, ticket: DiagnosisTicket, assignment_id: str, *,
                      status: str, attempt: int = 1, error_code: str = "") -> DiagnosisTicket:
        with self._lock:
            for assignment in ticket.assignments:
                if assignment.assignment_id == assignment_id:
                    assignment.status = status
                    assignment.attempt = int(attempt)
                    assignment.started_at = assignment.started_at or utc_now()
                    assignment.finished_at = utc_now()
                    assignment.metadata["error_code"] = error_code if error_code else None
                    break
            ticket.retry_count = max(ticket.retry_count, max(0, int(attempt) - 1))
            failed = any(item.status == AssignmentStatus.FAILED.value for item in ticket.assignments)
            all_done = bool(ticket.assignments) and all(item.status in {AssignmentStatus.COMPLETED.value, AssignmentStatus.FAILED.value} for item in ticket.assignments)
            ticket.health = "degraded" if failed else "healthy"
            ticket.stage = "fan_in" if all_done else "running"
            ticket.status = TicketStatus.WAITING_REVIEW.value if all_done else TicketStatus.RUNNING.value
            result = self._append_ticket(ticket)
            self.emit(
                "assignment.completed" if status == AssignmentStatus.COMPLETED.value else "assignment.failed",
                ticket_id=ticket.ticket_id,
                run_id=ticket.run_id,
                thread_id=ticket.thread_id,
                agent_id=next((item.agent_id for item in ticket.assignments if item.assignment_id == assignment_id), "worker"),
                payload={"assignment_id": assignment_id, "status": status, "attempt": int(attempt), "error_code": error_code},
            )
            return result

    def list(self) -> list[DiagnosisTicket]:
        with self._lock:
            rows = self._read(self.tickets_path)
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[str(row.get("ticket_id"))] = row
        return [DiagnosisTicket.from_dict(row) for row in latest.values()]

    def events(self, *, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        rows = self._read(self.events_path)
        return [row for row in rows if run_id is None or row.get("run_id") == run_id]


__all__ = ["TicketStore"]
