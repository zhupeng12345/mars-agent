"""Durable review records for precomputed presentation traces.

These records are isolated from production review checkpoints and case search.
Presentation fixtures can create candidate-only memory entries after approval;
they can never become active cases through this adapter.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .case_memory import SpecialCaseMemory
from .hitl import ALLOWED_HUMAN_ACTIONS, AlreadyResumedError, InvalidActionError
from .memory import _exclusive_file_lock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PresentationReviewStore:
    """Persist one idempotent review per flow-validation presentation run."""

    def __init__(self, runtime_root: Path | str):
        self.runtime_root = Path(runtime_root).expanduser()
        self.root = self.runtime_root / "presentation_demo"
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "review_audit.jsonl"
        self.case_memory = SpecialCaseMemory(self.root / "case_memory")

    def _review_path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", str(run_id)):
            raise ValueError("run_id contains unsupported characters")
        return self.root / "reviews" / f"{run_id}.json"

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self._review_path(run_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("run_id") != run_id:
            raise ValueError("persisted presentation review run_id does not match")
        return payload

    def apply(self, demo: Mapping[str, Any], action: str, amendment: str = "") -> dict[str, Any]:
        if action not in ALLOWED_HUMAN_ACTIONS:
            raise InvalidActionError(f"invalid presentation review action: {action!r}")
        run_id = str(demo.get("run_id", ""))
        thread_id = str(demo.get("thread_id", ""))
        path = self._review_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        amendments = {"operator_note": amendment} if amendment else {}

        with _exclusive_file_lock(path):
            if path.exists():
                updated = json.loads(path.read_text(encoding="utf-8"))
                existing = dict(updated.get("human_review") or {})
                if existing.get("action") != action or existing.get("amendments", {}) != amendments:
                    raise AlreadyResumedError(f"presentation run {run_id} already reviewed with a different decision")
            else:
                updated = json.loads(json.dumps(demo, ensure_ascii=False))
                review = dict(updated.get("human_review") or {})
                review.update({
                    "status": "waiting_for_retake" if action == "request_retake" else "completed",
                    "action": action,
                    "reviewer_id": "presentation_operator",
                    "reason": "PPT presentation review",
                    "amendments": amendments,
                    "decided_at": _utc_now(),
                    "audit_status": "flow_validation",
                })
                updated["human_review"] = review
                graph = dict(updated.get("graph") or {})
                graph["status"] = review["status"]
                result = dict(graph.get("result") or {})
                result["review"] = review
                result.setdefault("diagnosis", updated.get("diagnosis", {}))
                graph["result"] = result
                updated["graph"] = graph
                if action in {"approve", "amend"}:
                    candidate = self._candidate_for(updated)
                    if candidate is not None:
                        updated["case_candidate"] = candidate
                self._atomic_json(path, updated)

            review = dict(updated.get("human_review") or {})
            self._append_audit_once({
                "event_id": f"presentation-review-{run_id}",
                "run_id": run_id,
                "thread_id": thread_id,
                "action": action,
                "reviewer_id": review.get("reviewer_id", "presentation_operator"),
                "case_id": (updated.get("case_candidate") or {}).get("case_id"),
                "status": "flow_validation",
                "decided_at": review.get("decided_at", _utc_now()),
            })
            return updated

    def _candidate_for(self, demo: Mapping[str, Any]) -> dict[str, Any] | None:
        if demo.get("schema_version") != "presentation-demo-1":
            return None
        route = dict(demo.get("route_decision") or (demo.get("graph") or {}).get("route_decision") or {})
        route_name = str(route.get("route", ""))
        if route_name not in {"simple_loop", "full_loop"}:
            return None
        visual = dict(demo.get("visual") or demo.get("visual_evidence") or {})
        source_id = str((visual.get("evidence_ids") or [""])[0] or f"presentation:{demo.get('case_id', 'fixture')}")
        features = {
            key: float(route[key])
            for key in ("confidence", "risk_score", "novelty_score", "evidence_conflict")
            if isinstance(route.get(key), (int, float)) and not isinstance(route.get(key), bool)
        }
        if not features:
            features = {"confidence": 0.0}
        run_id = str(demo.get("run_id", ""))
        case_id = SpecialCaseMemory.case_id(source_id, features)
        existing = next((row for row in self.case_memory._snapshot() if row.get("case_id") == case_id and row.get("run_id") == run_id), None)
        if existing is not None:
            final_diagnosis = dict((demo.get("graph") or {}).get("result", {}).get("diagnosis") or demo.get("diagnosis", {}))
            conclusion = {
                "route": route_name,
                "diagnosis": final_diagnosis,
                "human_review": demo.get("human_review", {}),
                "claim_status": "flow_validation",
                "fixture_case_id": demo.get("case_id", ""),
            }
            return self.case_memory.update_candidate_conclusion(
                case_id,
                conclusion,
                reviewer_id=str((demo.get("human_review") or {}).get("reviewer_id", "presentation_operator")),
                reason="presentation review finalized candidate conclusion",
            )
        final_diagnosis = dict((demo.get("graph") or {}).get("result", {}).get("diagnosis") or demo.get("diagnosis", {}))
        return self.case_memory.add_candidate(
            source_id=source_id,
            features=features,
            conclusion={
                "route": route_name,
                "diagnosis": final_diagnosis,
                "human_review": demo.get("human_review", {}),
                "claim_status": "flow_validation",
                "fixture_case_id": demo.get("case_id", ""),
            },
            run_id=run_id,
            applicable_when=("presentation_fixture",),
        )

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _append_audit_once(self, event: Mapping[str, Any]) -> None:
        with _exclusive_file_lock(self.audit_path):
            rows = []
            if self.audit_path.exists():
                for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
            event_id = event.get("event_id")
            if any(row.get("event_id") == event_id for row in rows):
                return
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["PresentationReviewStore"]
