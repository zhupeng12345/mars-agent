"""Durable human-in-the-loop pause/resume service.

The service deliberately keeps the human decision boundary outside model
reasoning.  It stores the structured draft, evidence references, route and
model metadata needed to reproduce a report, then applies one of the four
explicitly supported actions.  A JSON checkpoint is used by default so the
feature remains available when the optional LangGraph SQLite package is not
installed.
"""

from __future__ import annotations

import html
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import uuid4

from .contracts import ReviewDecision, jsonable, utc_now
from .memory import JsonCheckpointStore, _exclusive_file_lock


ALLOWED_HUMAN_ACTIONS = ("approve", "amend", "reject", "request_retake")
_AUDIT_LOCK = Lock()


class HumanReviewError(RuntimeError):
    """Base class for durable human-review errors."""


class InvalidActionError(ValueError, HumanReviewError):
    """Raised when an action is outside the HITL allowlist."""


class ReviewNotFoundError(HumanReviewError):
    """Raised when a run has no persisted review checkpoint."""


class AlreadyResumedError(HumanReviewError):
    """Raised when a completed review is resumed with a different action."""


def _default_thread(run_id: str) -> str:
    return f"inspection_default_default_{run_id}" if run_id else ""


class _AuditStore:
    """Small append-only JSONL audit store with event-id de-duplication."""

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[Dict[str, Any]] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append_once(self, event_id: str, row: Mapping[str, Any]) -> Dict[str, Any]:
        with _AUDIT_LOCK, _exclusive_file_lock(self.path):
            rows = self._read()
            for existing in rows:
                if existing.get("event_id") == event_id:
                    return dict(existing)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(row)
            payload["event_id"] = event_id
            fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for existing in rows:
                        handle.write(json.dumps(existing, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.write(json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return payload

    def read_all(self) -> list[Dict[str, Any]]:
        with _AUDIT_LOCK, _exclusive_file_lock(self.path):
            return self._read()


class HumanReviewService:
    """Persistent checkpoint, review, audit and report API.

    ``root`` may be shared by multiple service instances.  Constructing a new
    instance after a process restart reads the same checkpoints and audit log.
    ``pause`` and ``resume`` are idempotent for a ``run_id``/``thread_id``;
    duplicate resumes return the original outcome without a second audit row.
    """

    allowed_actions = ALLOWED_HUMAN_ACTIONS

    def __init__(
        self,
        root: Optional[Path | str] = None,
        *,
        checkpoint_store: Optional[Any] = None,
        checkpoint_path: Optional[Path | str] = None,
        audit_path: Optional[Path | str] = None,
        runtime_dir: Optional[Path | str] = None,
    ):
        if root is None:
            root = runtime_dir
        if checkpoint_path is not None and root is None:
            root = Path(checkpoint_path).expanduser().parent
        self.root = Path(root or os.getenv("MARS_AGENT_RUNTIME_DIR", "runtime_data")).expanduser()
        self.checkpoints = checkpoint_store or JsonCheckpointStore(checkpoint_path or (self.root / "hitl_checkpoints.json"))
        self.audit = _AuditStore(Path(audit_path).expanduser() if audit_path else self.root / "hitl_audit.jsonl")
        self._lock = Lock()

    @staticmethod
    def _thread_id(run_id: str, thread_id: Optional[str]) -> str:
        resolved = str(thread_id or _default_thread(run_id))
        if not resolved:
            raise ValueError("run_id or thread_id is required")
        return resolved

    def pause(
        self,
        run_id: str,
        *,
        thread_id: Optional[str] = None,
        draft: Optional[Mapping[str, Any]] = None,
        evidence: Any = None,
        issues: Optional[Iterable[Any]] = None,
        review: Optional[Mapping[str, Any]] = None,
        route: str = "human",
        model_metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a pending human-review payload and return its snapshot."""
        if not run_id:
            raise ValueError("run_id is required")
        resolved_thread = self._thread_id(run_id, thread_id)
        current = self.checkpoints.get(resolved_thread)
        if current is not None:
            if current.get("run_id") != run_id:
                raise HumanReviewError("thread_id already belongs to another run")
            return current
        now = utc_now()
        checkpoint = {
            "run_id": run_id,
            "thread_id": resolved_thread,
            "status": "waiting_review",
            "created_at": now,
            "updated_at": now,
            "draft": dict(draft or {}),
            "evidence": evidence if evidence is not None else [],
            "issues": list(issues or []),
            "review": dict(review or {}),
            "route": route,
            "model_metadata": dict(model_metadata or {}),
            "context": dict(context or {}),
            "allowed_actions": list(ALLOWED_HUMAN_ACTIONS),
        }
        self.checkpoints.put(resolved_thread, checkpoint)
        return checkpoint

    # LangGraph terminology and service terminology both remain available.
    interrupt = pause
    create_checkpoint = pause

    def get(self, run_id: Optional[str] = None, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        if thread_id:
            checkpoint = self.checkpoints.get(thread_id)
            if checkpoint is not None and (run_id is None or checkpoint.get("run_id") == run_id):
                return checkpoint
            raise ReviewNotFoundError(run_id or thread_id or "review")
        if run_id:
            for checkpoint in self.checkpoints.list():
                if checkpoint.get("run_id") == run_id:
                    return checkpoint
        raise ReviewNotFoundError(run_id or thread_id or "review")

    def pending(self, run_id: Optional[str] = None, *, thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            checkpoint = self.get(run_id, thread_id=thread_id)
        except ReviewNotFoundError:
            return None
        return checkpoint if checkpoint.get("status") == "waiting_review" else None

    def pending_reviews(self) -> list[Dict[str, Any]]:
        return self.checkpoints.list(status="waiting_review")

    list_pending = pending_reviews

    @staticmethod
    def _coerce_decision(
        run_id: str,
        thread_id: str,
        decision: Optional[ReviewDecision | Mapping[str, Any]],
        *,
        action: Optional[str],
        reviewer_id: str,
        reason: str,
        amendments: Optional[Mapping[str, Any]],
        evidence_ids: Optional[Iterable[str]],
    ) -> ReviewDecision:
        if isinstance(decision, ReviewDecision):
            candidate = decision
        else:
            values = dict(decision or {})
            if action is not None:
                values["action"] = action
            candidate = ReviewDecision(
                run_id=run_id,
                thread_id=thread_id,
                action=str(values.get("action", "")),
                reviewer_id=str(values.get("reviewer_id", reviewer_id)),
                reason=str(values.get("reason", reason or "")),
                amendments=dict(values.get("amendments", amendments or {})),
                evidence_ids=list(values.get("evidence_ids", evidence_ids or [])),
            )
        if candidate.run_id != run_id:
            raise ValueError("review decision run_id does not match checkpoint")
        # ReviewDecision defaults to the conventional thread derived from the
        # run.  A checkpoint may intentionally use a tenant/session thread, so
        # treat that default as unspecified and bind it to the checkpoint.
        if not candidate.thread_id or candidate.thread_id == _default_thread(run_id):
            candidate.thread_id = thread_id
        if candidate.thread_id != thread_id:
            raise ValueError("review decision thread_id does not match checkpoint")
        if candidate.action not in ALLOWED_HUMAN_ACTIONS:
            raise InvalidActionError(f"invalid human review action: {candidate.action!r}; allowed={list(ALLOWED_HUMAN_ACTIONS)}")
        candidate.validate()
        return candidate

    @staticmethod
    def _apply_decision(checkpoint: Mapping[str, Any], decision: ReviewDecision) -> Dict[str, Any]:
        draft = dict(checkpoint.get("draft") or {})
        amendments = dict(decision.amendments or {})
        if decision.action == "amend":
            draft.update(amendments)
        route = str(checkpoint.get("route") or "human")
        status = {
            "approve": "approved",
            "amend": "approved",
            "reject": "rejected",
            "request_retake": "retake_requested",
        }[decision.action]
        result = {
            "run_id": checkpoint.get("run_id", decision.run_id),
            "thread_id": checkpoint.get("thread_id", decision.thread_id),
            "status": status,
            "route": "retake" if decision.action == "request_retake" else route,
            "diagnosis": draft,
            "evidence": checkpoint.get("evidence", []),
            "review": checkpoint.get("review", {}),
            "human_review_required": False,
            "human_decision": jsonable(decision),
        }
        if amendments:
            result["amendments"] = amendments
        return result

    def _audit_for(self, checkpoint: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        decision = checkpoint.get("decision") or {}
        event_id = decision.get("decision_id")
        if not event_id:
            return None
        for event in self.audit.read_all():
            if event.get("event_id") == event_id:
                return event
        return None

    def _ensure_audit(self, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
        decision = dict(checkpoint.get("decision") or {})
        event_id = str(decision.get("decision_id") or f"review-{checkpoint.get('run_id')}-{checkpoint.get('thread_id')}")
        row = {
            "event_type": "human_review",
            "run_id": checkpoint.get("run_id", ""),
            "thread_id": checkpoint.get("thread_id", ""),
            "status": checkpoint.get("status", ""),
            "review_status": (checkpoint.get("result") or {}).get("status", checkpoint.get("status", "")),
            "action": decision.get("action", ""),
            "reviewer_id": decision.get("reviewer_id", ""),
            "reason": decision.get("reason", ""),
            "amendments": decision.get("amendments", {}),
            "evidence_ids": decision.get("evidence_ids", []),
            "route": checkpoint.get("route", ""),
            "risk_level": (checkpoint.get("draft") or {}).get("risk_level", "unknown"),
            "model_profile": (checkpoint.get("model_metadata") or {}).get("model_profile", ""),
            "decided_at": decision.get("decided_at") or checkpoint.get("updated_at", utc_now()),
        }
        context = checkpoint.get("context") if isinstance(checkpoint.get("context"), Mapping) else {}
        model_metadata = checkpoint.get("model_metadata") if isinstance(checkpoint.get("model_metadata"), Mapping) else {}
        for key in ("parent_run_id", "retake_of_run_id"):
            value = context.get(key) or model_metadata.get(key)
            if value:
                row[key] = value
        return self.audit.append_once(event_id, row)

    def resume(
        self,
        run_id: str,
        decision: Optional[ReviewDecision | Mapping[str, Any]] = None,
        *,
        action: Optional[str] = None,
        reviewer_id: str = "local_operator",
        reason: str = "",
        amendments: Optional[Mapping[str, Any]] = None,
        evidence_ids: Optional[Iterable[str]] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply one allowlisted action exactly once and return the outcome."""
        if isinstance(decision, str) and action is None:
            action, decision = decision, None
        with self._lock:
            if thread_id:
                resolved_thread = self._thread_id(run_id, thread_id)
                checkpoint = self.get(run_id, thread_id=resolved_thread)
            else:
                checkpoint = self.get(run_id)
                resolved_thread = str(checkpoint.get("thread_id") or self._thread_id(run_id, None))
            if checkpoint.get("status") == "completed":
                previous = (checkpoint.get("decision") or {}).get("action")
                requested = action or (decision.get("action") if isinstance(decision, Mapping) else getattr(decision, "action", None))
                if requested and requested != previous:
                    raise AlreadyResumedError(f"run {run_id} already resumed with action {previous}")
                self._ensure_audit(checkpoint)
                return checkpoint
            candidate = self._coerce_decision(
                run_id,
                resolved_thread,
                decision,
                action=action,
                reviewer_id=reviewer_id,
                reason=reason,
                amendments=amendments,
                evidence_ids=evidence_ids,
            )
            now = utc_now()
            decision_payload = jsonable(candidate)
            decision_payload["decision_id"] = f"review-{uuid4().hex}"
            result = self._apply_decision(checkpoint, candidate)
            updated = dict(checkpoint)
            updated.update({
                "status": "completed",
                "updated_at": now,
                "decision": decision_payload,
                "result": result,
            })
            if not self.checkpoints.update_if(resolved_thread, "waiting_review", updated):
                # Another service instance completed the row between reads.
                latest = self.get(run_id, thread_id=resolved_thread)
                if latest.get("status") == "completed":
                    return latest
                raise HumanReviewError("review checkpoint changed while resuming")
            self._ensure_audit(updated)
            return updated

    resume_review = resume

    def history(
        self,
        *,
        run_id: Optional[str] = None,
        route: Optional[str] = None,
        risk_level: Optional[str] = None,
        review_status: Optional[str] = None,
        model_profile: Optional[str] = None,
        since: Optional[str] = None,
        date: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        rows = self.audit.read_all()
        def matches(row: Mapping[str, Any]) -> bool:
            if run_id and row.get("run_id") != run_id:
                return False
            if route and row.get("route") != route:
                return False
            if risk_level and row.get("risk_level") != risk_level:
                return False
            requested_status = review_status or status
            if requested_status and row.get("review_status", row.get("status")) != requested_status and row.get("status") != requested_status:
                return False
            if model_profile and row.get("model_profile") != model_profile:
                return False
            if date and not str(row.get("decided_at", "")).startswith(date):
                return False
            if since and str(row.get("decided_at", "")) < since:
                return False
            return True
        rows = [row for row in rows if matches(row)]
        return rows[-max(1, int(limit)):]

    list_history = history

    def report(self, run_id: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = self.get(run_id, thread_id=thread_id)
        draft = checkpoint.get("draft", {})
        result = checkpoint.get("result") or {}
        final_diagnosis = result.get("diagnosis", draft) if checkpoint.get("status") == "completed" else draft
        return {
            "run_id": checkpoint.get("run_id", run_id),
            "thread_id": checkpoint.get("thread_id", thread_id or ""),
            "status": checkpoint.get("status", ""),
            "route": checkpoint.get("route", ""),
            "draft": draft,
            "final_diagnosis": final_diagnosis,
            "evidence": checkpoint.get("evidence", []),
            "review": checkpoint.get("review", {}),
            "model_metadata": checkpoint.get("model_metadata", {}),
            "decision": checkpoint.get("decision"),
            "result": checkpoint.get("result"),
            "audit": self._audit_for(checkpoint),
        }

    def export_report(
        self,
        run_id: str,
        fmt: str = "json",
        destination: Optional[Path | str] = None,
        *,
        thread_id: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        """Render a report as JSON, Markdown, HTML or a PNG snapshot."""
        report = self.report(run_id, thread_id=thread_id)
        normalized = (format or fmt).lower().lstrip(".")
        binary: Optional[bytes] = None
        if normalized == "json":
            rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        elif normalized in {"md", "markdown"}:
            rendered = self._markdown_report(report)
        elif normalized == "html":
            rendered = self._html_report(report)
        elif normalized == "png":
            if destination is None:
                raise ValueError("png export requires a destination path")
            binary = self._png_report(report)
            rendered = str(destination)
        else:
            raise ValueError("format must be json, markdown, html or png")
        if destination is not None:
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            if binary is not None:
                target.write_bytes(binary)
            else:
                target.write_text(rendered, encoding="utf-8")
        return rendered

    @staticmethod
    def _markdown_report(report: Mapping[str, Any]) -> str:
        decision = report.get("decision") or {}
        final_diagnosis = report.get("final_diagnosis") or (report.get("result") or {}).get("diagnosis") or report.get("draft", {})
        original_draft = report.get("draft", {})
        lines = [
            f"# Inspection report: {report.get('run_id', '')}",
            "",
            f"- Status: `{report.get('status', '')}`",
            f"- Route: `{report.get('route', '')}`",
            f"- Thread: `{report.get('thread_id', '')}`",
            f"- Human action: `{decision.get('action', 'pending')}`",
            f"- Reviewer: `{decision.get('reviewer_id', '')}`",
            "",
            "## Final diagnosis",
            "",
            "```json",
            json.dumps(final_diagnosis, ensure_ascii=False, indent=2),
            "```",
        ]
        if original_draft != final_diagnosis:
            lines.extend([
                "",
                "## Original draft (audit)",
                "",
                "```json",
                json.dumps(original_draft, ensure_ascii=False, indent=2),
                "```",
            ])
        lines.extend([
            "",
            "## Evidence",
            "",
            "```json",
            json.dumps(report.get("evidence", []), ensure_ascii=False, indent=2),
            "```",
        ])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _html_report(report: Mapping[str, Any]) -> str:
        payload = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
        return "<!doctype html><html><head><meta charset='utf-8'><title>Inspection report</title></head><body><pre>" + payload + "</pre></body></html>"

    @staticmethod
    def _png_report(report: Mapping[str, Any]) -> bytes:
        """Render a compact per-run review snapshot for the workbench/PPT."""
        import io
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.font_manager import FontProperties
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(11, 6.2), dpi=150)
        axis.axis("off")
        decision = report.get("decision") or {}
        draft = report.get("final_diagnosis") or (report.get("result") or {}).get("diagnosis") or report.get("draft") or {}
        lines = [
            f"铝电解火眼诊断报告  {report.get('run_id', '')}",
            f"route={report.get('route', '')}    status={report.get('status', '')}    human={decision.get('action', 'pending')}",
            f"risk={draft.get('risk_level', 'unknown')}    confidence={draft.get('confidence', '—')}",
            "",
            str(draft.get("summary", "Evidence aggregation requires operator review.")),
            "",
            "发现：" + "；".join(str(item) for item in (draft.get("findings") or [])[:4]),
            "复核：" + "；".join(str(item) for item in ((report.get("review") or {}).get("issues") or [])[:4]),
        ]
        font_candidates = [
            Path(os.getenv("MARS_AGENT_REPORT_FONT", "")),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
        font_path = next((candidate for candidate in font_candidates if str(candidate) not in {"", "."} and candidate.is_file()), None)
        font = FontProperties(fname=str(font_path)) if font_path else FontProperties(family="DejaVu Sans")
        axis.text(0.03, 0.94, "\n".join(lines), va="top", ha="left", fontsize=13, fontproperties=font, wrap=True)
        buffer = io.BytesIO()
        figure.tight_layout()
        figure.savefig(buffer, format="png", facecolor="white", bbox_inches="tight")
        plt.close(figure)
        return buffer.getvalue()


__all__ = [
    "ALLOWED_HUMAN_ACTIONS", "AlreadyResumedError", "HumanReviewError",
    "HumanReviewService", "InvalidActionError", "ReviewNotFoundError",
]
