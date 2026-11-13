"""Streamlit-compatible helpers with no Streamlit import at module load time."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hitl import ALLOWED_HUMAN_ACTIONS, HumanReviewService


def pending_review(service: HumanReviewService, run_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return a UI-safe pending payload for a run or the first pending task."""
    if run_id:
        return service.pending(run_id)
    rows = service.pending_reviews()
    return rows[0] if rows else None


def apply_review(
    service: HumanReviewService,
    run_id: str,
    action: str,
    *,
    reviewer_id: str = "local_operator",
    reason: str = "",
    amendments: Optional[Mapping[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Submit a button/text-box decision from a Streamlit callback."""
    return service.resume(
        run_id,
        action=action,
        reviewer_id=reviewer_id,
        reason=reason,
        amendments=amendments,
        thread_id=thread_id,
    )


__all__ = ["ALLOWED_HUMAN_ACTIONS", "apply_review", "pending_review"]
