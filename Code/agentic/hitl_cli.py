"""CLI for local HITL checkpoints, audit history and report export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.platform.hitl import HumanReviewService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MARS-Agent human review service")
    parser.add_argument("--runtime-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    pending = sub.add_parser("pending", help="list pending reviews")
    pending.add_argument("--run-id")

    resume = sub.add_parser("resume", help="apply one allowlisted action")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--action", choices=["approve", "amend", "reject", "request_retake"], required=True)
    resume.add_argument("--reviewer-id", default="local_operator")
    resume.add_argument("--reason", default="")
    resume.add_argument("--thread-id")
    resume.add_argument("--amendments-json", default="{}")

    history = sub.add_parser("history", help="query human review audit")
    history.add_argument("--run-id")
    history.add_argument("--route")
    history.add_argument("--risk-level")
    history.add_argument("--review-status")
    history.add_argument("--model-profile")

    report = sub.add_parser("export", help="export a report")
    report.add_argument("--run-id", required=True)
    report.add_argument("--format", default="json", choices=["json", "markdown", "html"])
    report.add_argument("--output", required=True)
    report.add_argument("--thread-id")

    args = parser.parse_args(argv)
    service = HumanReviewService(Path(args.runtime_dir) if args.runtime_dir else None)
    if args.command == "pending":
        rows = [service.pending(args.run_id)] if args.run_id and service.pending(args.run_id) else service.pending_reviews()
        print(json.dumps([row for row in rows if row], ensure_ascii=False, indent=2))
    elif args.command == "resume":
        amendments = json.loads(args.amendments_json)
        result = service.resume(
            args.run_id,
            action=args.action,
            reviewer_id=args.reviewer_id,
            reason=args.reason,
            amendments=amendments,
            thread_id=args.thread_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "history":
        rows = service.history(run_id=args.run_id, route=args.route, risk_level=args.risk_level, review_status=args.review_status, model_profile=args.model_profile)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        service.export_report(args.run_id, args.format, args.output, thread_id=args.thread_id)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
