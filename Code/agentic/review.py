"""Record an explicit human decision for a local inspection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.platform import InspectionRuntime
from agentic.platform.contracts import ReviewDecision


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an MARS-Agent human review decision.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--action", choices=["approve", "reject", "request_retake", "amend"], required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--thread-id")
    parser.add_argument("--amendments-json", default="{}")
    parser.add_argument("--runtime-dir")
    args = parser.parse_args()

    runtime = InspectionRuntime(Path(args.runtime_dir) if args.runtime_dir else None)
    pending = runtime.hitl.pending(args.run_id, thread_id=args.thread_id)
    if pending is not None:
        runtime.resume_review(
            args.run_id,
            action=args.action,
            reviewer_id=args.reviewer_id,
            reason=args.reason,
            amendments=json.loads(args.amendments_json),
            thread_id=args.thread_id,
        )
    else:
        runtime.record_review(ReviewDecision(
            run_id=args.run_id,
            action=args.action,
            reviewer_id=args.reviewer_id,
            reason=args.reason,
            thread_id=args.thread_id or "",
        ))
    print(f"已记录人工决策: {args.run_id} -> {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
