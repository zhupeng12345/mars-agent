"""Known-reference-error smoke for the RSI candidate guard.

The command succeeds only when the intentionally invalid reference is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.rsi.candidate import CandidateReferenceError, build_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="RSI candidate known-reference-error smoke")
    parser.add_argument("--output")
    args = parser.parse_args()
    candidate = build_candidate(
        surface="knowledge_queries",
        before={"knowledge_queries": {"default": "old"}},
        after={"knowledge_queries": {"default": "new"}},
        source_refs=["known-reference-that-is-not-in-the-snapshot"],
    )
    try:
        candidate.validate(["route-stable", "route-risk"])
    except CandidateReferenceError as exc:
        result = {"status": "expected_reference_error", "error": str(exc), "candidate_id": candidate.candidate_id}
    else:
        result = {"status": "unexpected_acceptance", "candidate_id": candidate.candidate_id}
        if args.output:
            Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
