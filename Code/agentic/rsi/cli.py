"""Command-line entry point for the controlled offline RSI loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .candidate import Candidate, CandidateReferenceError, build_candidate
from .evaluator import OfflineReplayer, ReplayBundle
from .gate import GateConfig, evaluate_gate
from .publisher import Publisher
from .records import RsiRecordStore


def _json(path_or_value: str) -> Any:
    path = Path(path_or_value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(path_or_value)


def _candidate_from_args(args: argparse.Namespace) -> Candidate:
    before, after = _json(args.before), _json(args.after)
    return build_candidate(
        surface=args.surface,
        before=before,
        after=after,
        source_run_ids=args.source_run_id,
        parent_version=args.parent_version,
        generator_model=args.generator_model,
        source_refs=args.source_ref,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MARS-Agent controlled offline RSI")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="copy runtime/evaluation observations into RSI records")
    collect.add_argument("--root")
    collect.add_argument("--runtime-dir")
    collect.add_argument("--evaluation-dir")

    candidate = sub.add_parser("candidate", help="generate a declarative mutable-surface candidate")
    candidate.add_argument("--surface", required=True)
    candidate.add_argument("--before", required=True, help="JSON object or file")
    candidate.add_argument("--after", required=True, help="JSON object or file")
    candidate.add_argument("--source-run-id", action="append", default=[])
    candidate.add_argument("--source-ref", action="append", default=[])
    candidate.add_argument("--parent-version", default="base")
    candidate.add_argument("--generator-model", default="offline-deterministic")
    candidate.add_argument("--known-source-id", action="append", default=[])

    replay = sub.add_parser("replay", help="run train/frozen/sealed offline replay")
    replay.add_argument("--candidate", required=True)
    replay.add_argument("--train", required=True)
    replay.add_argument("--frozen", required=True)
    replay.add_argument("--sealed", required=True)
    replay.add_argument("--output")

    gate = sub.add_parser("gate", help="evaluate quality/safety/efficiency gates")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--proposed", required=True)
    gate.add_argument("--min-quality", type=float, default=0.50)
    gate.add_argument("--min-safety", type=float, default=0.95)
    gate.add_argument("--min-efficiency", type=float, default=0.0)

    approve = sub.add_parser("approve", help="record human approval")
    approve.add_argument("--root", required=True)
    approve.add_argument("--candidate", required=True)
    approve.add_argument("--reviewer-id", required=True)
    approve.add_argument("--reason", default="")
    approve.add_argument("--gate-result", required=True, help="accepted train/frozen/sealed gate JSON file")

    rollback = sub.add_parser("rollback", help="switch active version to a prior lineage snapshot")
    rollback.add_argument("--root", required=True)
    rollback.add_argument("--version-id")

    smoke = sub.add_parser("smoke", help="verify the known-reference-error candidate guard")
    smoke.add_argument("--known-source-id", action="append", default=["route-stable"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        store = RsiRecordStore(Path(args.root) if args.root else None)
        count = store.collect_runtime(Path(args.runtime_dir)) if args.runtime_dir else 0
        if args.evaluation_dir:
            count += store.collect_evaluation(Path(args.evaluation_dir), ("route_manifest.jsonl", "knowledge_queries.jsonl", "hitl_cases.jsonl"))
        print(json.dumps({"records": count}, sort_keys=True))
        return 0
    if args.command == "candidate":
        candidate = _candidate_from_args(args)
        if args.known_source_id:
            candidate.validate(args.known_source_id)
        print(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "replay":
        candidate = Candidate.from_dict(_json(args.candidate))
        bundle = ReplayBundle.from_paths(train=args.train, frozen=args.frozen, sealed=args.sealed)
        results = {name: item.to_dict() for name, item in OfflineReplayer().replay_bundle(candidate, bundle).items()}
        output = json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0
    if args.command == "gate":
        candidate = Candidate.from_dict(_json(args.candidate))
        result = evaluate_gate(
            candidate,
            _json(args.baseline),
            _json(args.proposed),
            config=GateConfig(min_quality=args.min_quality, min_safety=args.min_safety, min_efficiency=args.min_efficiency),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.accepted else 2
    if args.command == "approve":
        publisher = Publisher(Path(args.root))
        approval = publisher.approve(args.candidate, args.reviewer_id, args.reason, _json(args.gate_result))
        print(json.dumps(approval.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "rollback":
        version = Publisher(Path(args.root)).rollback(args.version_id)
        print(json.dumps(version.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "smoke":
        candidate = build_candidate(
            surface="knowledge_queries",
            before={"knowledge_queries": {"default": "old"}},
            after={"knowledge_queries": {"default": "new"}},
            source_refs=["known-reference-that-is-not-in-the-snapshot"],
        )
        try:
            candidate.validate(args.known_source_id)
        except CandidateReferenceError as exc:
            print(json.dumps({"status": "expected_reference_error", "error": str(exc)}, sort_keys=True))
            return 0
        print(json.dumps({"status": "unexpected_acceptance"}, sort_keys=True))
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
