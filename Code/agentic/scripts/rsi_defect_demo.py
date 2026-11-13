"""Run a complete offline RSI defect-to-publish-and-rollback demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic.rsi.candidate import build_candidate
from agentic.rsi.evaluator import OfflineReplayer, ReplayBundle
from agentic.rsi.gate import evaluate_gate
from agentic.rsi.publisher import Publisher
from agentic.rsi.records import RsiRecordStore, critique_from_run


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    publisher = Publisher(output_dir / "registry")
    record_store = RsiRecordStore(output_dir)
    defect_trace = {"run_id": "run-known-defect-001", "result": {"review": {"issues": ["missing_traceable_source_citation"]}}}
    record_store.append("runtime_run", defect_trace)
    critique = critique_from_run(defect_trace, repeat_count=2)
    record_store.append("critique", critique.to_dict())
    seed = build_candidate(
        surface="prompts",
        before={"prompts": {"diagnosis": "draft"}},
        after={"prompts": {"diagnosis": "draft_with_source_ids"}},
        source_run_ids=["run-known-defect-001"],
        source_refs=["run-known-defect-001"],
        risk_notes=["known defect: draft omitted source_ids in the report"],
        metadata={"defect": "missing_source_citation"},
    )
    publisher.register(seed)
    rows = {
        "train": [{"quality": 0.90, "safety": 1.0, "efficiency": 0.70, "route": "full_loop"}],
        "frozen": [{"quality": 0.90, "safety": 1.0, "efficiency": 0.70, "route": "full_loop"}],
        "sealed": [{"quality": 0.90, "safety": 1.0, "efficiency": 0.70, "route": "full_loop"}],
    }
    bundle = ReplayBundle.from_mapping(rows)
    replayer = OfflineReplayer()
    seed_results = replayer.replay_bundle(seed, bundle)
    seed_gate = evaluate_gate(seed, seed_results, seed_results)
    publisher.approve(seed, "offline-reviewer", "seed policy reviewed", seed_gate)
    seed_version = publisher.publish(seed, seed_gate)

    candidate = build_candidate(
        surface="prompts",
        before=seed_version.surfaces,
        after={"prompts": {"diagnosis": "draft_with_source_ids_and_conditions"}},
        parent_version=seed_version.version_id,
        source_run_ids=["run-known-defect-001"],
        source_refs=["run-known-defect-001"],
        risk_notes=["fixes missing source citation and condition binding"],
        metadata={"defect": "missing_source_citation", "fix": "require source_ids and conditions"},
    )
    publisher.register(candidate)
    evaluation_rows = {
        "train": [{"sample_id": "train-1", "expected_route": "full_loop"}],
        "frozen": [{"sample_id": "frozen-1", "expected_route": "full_loop"}],
        "sealed": [{"sample_id": "sealed-1", "expected_route": "full_loop"}],
    }
    same_inputs = ReplayBundle.from_mapping(evaluation_rows)

    def baseline_evaluator(_candidate, _rows, _split):
        return {"quality": 0.90, "safety": 1.0, "efficiency": 0.70}

    def candidate_evaluator(_candidate, _rows, _split):
        return {"quality": 0.93, "safety": 1.0, "efficiency": 0.82}

    baseline = OfflineReplayer(baseline_evaluator).replay_bundle(candidate, same_inputs)
    proposed = OfflineReplayer(candidate_evaluator).replay_bundle(candidate, same_inputs)
    gate = evaluate_gate(candidate, baseline, proposed)
    approval = publisher.approve(candidate, "offline-reviewer", "known defect fixed; sealed replay passed", gate)
    version = publisher.publish(candidate, gate)
    rolled_back = publisher.rollback(seed_version.version_id)
    report = {
        "schema_version": "rsi-defect-demo-1",
        "status": "completed",
        "known_defect": "missing_source_citation",
        "critique": critique.to_dict(),
        "candidate": candidate.to_dict(),
        "replay": {"baseline": {k: v.to_dict() for k, v in baseline.items()}, "proposed": {k: v.to_dict() for k, v in proposed.items()}},
        "gate": gate.to_dict(),
        "approval": approval.to_dict(),
        "published_version": version.to_dict(),
        "rollback_version": rolled_back.to_dict(),
        "rollback_verified": rolled_back.version_id == seed_version.version_id,
    }
    path = output_dir / "rsi_defect_demo.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "runtime_data" / "rsi")
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"status": report["status"], "gate_accepted": report["gate"]["accepted"], "rollback_verified": report["rollback_verified"]}, ensure_ascii=False))
    return 0 if report["gate"]["accepted"] and report["rollback_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
