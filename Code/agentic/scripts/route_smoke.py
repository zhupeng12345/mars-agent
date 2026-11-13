"""Replay JEV route fixtures and optionally execute the selected loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic.orchestration.diagnosis_graph import run_diagnosis_graph
from agentic.platform.contracts import InspectionFeatures
from agentic.platform.jev.router import JevRouter


class _TimeoutAdapter:
    def predict(self, _features):
        raise TimeoutError("synthetic_jev_timeout")


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(path: Path, execute: bool = True) -> dict[str, Any]:
    router = JevRouter()
    rows = []
    for fixture in load_fixtures(path):
        features = InspectionFeatures.from_dict(fixture["features"])
        adapter = _TimeoutAdapter() if fixture.get("adapter") == "timeout" else None
        selected_router = JevRouter(adapter=adapter) if adapter is not None else router
        decision = selected_router.route(features)
        row: dict[str, Any] = {
            "case_id": fixture["case_id"],
            "expected_route": fixture["expert_route"],
            "route": decision.route,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
            "router_backend": decision.router_backend,
            "model_profile": decision.model_profile,
            "gate_version": decision.gate_version,
            "passed": decision.route == fixture["expert_route"],
            "adapter": fixture.get("adapter", "none"),
            "fallback": any("adapter_error" in reason for reason in decision.reasons),
        }
        if execute:
            result = run_diagnosis_graph(features, context={"knowledge": [{"source_id": "knowledge-1", "text": "fixture"}]}, router=selected_router)
            row["status"] = result.get("result", {}).get("status")
            row["trace_nodes"] = [event.get("node") for event in result.get("trace", [])]
            row["failed_agents"] = [item.get("agent_id") for item in result.get("result", {}).get("evidence_bundle", []) if item.get("status") == "failed"]
        rows.append(row)
    return {"fixture_count": len(rows), "passed": all(row["passed"] for row in rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay JEV route fixtures")
    parser.add_argument("--fixtures", type=Path, default=Path("data/evaluation/route_manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args(argv)
    payload = replay(args.fixtures, execute=not args.no_execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fixture_count": payload["fixture_count"], "passed": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
