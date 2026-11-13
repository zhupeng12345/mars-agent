"""Generate compact, deterministic presentation traces for the PPT demo.

The presentation traces are intentionally labelled ``flow_validation``.  They
are a reproducible UI/PPT fixture layer, not a replacement for model training
or route-quality evaluation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CASE_ORDER = ("stable_case", "conflict_high_risk", "bad_quality", "ood_unknown_case")
WORKER_IDS = (
    "visual_review",
    "evidence_quality",
    "knowledge_rag",
    "case_compare",
    "process_state",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "data" / "evaluation" / "route_manifest.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def _compact_worker(agent_id: str, *, route: str, run_id: str, source: str) -> dict[str, Any]:
    labels = {
        "visual_review": ("completed", "关键帧与 ROI 序列复核"),
        "evidence_quality": ("completed", "图像质量和证据覆盖检查"),
        "knowledge_rag": ("completed", "工艺规则片段和引用 ID"),
        "case_compare": ("completed", "已审核特殊案例条件比对"),
        "process_state": ("completed", "14 维过程状态摘要；视频对齐未知"),
    }
    status, summary = labels[agent_id]
    return {
        "agent_id": agent_id,
        "role": agent_id,
        "status": status,
        "attempt": 1,
        "error_code": None,
        "duration_ms": None,
        "duration_scope": "not_benchmarked",
        "evidence_ids": [f"{agent_id}:{run_id}"],
        "source": source,
        "summary": summary,
        "implementation": "presentation_fixture",
        "route": route,
    }


def _simple_agents(run_id: str) -> list[dict[str, Any]]:
    return [
        _compact_worker("case_compare", route="simple_loop", run_id=run_id, source="special_case_memory"),
        _compact_worker("knowledge_rag", route="simple_loop", run_id=run_id, source="knowledge_pack_v1"),
        {
            "agent_id": "light_diagnosis",
            "role": "轻量诊断摘要 Agent",
            "status": "completed",
            "attempt": 1,
            "error_code": None,
            "duration_ms": None,
            "duration_scope": "not_benchmarked",
            "evidence_ids": [f"light_diagnosis:{run_id}"],
            "source": "simple_loop_policy",
            "summary": "依据已审核案例和规则形成结构化摘要",
            "implementation": "presentation_fixture",
            "route": "simple_loop",
        },
        {
            "agent_id": "evidence_review",
            "role": "确定性证据复核 Agent",
            "status": "completed",
            "attempt": 1,
            "error_code": None,
            "duration_ms": None,
            "duration_scope": "not_benchmarked",
            "evidence_ids": [f"evidence_review:{run_id}"],
            "source": "review_guard",
            "summary": "证据充分，保留人工确认出口",
            "implementation": "presentation_fixture",
            "route": "simple_loop",
        },
    ]


def _load_full_worker_source(root: Path) -> str:
    # Keep the source provenance real when the prior end-to-end demo exists.
    candidate = root / "runtime_data" / "demo_chain_high.json"
    return str(candidate) if candidate.exists() else "runtime_data/demo_chain_high.json"


def _build_simple_upgrade_case() -> dict[str, Any]:
    """Capture the real graph's deterministic simple-to-full escalation path."""
    from agentic.orchestration.diagnosis_graph import run_diagnosis_graph
    from agentic.platform.contracts import InspectionFeatures

    run_id = "ppt-simple-upgrade-001"
    thread_id = "presentation-simple-upgrade-001"
    features = InspectionFeatures(
        run_id=run_id,
        thread_id=thread_id,
        visual_confidence=0.9,
        risk_score=0.1,
        knowledge_coverage=1.0,
        case_similarity=0.9,
        novelty_score=0.1,
        features={"quality_status": "pass", "target_present": True, "case_applicable": True},
    )
    graph = run_diagnosis_graph(
        features,
        context={"run_id": run_id, "thread_id": thread_id, "llm_profile": "fake_json", "visual": {"confidence": 0.9}},
    )
    result = graph["result"]
    if result.get("route_history") != ["simple_loop", "full_loop"] or result.get("upgrade_count") != 1:
        raise RuntimeError("simple_upgrade fixture did not traverse the expected graph path")

    agents = []
    for item in result.get("evidence_bundle", []):
        agent_id = str(item.get("agent_id", ""))
        if agent_id not in WORKER_IDS:
            continue
        output = item.get("output") if isinstance(item.get("output"), Mapping) else {}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        source_ids = output.get("source_ids") or output.get("evidence_ids") or []
        agents.append({
            "agent_id": agent_id,
            "role": agent_id,
            "status": item.get("status", "unknown"),
            "attempt": item.get("attempt", metadata.get("attempt", 1)),
            "retry_count": metadata.get("retry_count", 0),
            "retry_policy_id": metadata.get("retry_policy_id"),
            "timeout_s": metadata.get("timeout_s"),
            "error_code": item.get("error_code"),
            "duration_ms": item.get("elapsed_ms"),
            "duration_scope": "single_flow_validation_run",
            "evidence_ids": list(source_ids) if isinstance(source_ids, list) else [],
            "source": ", ".join(str(source) for source in source_ids) if source_ids else "none",
            "summary": str(output.get("interpretation") or output.get("analysis_status") or item.get("status", "unknown")),
            "implementation": "diagnosis_graph",
            "route": "full_loop",
        })
    if {item["agent_id"] for item in agents} != set(WORKER_IDS):
        raise RuntimeError("simple_upgrade fixture is missing a full-loop specialist")

    return {
        "schema_version": "presentation-demo-1",
        "status": "flow_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "thread_id": thread_id,
        "case_id": "simple_upgrade",
        "presentation_title": "证据缺口触发循环升级",
        "input": {
            "source": "presentation://missing-visual-source",
            "kind": "fixture",
            "sampling_policy": "single_evidence_fixture",
            "quality_status": "pass",
        },
        "route_decision": result.get("route_decision", {}),
        "initial_route": result["initial_route"],
        "route_history": result["route_history"],
        "upgrade_count": result["upgrade_count"],
        "graph_trace": graph.get("trace", []),
        "graph_result": result,
        "visual_evidence": {
            "model_profile": "flow-validation-evidence-gap",
            "confidence": 0.9,
            "frame_ids": [],
            "roi_status": "source_missing",
            "class_output_status": "unknown",
            "issues": ["visual_source_missing"],
            "expert_video_labels_available": False,
            "evidence_ids": [],
        },
        "process_state": {
            "agent_id": "process_state",
            "variable_count": 14,
            "process_regime": "unknown",
            "event_risk": "independent_prior",
            "process_missing": ["process_snapshot"],
            "alignment_quality": "unknown",
            "status": "missing",
        },
        "agents": agents,
        "fan_out": {
            "status": "completed",
            "agent_count": len(agents),
            "reducer": "evidence_bundle",
            "max_rounds": 2,
            "retry_policy": "worker_policies.json",
        },
        "diagnosis": result.get("diagnosis", {}),
        "review": result.get("review", {}),
        "human_review": {
            "status": "waiting_review",
            "allowed_actions": ["approve", "amend", "reject", "request_retake"],
            "preview_action": "request_retake",
            "reviewer_id": "demo_operator",
            "audit_status": "flow_validation",
            "issues": result.get("review", {}).get("issues", []),
        },
        "memory": {"case_status": "candidate", "approved_for_active_search": False, "rsi_input": True},
        "presentation": {
            "primary_message": "轻量复核发现来源缺口，升级完整多 Agent 取证并交人工复核",
            "route_history": result["route_history"],
            "upgrade_count": result["upgrade_count"],
            "slide_groups": ["route", "multi_agent", "hitl", "audit"],
            "visible_agent_count": len(agents),
            "watermark": "FLOW VALIDATION",
        },
        "evaluation": {
            "status": "structure_validation_only",
            "visual_labels": "unavailable",
            "route_labels": "synthetic_evidence_gap_regression",
            "process_video_alignment": "unknown",
            "metrics_claimable": False,
            "display_cards": ["route_history", "agent_handoff", "evidence_gap", "human_audit"],
        },
        "claims": {
            "flow_validation_only": True,
            "expert_video_labels_available": False,
            "real_superheat_measurement": False,
            "process_video_alignment": "unknown",
            "timing_benchmark_available": False,
        },
    }


def build_case(root: Path, case_id: str, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if case_id == "simple_upgrade":
        return _build_simple_upgrade_case()
    if case_id not in rows:
        raise KeyError(f"unknown route fixture: {case_id}")
    row = dict(rows[case_id])
    features = dict(row.get("features", {}))
    route = str(row.get("expert_route", "human"))
    run_id = f"ppt-{case_id}-001"
    thread_id = f"presentation-{case_id}-001"
    quality_status = str(features.get("features", {}).get("quality_status", "pass"))
    source = {
        "stable_case": "data/video/正常.mp4",
        "conflict_high_risk": "data/video/偏高.mp4",
        "bad_quality": "presentation://bad-quality-fixture",
        "ood_unknown_case": "data/video/偏高.mp4",
    }[case_id]
    if route == "full_loop":
        agents = [_compact_worker(agent_id, route=route, run_id=run_id, source=_load_full_worker_source(root)) for agent_id in WORKER_IDS]
    elif route == "simple_loop":
        agents = _simple_agents(run_id)
    else:
        agents = []
    if route == "retake":
        human_action = "request_retake"
        human_status = "waiting_for_retake"
        diagnosis_status = "quality_gate_stopped"
    elif route == "human":
        human_action = "approve"
        human_status = "waiting_review"
        diagnosis_status = "evidence_gap_review"
    else:
        human_action = "approve"
        human_status = "waiting_review"
        diagnosis_status = "draft_ready"
    return {
        "schema_version": "presentation-demo-1",
        "status": "flow_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "thread_id": thread_id,
        "case_id": case_id,
        "input": {
            "source": source,
            "kind": "video" if source.endswith(".mp4") else "fixture",
            "sampling_policy": "fixed_window_8_frames",
            "quality_status": quality_status,
        },
        "route_decision": {
            "route": route,
            "router_backend": "deterministic_rule",
            "router_version": "deterministic_alu_router_v1",
            "gate_version": "alu-jev-safety-0.1",
            "confidence": float(features.get("visual_confidence", 0.0) or 0.0),
            "risk_score": float(features.get("risk_score", 0.0) or 0.0),
            "novelty_score": float(features.get("novelty_score", 0.0) or 0.0),
            "evidence_conflict": float(features.get("evidence_conflict", 0.0) or 0.0),
            "reasons": {
                "simple_loop": "稳定视觉证据 + 已审核案例/规则命中",
                "full_loop": "风险或证据冲突，启动五 Agent 并行取证",
                "retake": "质量门控未通过，请求补拍",
                "human": "新颖性或证据缺口较高，进入人工复核",
            }[route],
            "alignment_quality": "unknown",
            "model_profile": "deterministic_alu_router_v1",
            "implementation": "presentation_fixture",
        },
        "visual_evidence": {
            "model_profile": "fgmvae-t-flow-validation",
            "frame_ids": list(range(8)) if route not in {"retake", "human"} else [],
            "roi_status": "available" if route not in {"retake", "human"} else "pending",
            "class_output_status": "flow_validation_only",
            "expert_video_labels_available": False,
            "evidence_ids": [f"visual:{run_id}"] if route not in {"retake", "human"} else [],
        },
        "process_state": {
            "agent_id": "process_state",
            "variable_count": 14,
            "variable_groups": ["electrical_state", "fluctuation", "feeding_operation", "equipment_state"],
            "process_regime": "demo_summary",
            "event_risk": "independent_prior",
            "process_missing": ["video_mapping"],
            "alignment_quality": "unknown",
            "status": "independent_prior",
        },
        "agents": agents,
        "fan_out": {
            "status": "completed" if route == "full_loop" else "skipped_by_route",
            "agent_count": len(agents),
            "reducer": "evidence_bundle",
            "max_rounds": 2,
            "retry_policy": "bounded_demo_policy",
        },
        "diagnosis": {
            "status": diagnosis_status,
            "risk_level": "unknown",
            "diagnosis_class": "C/N/H_status_pending_expert_label",
            "numeric_superheat": None,
            "source_ids": [item for agent in agents for item in agent.get("evidence_ids", [])],
            "requires_human_review": True,
        },
        "human_review": {
            "status": human_status,
            "allowed_actions": ["approve", "amend", "reject", "request_retake"],
            "preview_action": human_action,
            "reviewer_id": "demo_operator",
            "audit_status": "flow_validation",
        },
        "memory": {
            "case_status": "candidate" if route in {"simple_loop", "full_loop"} else "not_created",
            "approved_for_active_search": False,
            "rsi_input": True,
        },
        "presentation": {
            "primary_message": {
                "simple_loop": "稳定证据进入轻量诊断并保留人工确认",
                "full_loop": "风险/冲突触发五 Agent 并行取证与聚合",
                "retake": "质量门控拦截后请求补拍",
                "human": "新颖性和证据缺口触发人工复核",
            }[route],
            "slide_groups": ["route", "multi_agent", "hitl", "audit"],
            "visible_agent_count": len(agents),
            "watermark": "FLOW VALIDATION",
        },
        "evaluation": {
            "status": "structure_validation_only",
            "visual_labels": "unavailable",
            "route_labels": "fixture_replay",
            "process_video_alignment": "unknown",
            "metrics_claimable": False,
            "display_cards": ["evidence_chain", "route_reason", "agent_handoff", "human_audit"],
        },
        "rsi": {
            "status": "offline_demo",
            "candidate_id": f"candidate-{case_id}-citation-fix",
            "critique": "演示：检查证据引用覆盖和路由理由完整性",
            "gate": "sealed_replay_required",
            "approval": "pending_human_approval",
            "publish": "registry_only",
            "rollback": "available_in_offline_registry",
        },
        "claims": {
            "flow_validation_only": True,
            "expert_video_labels_available": False,
            "real_superheat_measurement": False,
            "process_video_alignment": "unknown",
            "timing_benchmark_available": False,
        },
    }


def build_all(root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _route_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for case_id in (*CASE_ORDER, "simple_upgrade"):
        payload = build_case(root, case_id, rows)
        target = output_dir / f"{case_id}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sidecar = {
            "asset_id": f"ppt-demo-{case_id}",
            "case_id": case_id,
            "run_id": payload["run_id"],
            "source_files": [payload["input"]["source"]],
            "status": "flow_validation",
            "claim_policy": "演示结构验证，不代表模型精度或真实过热度测量",
            "script": "agentic.scripts.presentation_demo",
        }
        (output_dir / f"{case_id}.sidecar.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cases.append({"case_id": case_id, "route": payload["route_decision"]["route"], "path": str(target), "run_id": payload["run_id"]})
    index = {
        "schema_version": "presentation-demo-index-1",
        "status": "flow_validation",
        "case_count": len(cases),
        "cases": cases,
        "claims": {"metrics": False, "real_superheat": False, "video_process_alignment": False},
    }
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "runtime_data" / "presentation_demo").resolve()
    print(json.dumps(build_all(root, output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
