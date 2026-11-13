"""Model-free evaluation metrics for agent routing and evidence health."""

from __future__ import annotations

import json
import statistics
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


def route_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    correct = sum(bool(row.get("passed", row.get("route") == row.get("expected_route"))) for row in rows)
    expected_by_route = {
        route: [row for row in rows if row.get("expected_route") == route]
        for route in ("simple_loop", "full_loop", "retake", "human")
    }
    simple = expected_by_route["simple_loop"]
    full = expected_by_route["full_loop"]
    simple_tp = sum(row.get("route") == "simple_loop" for row in simple)
    full_tp = sum(row.get("route") == "full_loop" for row in full)
    dangerous_simple = sum(row.get("route") == "simple_loop" and row.get("expected_route") in {"full_loop", "human"} for row in rows)
    predicted_simple = sum(row.get("route") == "simple_loop" for row in rows)
    complex_count = sum(row.get("expected_route") in {"full_loop", "human"} for row in rows)
    return {
        "evaluation_kind": "policy_fixture_regression",
        "model_accuracy_claimable": False,
        "fixture_count": total,
        "fixture_route_match_rate": correct / total if total else None,
        "simple_fixture_precision": simple_tp / predicted_simple if predicted_simple else None,
        "simple_fixture_recall": simple_tp / len(simple) if simple else None,
        "complex_fixture_full_recall": full_tp / len(full) if full else None,
        "retake_fixture_recall": (
            sum(row.get("route") == "retake" for row in expected_by_route["retake"])
            / len(expected_by_route["retake"])
            if expected_by_route["retake"]
            else None
        ),
        "human_fixture_recall": (
            sum(row.get("route") == "human" for row in expected_by_route["human"])
            / len(expected_by_route["human"])
            if expected_by_route["human"]
            else None
        ),
        "dangerous_fixture_simple_leakage_rate": dangerous_simple / complex_count if complex_count else None,
    }


def route_flow_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit route transitions without treating them as expert accuracy."""
    rows = [row for row in results if isinstance(row, Mapping)]
    eligible = [row for row in rows if row.get("initial_route") or row.get("route_history")]
    transitions: dict[str, int] = {}
    initial: dict[str, int] = {}
    final: dict[str, int] = {}
    upgrades = 0
    for row in eligible:
        history = row.get("route_history")
        history = [str(item) for item in history if item] if isinstance(history, (list, tuple)) else []
        initial_route = str(row.get("initial_route") or (history[0] if history else row.get("route", "unknown")))
        final_route = str(row.get("route", history[-1] if history else "unknown"))
        initial[initial_route] = initial.get(initial_route, 0) + 1
        final[final_route] = final.get(final_route, 0) + 1
        key = f"{initial_route}->{final_route}"
        transitions[key] = transitions.get(key, 0) + 1
        upgrades += int(row.get("upgrade_count", 0) or 0)
    return {
        "evaluation_kind": "route_flow_audit",
        "status": "evaluated" if eligible else "not_run",
        "run_count": len(eligible),
        "initial_routes": dict(sorted(initial.items())),
        "final_routes": dict(sorted(final.items())),
        "transitions": dict(sorted(transitions.items())),
        "upgrade_count": upgrades,
        "upgrade_rate": upgrades / len(eligible) if eligible else None,
        "accuracy_claimable": False,
        "efficiency_gain_claimable": False,
    }


def rsi_runtime_policy_metrics(run_records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit policy-version provenance in completed runtime traces only."""
    rows = list(run_records)
    completed = [
        record for record in rows
        if record.get("finished_at") and isinstance(record.get("result"), Mapping)
    ]
    versions: dict[str, dict[str, Any]] = {}
    missing = 0
    prompt_variants: dict[str, int] = {}
    query_term_sets: dict[str, int] = {}
    runs_with_prompt_variant = 0
    runs_with_query_policy = 0
    for record in completed:
        result = record.get("result", {})
        if not isinstance(result, Mapping):
            missing += 1
            continue
        decision = result.get("route_decision", {})
        if not isinstance(decision, Mapping) or not decision.get("policy_version"):
            missing += 1
            continue
        version = str(decision["policy_version"])
        entry = versions.setdefault(version, {
            "run_count": 0,
            "policy_hashes": set(),
            "routes": {},
            "backends": {},
        })
        entry["run_count"] += 1
        policy_hash = str(decision.get("policy_hash", ""))
        if policy_hash:
            entry["policy_hashes"].add(policy_hash)
        route = str(decision.get("route", result.get("route", "unknown")))
        backend = str(decision.get("router_backend", "unknown"))
        entry["routes"][route] = entry["routes"].get(route, 0) + 1
        entry["backends"][backend] = entry["backends"].get(backend, 0) + 1
        runtime = result.get("rsi_runtime") if isinstance(result.get("rsi_runtime"), Mapping) else {}
        prompt_variant = runtime.get("prompt_variant") or result.get("prompt_version")
        if prompt_variant:
            prompt_key = str(prompt_variant)
            prompt_variants[prompt_key] = prompt_variants.get(prompt_key, 0) + 1
            runs_with_prompt_variant += 1
        query_terms = runtime.get("knowledge_query_terms")
        if isinstance(query_terms, (list, tuple)):
            query_key = json.dumps([str(item) for item in query_terms], ensure_ascii=False, separators=(",", ":"))
            query_term_sets[query_key] = query_term_sets.get(query_key, 0) + 1
            runs_with_query_policy += 1

    normalized = {
        version: {
            "run_count": item["run_count"],
            "policy_hashes": sorted(item["policy_hashes"]),
            "routes": item["routes"],
            "backends": item["backends"],
        }
        for version, item in sorted(versions.items())
    }
    return {
        "evaluation_kind": "runtime_policy_trace_audit",
        "status": "evaluated" if completed else "not_run",
        "input_run_record_count": len(rows),
        "selected_completed_run_count": len(completed),
        "incomplete_run_record_count": len(rows) - len(completed),
        "runs_with_policy_version": sum(item["run_count"] for item in versions.values()),
        "runs_missing_policy_version": missing,
        "versions": normalized,
        "prompt_variants": dict(sorted(prompt_variants.items())),
        "query_term_sets": dict(sorted(query_term_sets.items())),
        "runs_with_prompt_variant": runs_with_prompt_variant,
        "runs_with_query_policy": runs_with_query_policy,
        "accuracy_claimable": False,
        "behavior_gain_claimable": False,
    }


_PLACEHOLDER_SUMMARIES = {
    "fake provider structured response",
    "review the visual evidence and cited process knowledge.",
    "请人工确认视觉证据、知识引用和现场检查项",
}


def _source_ids(value: Any) -> list[str]:
    """Extract source IDs from strings, EvidenceRef mappings, or nested lists."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        source_id = value.get("source_id")
        ids: list[str] = []
        if isinstance(source_id, str) and source_id.strip():
            ids.append(source_id)
        for key in ("evidence", "evidence_refs", "knowledge_evidence", "sources"):
            ids.extend(_source_ids(value.get(key)))
        return list(dict.fromkeys(ids))
    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(_source_ids(item))
        return ids
    return []


def _is_eligible_conclusion(result: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> bool:
    """Exclude routing handoffs and empty aggregator/template drafts."""
    route = result.get("route")
    if route in {"retake", "human"}:
        return False
    if isinstance(diagnosis.get("diagnosis"), str) and diagnosis["diagnosis"].strip():
        return True
    summary = diagnosis.get("summary")
    if isinstance(summary, str):
        normalized = summary.strip().casefold()
        if normalized and normalized not in _PLACEHOLDER_SUMMARIES and not normalized.endswith(" evidence aggregation"):
            return True
    findings = [str(item).strip() for item in diagnosis.get("findings", []) or []]
    if any(item and not item.casefold().startswith(("evidence_agents=", "failed_agents=")) for item in findings):
        return True
    recommendations = [str(item).strip() for item in diagnosis.get("recommendations", []) or []]
    return any(
        item and item != "请人工确认视觉证据、知识引用和现场检查项"
        for item in recommendations
    )


def evidence_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = list(results)
    citations: list[str] = []
    valid = 0
    complete_conclusions = 0
    conclusions_without_source = 0
    eligible_conclusions = 0
    for result in results:
        diagnosis = result.get("diagnosis", {})
        if not isinstance(diagnosis, Mapping):
            diagnosis = {}
        known_sources: set[str] = set()
        visual = result.get("visual", {})
        if isinstance(visual, Mapping):
            known_sources.update(_source_ids(visual))
        known_sources.update(_source_ids(result.get("knowledge", [])))
        for case in result.get("cases", []) or []:
            if isinstance(case, Mapping):
                known_sources.update(_source_ids(case))
                if case.get("case_id"):
                    known_sources.add(str(case["case_id"]))
        for item in result.get("evidence_bundle", []) or []:
            if not isinstance(item, Mapping):
                continue
            output = item.get("output", {}) if isinstance(item, Mapping) else {}
            sources = output.get("sources", []) if isinstance(output, Mapping) else []
            known_sources.update(_source_ids(sources))
            refs = item.get("evidence", [])
            known_sources.update(_source_ids(refs))
            if isinstance(output, Mapping) and output.get("source_id"):
                known_sources.add(str(output["source_id"]))
        known_sources.update(_source_ids(result.get("process_state", {})))
        if not _is_eligible_conclusion(result, diagnosis):
            continue
        eligible_conclusions += 1
        cited_ids = list(dict.fromkeys(
            _source_ids(diagnosis.get("source_ids", []))
            + _source_ids(diagnosis.get("knowledge_evidence", []))
            + _source_ids(diagnosis.get("evidence", []))
        ))
        citations.extend(cited_ids)
        valid += sum(item in known_sources for item in cited_ids)
        if not cited_ids:
            conclusions_without_source += 1
        complete_conclusions += bool(result.get("evidence_bundle") or diagnosis.get("findings") or diagnosis.get("evidence"))
    return {
        "runs": len(results),
        "eligible_conclusions": eligible_conclusions,
        "excluded_nonconclusions": len(results) - eligible_conclusions,
        "evidence_chain_complete_rate": complete_conclusions / eligible_conclusions if eligible_conclusions else None,
        "citation_count": len(citations),
        "citation_exists_rate": valid / len(citations) if citations else None,
        "citation_support_rate": None,
        "citation_support_status": "not_evaluated_without_human_entailment_labels",
        "no_source_conclusion_count": conclusions_without_source,
        "no_source_conclusion_ratio": conclusions_without_source / eligible_conclusions if eligible_conclusions else None,
        "evaluation_kind": "diagnosis_evidence_coverage",
    }


FULL_LOOP_SPECIALISTS = (
    "visual_review",
    "evidence_quality",
    "knowledge_rag",
    "case_compare",
    "process_state",
)


def multi_agent_flow_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure full-loop worker coverage and evidence handoff integrity.

    This is an orchestration metric, not a measure of diagnostic correctness.
    """
    full_runs = [
        result for result in results
        if isinstance(result, Mapping) and result.get("route") == "full_loop"
    ]
    per_agent: dict[str, dict[str, int]] = {
        agent_id: {"completed": 0, "failed": 0, "missing": 0, "abstained": 0}
        for agent_id in FULL_LOOP_SPECIALISTS
    }
    expected_slots = len(full_runs) * len(FULL_LOOP_SPECIALISTS)
    present_slots = completed_slots = failed_slots = missing_slots = abstained_slots = 0
    complete_runs = 0
    lineage_checks = lineage_violations = lineage_missing = 0

    for result in full_runs:
        bundle = result.get("evidence_bundle", [])
        by_agent: dict[str, list[Mapping[str, Any]]] = {agent_id: [] for agent_id in FULL_LOOP_SPECIALISTS}
        if isinstance(bundle, (list, tuple)):
            for item in bundle:
                if not isinstance(item, Mapping):
                    continue
                agent_id = str(item.get("agent_id", ""))
                if agent_id in by_agent:
                    by_agent[agent_id].append(item)

        run_complete = True
        for agent_id in FULL_LOOP_SPECIALISTS:
            rows = by_agent[agent_id]
            if not rows:
                missing_slots += 1
                per_agent[agent_id]["missing"] += 1
                run_complete = False
                continue
            present_slots += 1
            if len(rows) != 1:
                run_complete = False
            worker = rows[-1]
            status = str(worker.get("status", "unknown")).casefold()
            output = worker.get("output", {})
            output = output if isinstance(output, Mapping) else {}
            if status in {"completed", "success", "succeeded"}:
                completed_slots += 1
                per_agent[agent_id]["completed"] += 1
                abstained = str(output.get("analysis_status", "")).casefold() in {"abstain", "missing", "insufficient"}
                if abstained:
                    abstained_slots += 1
                    per_agent[agent_id]["abstained"] += 1
                source_ids = set(_source_ids(output.get("source_ids", [])))
                raw_evidence = worker.get("evidence", [])
                refs = raw_evidence if isinstance(raw_evidence, (list, tuple)) else []
                evidence_ids = {
                    source_id
                    for ref in refs if isinstance(ref, Mapping) and str(ref.get("agent_id", "")) == agent_id
                    for source_id in _source_ids(ref)
                }
                if not abstained:
                    lineage_checks += 1
                    if not source_ids:
                        lineage_missing += 1
                        per_agent[agent_id]["missing_lineage"] = per_agent[agent_id].get("missing_lineage", 0) + 1
                    elif not source_ids.issubset(evidence_ids):
                        lineage_violations += 1
                        per_agent[agent_id]["lineage_violations"] = per_agent[agent_id].get("lineage_violations", 0) + 1
                if len(rows) != 1:
                    run_complete = False
            else:
                failed_slots += 1
                per_agent[agent_id]["failed"] += 1
                run_complete = False
        complete_runs += int(run_complete)

    status = "evaluated" if full_runs else "not_run"
    return {
        "evaluation_kind": "multi_agent_flow_integrity",
        "accuracy_claimable": False,
        "status": status,
        "eligible_full_loop_runs": len(full_runs),
        "expected_specialist_ids": list(FULL_LOOP_SPECIALISTS),
        "expected_worker_slots": expected_slots,
        "present_worker_slots": present_slots,
        "worker_slot_coverage_rate": present_slots / expected_slots if expected_slots else None,
        "completed_worker_slots": completed_slots,
        "failed_worker_slots": failed_slots,
        "missing_worker_slots": missing_slots,
        "abstained_worker_outputs": abstained_slots,
        "fully_successful_runs": complete_runs,
        "fully_successful_run_rate": complete_runs / len(full_runs) if full_runs else None,
        "source_lineage_checks": lineage_checks,
        "source_lineage_missing": lineage_missing,
        "source_lineage_violations": lineage_violations,
        "source_lineage_integrity_rate": (lineage_checks - lineage_violations - lineage_missing) / lineage_checks if lineage_checks else None,
        "by_agent": per_agent,
    }


def worker_execution_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize configured full-loop worker reliability without efficacy claims."""
    by_agent: dict[str, dict[str, int]] = {}
    policy_counts: dict[str, int] = {}
    worker_count = retries_total = retried_workers = failed = timed_out = configured = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        bundle = result.get("evidence_bundle", [])
        if not isinstance(bundle, (list, tuple)):
            continue
        for worker in bundle:
            if not isinstance(worker, Mapping):
                continue
            agent_id = str(worker.get("agent_id", ""))
            if agent_id not in FULL_LOOP_SPECIALISTS:
                continue
            worker_count += 1
            metadata = worker.get("metadata") if isinstance(worker.get("metadata"), Mapping) else {}
            try:
                retry_count = max(0, int(metadata.get("retry_count", worker.get("retry_count", 0)) or 0))
            except (TypeError, ValueError):
                retry_count = 0
            retries_total += retry_count
            retried_workers += int(retry_count > 0)
            policy_id = metadata.get("retry_policy_id", worker.get("retry_policy_id"))
            if isinstance(policy_id, str) and policy_id.strip():
                policy_key = policy_id.strip()
                configured += 1
                policy_counts[policy_key] = policy_counts.get(policy_key, 0) + 1
            status = str(worker.get("status", "unknown")).casefold()
            error_code = str(worker.get("error_code") or metadata.get("error_code") or "").casefold()
            timeout = status in {"timeout", "timed_out"} or "timeout" in error_code
            failed_worker = status in {"failed", "error", "timeout", "timed_out"}
            failed += int(failed_worker)
            timed_out += int(timeout)
            row = by_agent.setdefault(agent_id, {"workers": 0, "failed": 0, "timed_out": 0, "retries": 0})
            row["workers"] += 1
            row["failed"] += int(failed_worker)
            row["timed_out"] += int(timeout)
            row["retries"] += retry_count

    return {
        "evaluation_kind": "worker_execution_integrity",
        "status": "evaluated" if worker_count else "not_run",
        "worker_count": worker_count,
        "retry_total": retries_total,
        "workers_retried": retried_workers,
        "failed_worker_count": failed,
        "timeout_worker_count": timed_out,
        "policy_coverage_rate": configured / worker_count if worker_count else None,
        "policy_counts": dict(sorted(policy_counts.items())),
        "by_agent": dict(sorted(by_agent.items())),
        "diagnostic_accuracy_claimable": False,
        "efficiency_gain_claimable": False,
    }


def latency_metrics(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    raw_values = [event.get("elapsed_ms") for event in events if event.get("elapsed_ms") is not None]
    values = []
    for value in raw_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 0:
            values.append(numeric)
    excluded_invalid_count = len(raw_values) - len(values)
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "excluded_invalid_count": excluded_invalid_count}
    values.sort()
    def percentile(value: float) -> float:
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * value))))
        return round(values[index], 3)
    return {"count": len(values), "p50_ms": percentile(0.5), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99), "mean_ms": round(statistics.mean(values), 3), "excluded_invalid_count": excluded_invalid_count}


def _parse_runtime_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def end_to_end_latency_metrics(
    runs: Iterable[Mapping[str, Any]],
    *,
    selected_run_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Pair runtime start/finish records by run_id and report complete runs only."""
    selected = {str(item).strip() for item in selected_run_ids or () if str(item).strip()} or None
    grouped: dict[str, dict[str, list[Any]]] = {}
    invalid_timestamp_count = 0
    for row in runs:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id or (selected is not None and run_id not in selected):
            continue
        state = grouped.setdefault(run_id, {"starts": [], "finishes": []})
        if row.get("started_at") is not None:
            parsed = _parse_runtime_timestamp(row.get("started_at"))
            if parsed is None:
                invalid_timestamp_count += 1
            else:
                state["starts"].append(parsed)
        if row.get("finished_at") is not None:
            parsed = _parse_runtime_timestamp(row.get("finished_at"))
            if parsed is None:
                invalid_timestamp_count += 1
            else:
                state["finishes"].append(parsed)

    durations: list[float] = []
    missing_timing_run_ids: list[str] = []
    negative_duration_count = 0
    for run_id, state in sorted(grouped.items()):
        starts = sorted(state["starts"])
        finishes = sorted(state["finishes"])
        pairs = min(len(starts), len(finishes))
        if pairs == 0:
            missing_timing_run_ids.append(run_id)
            continue
        for started_at, finished_at in zip(starts[-pairs:], finishes[-pairs:]):
            duration_ms = (finished_at - started_at).total_seconds() * 1000.0
            if duration_ms < 0:
                negative_duration_count += 1
                missing_timing_run_ids.append(run_id)
                continue
            durations.append(duration_ms)
        if len(starts) != len(finishes):
            missing_timing_run_ids.append(run_id)

    values = sorted(durations)

    def percentile(value: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * value))))
        return round(values[index], 3)

    return {
        "count": len(values),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
        "excluded_incomplete_run_count": len(set(missing_timing_run_ids)),
        "excluded_invalid_timestamp_count": invalid_timestamp_count,
        "excluded_negative_duration_count": negative_duration_count,
        "missing_timing_run_ids": sorted(set(missing_timing_run_ids)),
        "evaluation_kind": "runtime_end_to_end_latency",
    }


def visual_data_coverage(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(manifest_rows)
    videos = [row for row in rows if row.get("source") == "video"]
    weak = sum(row.get("label_status") == "unlabeled/weak" for row in rows)
    return {"records": len(rows), "video_windows": len(videos), "weak_or_unlabeled": weak, "expert_accuracy_available": False, "metric_status": "structure_validation_only"}


def system_health_metrics(
    events: Iterable[Mapping[str, Any]],
    runs: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compute auditable operational rates from observable runtime records."""
    events = list(events)
    run_rows = [row for row in runs if isinstance(row, Mapping) and isinstance(row.get("result"), Mapping)]
    reviews = list(reviews)
    total_runs = len(run_rows)
    failed = sum(str(event.get("status", "")).lower() in {"failed", "error", "timeout", "timed_out"} for event in events)
    timed_out = sum(str(event.get("status", "")).lower() in {"timeout", "timed_out"} or str(event.get("error_code", "")).lower() in {"timeout", "timeouterror"} for event in events)
    retries = sum(int(event.get("attempt", 1) or 1) > 1 for event in events)
    fallbacks = sum("fallback" in json_text(event.get("summary", {})).lower() or str(event.get("node", "")).lower().endswith("fallback") for event in events)
    handoffs = sum(isinstance(row.get("result", {}).get("diagnosis"), Mapping) and isinstance(row.get("result", {}).get("evidence_bundle"), list) for row in run_rows)
    decisions = [row for row in reviews if row.get("action") in {"approve", "amend", "reject", "request_retake"}]
    adopted = sum(row.get("action") in {"approve", "amend"} for row in decisions)
    amended = sum(row.get("action") == "amend" for row in decisions)
    action_counts = {
        action: sum(row.get("action") == action for row in decisions)
        for action in ("approve", "amend", "reject", "request_retake")
    }
    denominator = max(1, total_runs)
    return {
        "run_count": total_runs,
        "event_count": len(events),
        "node_failure_rate": failed / max(1, len(events)) if events else None,
        "timeout_rate": timed_out / max(1, len(events)) if events else None,
        "retry_rate": retries / max(1, len(events)) if events else None,
        "fallback_rate": fallbacks / denominator if total_runs else None,
        "structured_handoff_rate": handoffs / denominator if total_runs else None,
        "human_decision_count": len(decisions),
        "human_action_counts": action_counts,
        "human_adoption_rate": adopted / len(decisions) if decisions else None,
        "human_amend_rate": amended / len(decisions) if decisions else None,
    }


def json_text(value: Any) -> str:
    """Stable stringification for optional summary metadata."""
    try:
        import json
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
