"""Conditional JEV routing and controlled full-loop fan-out.

This graph is intentionally independent from the legacy single-frame workflow.
It accepts already normalized evidence and can therefore be exercised without a
model service. P4 adds a persistent interrupt/resume checkpoint around the
human-review node.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import operator
import queue
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional, TypedDict

try:  # LangGraph is optional for the local JSON-backed HITL service.
    from langgraph.graph import END, START, StateGraph
    try:
        from langgraph.types import interrupt as langgraph_interrupt
    except ImportError:  # pragma: no cover - older LangGraph releases
        langgraph_interrupt = None
    try:
        from langgraph.types import Send
    except ImportError:  # pragma: no cover - older LangGraph releases
        Send = None  # type: ignore[assignment,misc]
    try:
        from langgraph.types import Command
    except ImportError:  # pragma: no cover
        Command = None  # type: ignore[assignment,misc]
except ImportError:  # pragma: no cover - exercised by the minimal test env
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment,misc]
    langgraph_interrupt = None
    Send = None  # type: ignore[assignment,misc]
    Command = None  # type: ignore[assignment,misc]

from agentic.agents.review_guard import review_draft
from agentic.agents.specialists import build_full_loop_workers
from agentic.platform.contracts import AgentAssignment, AgentResult, AgentStatus, DiagnosisDraft, DiagnosisDraftPayload, DiagnosisTicket, EvidenceKind, EvidenceRef, InspectionFeatures, JevRouteDecision, ReviewDecision, TicketStatus, jsonable, utc_now
from agentic.platform.hitl import ALLOWED_HUMAN_ACTIONS, InvalidActionError
from agentic.platform.jev.router import JevRouter
from agentic.platform.llm_gateway import LLMGateway
from agentic.platform.tickets import TicketStore


class DiagnosisGraphState(TypedDict, total=False):
    run_id: str
    thread_id: str
    features: InspectionFeatures
    route_decision: JevRouteDecision
    route: str
    route_history: Annotated[List[str], operator.add]
    upgrade_count: int
    ticket: Dict[str, Any]
    context: Dict[str, Any]
    evidence_bundle: Annotated[List[Dict[str, Any]], operator.add]
    diagnosis: Dict[str, Any]
    review: Dict[str, Any]
    model_metadata: Dict[str, Any]
    human_decision: Dict[str, Any]
    result: Dict[str, Any]
    trace: Annotated[List[Dict[str, Any]], operator.add]


Worker = Callable[[Mapping[str, Any]], Mapping[str, Any]]


# Send payloads must remain checkpointable. Worker callables are therefore
# kept in a short-lived in-process registry; dynamic task state carries only
# the worker id and JSON-safe context.
_WORKER_REGISTRY: Dict[tuple[str, str], Dict[str, Worker]] = {}
_WORKER_REGISTRY_LOCK = threading.Lock()


def _worker_key(run_id: str, thread_id: str) -> tuple[str, str]:
    return (str(run_id), str(thread_id))


def _json_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop runtime-only worker callables before state/checkpoint writes."""
    filtered = {key: value for key, value in context.items() if key != "workers"}
    # Process snapshots, contracts and enum values must be plain JSON before
    # they enter Send payloads or a durable checkpointer.
    value = jsonable(filtered)
    return dict(value) if isinstance(value, Mapping) else {}


def _register_workers(run_id: str, thread_id: str, workers: Mapping[str, Worker]) -> None:
    with _WORKER_REGISTRY_LOCK:
        _WORKER_REGISTRY[_worker_key(run_id, thread_id)] = dict(workers)


def _get_worker(run_id: str, thread_id: str, agent_id: str) -> Optional[Worker]:
    with _WORKER_REGISTRY_LOCK:
        return _WORKER_REGISTRY.get(_worker_key(run_id, thread_id), {}).get(agent_id)


def _get_registered_workers(run_id: str, thread_id: str) -> Dict[str, Worker]:
    with _WORKER_REGISTRY_LOCK:
        return dict(_WORKER_REGISTRY.get(_worker_key(run_id, thread_id), {}))


def _release_workers(run_id: str, thread_id: str) -> None:
    with _WORKER_REGISTRY_LOCK:
        _WORKER_REGISTRY.pop(_worker_key(run_id, thread_id), None)


def _event(state: DiagnosisGraphState, node: str, status: str, started: float, **extra: Any) -> Dict[str, Any]:
    return {
        "run_id": state.get("run_id", ""),
        "thread_id": state.get("thread_id", ""),
        "node": node,
        "status": status,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        **extra,
    }


def _ticket_for_state(state: DiagnosisGraphState, *, route: Optional[str] = None,
                     stage: Optional[str] = None, status: Optional[str] = None,
                     worker_id: Optional[str] = None, human_decision: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Create/update the explicit Ticket envelope carried by graph state."""
    raw = state.get("ticket") or state.get("context", {}).get("ticket") or {}
    try:
        ticket = DiagnosisTicket.from_dict(raw) if raw else DiagnosisTicket(
            run_id=state.get("run_id", ""), thread_id=state.get("thread_id", ""),
            implementation="agentic.orchestration.diagnosis_graph",
        )
    except (TypeError, ValueError):
        ticket = DiagnosisTicket(
            run_id=state.get("run_id", ""), thread_id=state.get("thread_id", ""),
            implementation="agentic.orchestration.diagnosis_graph",
        )
    if route is not None:
        ticket.route = str(route)
    if stage is not None:
        ticket.stage = str(stage)
    if status is not None:
        ticket.status = str(status)
    if worker_id is not None:
        ticket.worker_id = str(worker_id)
    if human_decision is not None:
        ticket.human_decision = dict(human_decision)
    ticket.metadata.setdefault("max_collaboration_rounds", 2)
    ticket.metadata["upgrade_count"] = int(state.get("upgrade_count", 0) or 0)
    ticket.updated_at = utc_now()
    return ticket.to_dict()


def _route_update(state: DiagnosisGraphState, decision: JevRouteDecision, started: float) -> DiagnosisGraphState:
    ticket = _ticket_for_state(state, route=decision.route, stage="routed", status=TicketStatus.ASSIGNED.value,
                               worker_id="jev_router")
    return {
        "route_decision": decision.to_dict(),
        "route": decision.route,
        "route_history": [decision.route],
        "ticket": ticket,
        "trace": [_event(
            state, "jev_router", "completed", started,
            route=decision.route,
            router_backend=decision.router_backend,
            model_profile=decision.model_profile,
            gate_version=decision.gate_version,
            policy_version=decision.policy_version,
            policy_hash=decision.policy_hash,
            confidence=decision.confidence,
            reasons=decision.reasons,
            ticket_id=ticket.get("ticket_id"),
        )],
    }


def _route_decision_for_state(state: DiagnosisGraphState, router: Optional[JevRouter] = None) -> JevRouteDecision:
    """Reuse the pre-route only when it was computed for this exact feature snapshot."""
    context = state.get("context", {})
    cached = context.get("pre_route_decision") if isinstance(context, Mapping) else None
    cached_features = context.get("pre_route_features") if isinstance(context, Mapping) else None
    if isinstance(cached, Mapping) and isinstance(cached_features, Mapping):
        try:
            current = InspectionFeatures.from_dict(dict(state["features"]))
            recorded = InspectionFeatures.from_dict(dict(cached_features))
            decision = JevRouteDecision.from_dict(cached)
            if (
                current.to_dict() == recorded.to_dict()
                and decision.run_id == current.run_id
                and decision.thread_id == current.thread_id
            ):
                return decision
        except (KeyError, TypeError, ValueError):
            pass
    selected_router = router or JevRouter()
    return selected_router.route(state["features"])


def _route_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    started = time.perf_counter()
    decision = _route_decision_for_state(state)
    # Persist plain JSON in graph state so SQLite/MsgPack checkpointers do not
    # need to deserialize project-specific dataclass objects.
    return _route_update(state, decision, started)


def _simple_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    started = time.perf_counter()
    context = state.get("context", {})
    evidence = []
    if context.get("case_match"):
        evidence.append({"agent_id": "case_memory", "status": "completed", "output": context["case_match"]})
    if context.get("knowledge"):
        evidence.append({"agent_id": "knowledge_rag", "status": "completed", "output": context["knowledge"]})
    evidence.append({"agent_id": "light_diagnosis", "status": "completed", "output": {"mode": "simple_loop"}})
    return {"evidence_bundle": evidence, "ticket": _ticket_for_state(state, stage="simple_loop", status=TicketStatus.RUNNING.value, worker_id="light_diagnosis"), "trace": [_event(state, "simple_loop", "completed", started, evidence_count=len(evidence))]}


def _worker_policy(context: Mapping[str, Any], agent_id: str) -> Dict[str, Any]:
    """Resolve bounded per-worker controls with task and environment overrides."""
    policy: Dict[str, Any] = {
        "timeout_s": 30.0,
        "retries": 1,
        "backoff_seconds": 0.1,
        "max_backoff_seconds": 1.0,
        "retryable_errors": ["*"],
        "policy_id": "builtin-default",
    }
    config_path = Path(__file__).resolve().parents[2] / "config" / "worker_policies.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config, Mapping):
            defaults = config.get("default", {})
            worker_policies = config.get("workers", {})
            if isinstance(defaults, Mapping):
                policy.update(defaults)
            if isinstance(worker_policies, Mapping) and isinstance(worker_policies.get(agent_id), Mapping):
                policy.update(worker_policies[agent_id])
                policy["policy_id"] = str(worker_policies[agent_id].get("policy_id", agent_id))
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    try:
        policy["timeout_s"] = float(os.getenv("MARS_AGENT_WORKER_TIMEOUT", policy["timeout_s"]))
    except (TypeError, ValueError):
        pass
    try:
        policy["retries"] = int(os.getenv("MARS_AGENT_WORKER_RETRIES", policy["retries"]))
    except (TypeError, ValueError):
        pass
    for key in ("worker_timeout_s", "worker_retries", "worker_backoff_seconds", "worker_max_backoff_seconds"):
        if key in context:
            target = {
                "worker_timeout_s": "timeout_s",
                "worker_retries": "retries",
                "worker_backoff_seconds": "backoff_seconds",
                "worker_max_backoff_seconds": "max_backoff_seconds",
            }[key]
            policy[target] = context[key]
    specific = context.get("worker_policies", {})
    if isinstance(specific, Mapping) and isinstance(specific.get(agent_id), Mapping):
        policy.update(specific[agent_id])
        policy["policy_id"] = str(specific[agent_id].get("policy_id", agent_id))

    try:
        timeout = max(0.05, min(float(policy["timeout_s"]), 300.0))
    except (TypeError, ValueError):
        timeout = 30.0
    try:
        retries = max(0, min(int(policy["retries"]), 2))
    except (TypeError, ValueError):
        retries = 1
    try:
        backoff = max(0.0, min(float(policy["backoff_seconds"]), 5.0))
    except (TypeError, ValueError):
        backoff = 0.1
    try:
        max_backoff = max(backoff, min(float(policy["max_backoff_seconds"]), 10.0))
    except (TypeError, ValueError):
        max_backoff = max(backoff, 1.0)
    retryable = policy.get("retryable_errors", ["*"])
    if isinstance(retryable, str):
        retryable = [retryable]
    if not isinstance(retryable, (list, tuple)):
        retryable = ["*"]
    return {
        "timeout_s": timeout,
        "retries": retries,
        "backoff_seconds": backoff,
        "max_backoff_seconds": max_backoff,
        "retryable_errors": [str(item) for item in retryable if str(item)],
        "policy_id": str(policy.get("policy_id", "builtin-default")),
    }


def _agent_result(
    agent_id: str,
    *,
    run_id: str,
    thread_id: str,
    status: str,
    output: Optional[Mapping[str, Any]] = None,
    issues: Optional[List[str]] = None,
    error_code: Optional[str] = None,
    error: Optional[str] = None,
    elapsed_ms: float = 0.0,
    attempt: int = 1,
    timeout_s: Optional[float] = None,
    policy_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize each worker result through the shared AgentResult contract."""
    normalized_output = dict(output or {})
    raw_sources = normalized_output.get("source_ids", [])
    source_ids = list(dict.fromkeys(str(item) for item in raw_sources if item)) if isinstance(raw_sources, (list, tuple)) else []
    evidence_kind = {
        "visual_review": EvidenceKind.VISUAL_ROI.value,
        "evidence_quality": EvidenceKind.DETECTOR_OUTPUT.value,
        "knowledge_rag": EvidenceKind.KNOWLEDGE.value,
        "case_compare": EvidenceKind.UNKNOWN.value,
        "process_state": EvidenceKind.PROCESS_STATE.value,
    }.get(agent_id, EvidenceKind.UNKNOWN.value)
    evidence_refs = []
    for source_id in source_ids:
        ref = EvidenceRef(
            kind=evidence_kind,
            source_id=source_id,
            locator=f"agent:{agent_id}",
            run_id=run_id,
            thread_id=thread_id,
            agent_id=agent_id,
            implementation="agentic.agents.specialists",
        )
        try:
            ref.validate()
        except ValueError:
            continue
        evidence_refs.append(ref)
    normalized_output["evidence_ids"] = [item.source_id for item in evidence_refs]
    metadata: Dict[str, Any] = {
        "attempt": int(attempt),
        "retry_count": max(0, int(attempt) - 1),
        "execution_mode": "cooperative_daemon_thread",
    }
    if timeout_s is not None:
        metadata["timeout_s"] = float(timeout_s)
    if policy_metadata:
        metadata.update(dict(policy_metadata))
    if error_code:
        metadata["error_code"] = error_code
    if error:
        metadata["error"] = error
    contract = AgentResult(
        agent_id=agent_id,
        status=status,
        output=normalized_output,
        evidence=evidence_refs,
        issues=list(issues or ([error_code] if error_code else [])),
        metadata=metadata,
        run_id=run_id,
        thread_id=thread_id,
        implementation="agentic.orchestration.diagnosis_graph",
    )
    payload = contract.to_dict()
    # Keep operational fields at the top level for existing trace/UI readers.
    payload["elapsed_ms"] = round(float(elapsed_ms), 1)
    payload["attempt"] = int(attempt)
    if error_code:
        payload["error_code"] = error_code
    if error:
        payload["error"] = error
    return payload


def _run_worker(
    agent_id: str,
    worker: Worker,
    context: Mapping[str, Any],
    *,
    run_id: str = "",
    thread_id: str = "",
    timeout_s: Optional[float] = None,
    retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a child Agent with a bounded timeout and at most two retries."""
    policy = _worker_policy(context, agent_id)
    timeout_s = policy["timeout_s"] if timeout_s is None else max(0.05, min(float(timeout_s), 300.0))
    retries = policy["retries"] if retries is None else max(0, min(int(retries), 2))
    run_id = run_id or str(context.get("run_id", ""))
    thread_id = thread_id or str(context.get("thread_id", ""))
    # Direct unit callers may exercise a worker without a full task envelope;
    # keep the shared contract valid while still making the missing envelope
    # visible through the generated implementation identifiers.
    run_id = run_id or f"run-worker-{agent_id}"
    thread_id = thread_id or f"inspection_default_default_{run_id}"
    started = time.perf_counter()
    last_error: Optional[tuple[str, str]] = None
    retry_delays: list[float] = []
    policy_metadata = {
        "retry_policy_id": policy["policy_id"],
        "max_retries": retries,
        "retryable_errors": list(policy["retryable_errors"]),
        "backoff_seconds": policy["backoff_seconds"],
        "max_backoff_seconds": policy["max_backoff_seconds"],
    }
    for attempt in range(1, retries + 2):
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        worker_context = dict(context)
        worker_context["_cancel_event"] = cancel_event

        def invoke() -> None:
            try:
                result_queue.put((True, worker(worker_context)))
            except Exception as exc:
                result_queue.put((False, exc))

        thread = threading.Thread(target=invoke, name=f"worker-{agent_id}-{attempt}", daemon=True)
        thread.start()
        try:
            succeeded, raw_output = result_queue.get(timeout=timeout_s)
            if not succeeded:
                raise raw_output
            output = dict(raw_output or {})
            elapsed = (time.perf_counter() - started) * 1000
            return _agent_result(
                agent_id,
                run_id=run_id,
                thread_id=thread_id,
                status=AgentStatus.COMPLETED.value,
                output=output,
                elapsed_ms=elapsed,
                attempt=attempt,
                timeout_s=timeout_s,
                policy_metadata={**policy_metadata, "retry_delays_seconds": list(retry_delays)},
            )
        except queue.Empty:
            cancel_event.set()
            last_error = ("timeout", f"worker exceeded {timeout_s:.3g}s timeout")
        except Exception as exc:  # child errors are isolated from fan-in
            last_error = (type(exc).__name__, str(exc))
        retryable_errors = set(policy["retryable_errors"])
        if attempt <= retries and ("*" in retryable_errors or last_error[0] in retryable_errors):
            delay = min(policy["backoff_seconds"] * (2 ** (attempt - 1)), policy["max_backoff_seconds"])
            retry_delays.append(round(delay, 4))
            if delay:
                time.sleep(delay)
            continue
        break
    error_code, error = last_error or ("worker_failed", "worker failed without an exception")
    return _agent_result(
        agent_id,
        run_id=run_id,
        thread_id=thread_id,
        status=AgentStatus.FAILED.value,
        error_code=error_code,
        error=error,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        attempt=attempt,
        timeout_s=timeout_s,
        policy_metadata={**policy_metadata, "retry_delays_seconds": list(retry_delays)},
    )


def _default_workers() -> Dict[str, Worker]:
    return build_full_loop_workers()


def _full_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    """Fallback fan-out used when LangGraph/Send is unavailable."""
    started = time.perf_counter()
    context = state.get("context", {})
    workers = context.get("workers") or _get_registered_workers(state.get("run_id", ""), state.get("thread_id", "")) or _default_workers()
    results: List[Dict[str, Any]] = []
    max_workers = min(5, max(1, len(workers)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="full-loop") as executor:
        futures = [executor.submit(_run_worker, agent_id, worker, context, run_id=state.get("run_id", ""), thread_id=state.get("thread_id", "")) for agent_id, worker in workers.items()]
        for future in futures:
            results.append(future.result())
    return {
        "evidence_bundle": results,
        "trace": [_event(state, "full_loop_fanout", "completed", started, child_count=len(results), failed=sum(item["status"] == "failed" for item in results))],
    }


def _full_dispatch_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    """Register workers and emit a trace before dynamic ``Send`` fan-out."""
    started = time.perf_counter()
    context = state.get("context", {})
    run_id = state.get("run_id", "")
    thread_id = state.get("thread_id", "")
    workers = context.get("workers") or _get_registered_workers(run_id, thread_id) or _default_workers()
    _register_workers(run_id, thread_id, workers)
    ticket = _ticket_for_state(state, stage="fan_out", status=TicketStatus.RUNNING.value, worker_id="full_loop_dispatch")
    ticket_obj = DiagnosisTicket.from_dict(ticket)
    existing = {item.agent_id for item in ticket_obj.assignments}
    for agent_id in workers:
        if agent_id not in existing:
            ticket_obj.assignments.append(AgentAssignment(
                ticket_id=ticket_obj.ticket_id, agent_id=agent_id, role=agent_id,
                status="assigned", model_profile=str(context.get("llm_profile", "")),
                run_id=state.get("run_id", ""), thread_id=state.get("thread_id", ""),
            ))
    ticket_obj.stage = "fan_out"
    ticket_obj.status = TicketStatus.RUNNING.value
    ticket_obj.worker_id = "full_loop_dispatch"
    ticket = ticket_obj.to_dict()
    ticket_root = context.get("ticket_store_root") if isinstance(context, Mapping) else None
    if ticket_root:
        try:
            store = TicketStore(ticket_root)
            persisted = store.by_run(run_id)
            if persisted is not None:
                for assignment in ticket_obj.assignments:
                    if not any(item.assignment_id == assignment.assignment_id or item.agent_id == assignment.agent_id for item in persisted.assignments):
                        store.assign(persisted, agent_id=assignment.agent_id, role=assignment.role, model_profile=assignment.model_profile)
                        persisted = store.by_run(run_id) or persisted
                ticket = persisted.to_dict()
        except (OSError, TypeError, ValueError):
            pass
    return {
        "ticket": ticket,
        "trace": [_event(
            state,
            "full_loop_dispatch",
            "started",
            started,
            child_count=len(workers),
            fanout="langgraph_send",
        )]
    }


def _full_send_targets(state: DiagnosisGraphState) -> List[Any]:
    """Map each registered worker to a dynamic LangGraph task.

    The task payload intentionally excludes callable objects so that graph
    checkpoints remain serializable. The worker registry is process-local and
    is released by ``finalize`` once all fan-in work has completed.
    """
    if Send is None:  # pragma: no cover - guarded by the builder
        return []
    context = state.get("context", {})
    run_id = state.get("run_id", "")
    thread_id = state.get("thread_id", "")
    workers = context.get("workers") or _get_registered_workers(run_id, thread_id) or _default_workers()
    # Direct ``build_diagnosis_graph`` callers may bypass run_diagnosis_graph.
    # Register here as well so custom test workers still execute.
    _register_workers(run_id, thread_id, workers)
    task_context = _json_context(context)
    return [
        Send(
            "full_loop_worker",
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "worker_id": agent_id,
                "worker_context": task_context,
            },
        )
        for agent_id in workers
    ]


def _full_worker_node(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Run one dynamic worker and reduce its result into ``evidence_bundle``."""
    started = time.perf_counter()
    run_id = str(task.get("run_id", ""))
    thread_id = str(task.get("thread_id", ""))
    agent_id = str(task.get("worker_id", ""))
    worker = _get_worker(run_id, thread_id, agent_id) or _default_workers().get(agent_id)
    if worker is None:
        result = _agent_result(
            agent_id,
            run_id=run_id,
            thread_id=thread_id,
            status=AgentStatus.FAILED.value,
            error_code="worker_not_found",
            error=f"worker {agent_id!r} is not registered",
        )
    else:
        result = _run_worker(
            agent_id,
            worker,
            task.get("worker_context", {}),
            run_id=run_id,
            thread_id=thread_id,
        )
    return {
        # ``operator.add`` on DiagnosisGraphState.evidence_bundle is the
        # fan-in reducer: every Send worker contributes one independent item.
        "evidence_bundle": [result],
        "trace": [{
            "run_id": run_id,
            "thread_id": thread_id,
            "node": f"full_loop_worker:{agent_id}",
            "status": result.get("status", "failed"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "agent_id": agent_id,
            "attempt": result.get("attempt", 1),
            **({"error_code": result.get("error_code")} if result.get("error_code") else {}),
        }],
    }


def _full_join_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    """Barrier after dynamic workers have reduced into the parent state."""
    started = time.perf_counter()
    bundle = state.get("evidence_bundle", [])
    context = state.get("context", {})
    failed_count = sum(item.get("status") == "failed" for item in bundle)
    ticket = _ticket_for_state(
        state, stage="fan_in", status=TicketStatus.WAITING_REVIEW.value,
        worker_id="full_loop_join",
    )
    ticket_obj = DiagnosisTicket.from_dict(ticket)
    ticket_obj.health = "degraded" if failed_count else "healthy"
    ticket_obj.retry_count = max([int(item.get("retry_count", 0) or 0) for item in bundle] or [0])
    ticket = ticket_obj.to_dict()
    # Persist per-worker attempts and health when the demo/runtime supplies a
    # TicketStore root.  This is done at the fan-in barrier to avoid concurrent
    # append races between Send workers.
    ticket_root = context.get("ticket_store_root") if isinstance(context, Mapping) else None
    if ticket_root:
        try:
            store = TicketStore(ticket_root)
            persisted = store.by_run(state.get("run_id", ""))
            if persisted is not None:
                for item in bundle:
                    assignment = next((candidate for candidate in persisted.assignments if candidate.agent_id == item.get("agent_id")), None)
                    if assignment is not None:
                        persisted = store.worker_result(
                            persisted, assignment.assignment_id,
                            status=str(item.get("status", AgentStatus.FAILED.value)),
                            attempt=int(item.get("attempt", 1) or 1),
                            error_code=str(item.get("error_code", "") or ""),
                        )
                ticket = persisted.to_dict()
        except (OSError, TypeError, ValueError):
            pass
    return {
        "ticket": ticket,
        "trace": [_event(
            state,
            "full_loop_fanout",
            "completed",
            started,
            child_count=len(bundle),
            failed=failed_count,
            fanin="reducer_operator_add",
        )]
    }


def _empty_node(state: DiagnosisGraphState, node: str) -> DiagnosisGraphState:
    started = time.perf_counter()
    return {"ticket": _ticket_for_state(state, stage=node, status=TicketStatus.WAITING_REVIEW.value, worker_id=node), "trace": [_event(state, node, "completed", started, requires_human_review=True)]}


def _simple_upgrade_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    """Escalate a simple-loop draft once when deterministic review finds gaps."""
    started = time.perf_counter()
    decision = dict(state.get("route_decision") or {})
    reasons = list(decision.get("reasons", []))
    if "simple_loop_review_upgrade" not in reasons:
        reasons.append("simple_loop_review_upgrade")
    decision.update({"route": "full_loop", "reasons": reasons, "gate_version": f"{decision.get('gate_version', 'alu-jev-safety-0.1')}:upgrade"})
    ticket = _ticket_for_state(state, route="full_loop", stage="upgrade", status=TicketStatus.RUNNING.value, worker_id="jev_router")
    return {
        "route": "full_loop",
        "route_history": ["full_loop"],
        "route_decision": decision,
        "upgrade_count": int(state.get("upgrade_count", 0) or 0) + 1,
        "ticket": ticket,
        "trace": [_event(state, "simple_loop_upgrade", "completed", started, reason="review_evidence_gap", route="full_loop")],
    }


def _review_next_target(state: DiagnosisGraphState, *, human_review: bool) -> str:
    review = state.get("review", {}) or {}
    context = state.get("context", {}) or {}
    has_runtime_evidence = bool(context.get("visual") or context.get("knowledge") or context.get("cases"))
    if state.get("route") == "simple_loop" and has_runtime_evidence and int(state.get("upgrade_count", 0) or 0) < 1:
        action = str(review.get("action", ""))
        if action in {"request_more_evidence", "human_review"} or review.get("issues"):
            return "simple_upgrade"
    return "human_review" if human_review else "finalize"


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return []


def _agent_prompt_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    agent_id = str(result.get("agent_id", "unknown"))
    output = result.get("output", {}) if isinstance(result.get("output"), Mapping) else {}
    summary: dict[str, Any] = {"agent_id": agent_id, "status": result.get("status", "unknown")}
    if agent_id == "knowledge_rag":
        summary["analysis_status"] = output.get("analysis_status", "unknown")
        summary["issues"] = list(output.get("issues", []))
        summary["sources"] = [
            {key: item.get(key) for key in ("source_id", "title", "source_version", "text", "applicability", "risk_level") if item.get(key) is not None}
            for item in output.get("sources", [])[:4] if isinstance(item, Mapping)
        ]
        for source in summary["sources"]:
            if isinstance(source.get("text"), str):
                source["text"] = source["text"][:700]
    elif agent_id == "case_compare":
        summary["analysis_status"] = output.get("analysis_status", "unknown")
        summary["issues"] = list(output.get("issues", []))
        summary["matches"] = [
            {key: item.get(key) for key in ("case_id", "similarity", "applicability", "conclusion", "source_id") if item.get(key) is not None}
            for item in output.get("matches", [])[:3] if isinstance(item, Mapping)
        ]
    elif agent_id == "process_state":
        summary["process_analysis"] = {
            key: output[key]
            for key in ("analysis_status", "process_regime", "event_risk", "anomaly_score", "process_tags", "evidence_window", "process_missing", "alignment_quality", "video_relation", "variable_count", "issues", "source_ids")
            if key in output
        }
    elif agent_id == "visual_review":
        summary["visual_review"] = {
            key: output[key]
            for key in ("analysis_status", "label", "class_probabilities", "confidence", "uncertainty", "temporal_consistency", "frame_ids", "key_frame_ids", "source_ids", "issues")
            if key in output
        }
    elif agent_id == "evidence_quality":
        summary["evidence_quality"] = {
            key: output[key]
            for key in ("analysis_status", "quality_status", "quality_issues", "missing_requirements", "frame_count", "roi_coverage", "source_ids", "issues")
            if key in output
        }
    else:
        summary["specialist_findings"] = {
            key: output[key]
            for key in ("analysis_status", "issues", "source_ids", "missing_requirements", "retrieval_mode", "query", "process_boundary", "process_missing", "alignment_quality", "video_relation", "process_regime", "event_risk", "variable_count")
            if key in output
        }
    return summary


def _aggregate_model_call(state: DiagnosisGraphState, bundle: list[Mapping[str, Any]]) -> tuple[Optional[Mapping[str, Any]], dict[str, Any]]:
    context = state.get("context", {})
    route = state.get("route", "human")
    profile = str(context.get("llm_profile", "")).strip()
    if route != "full_loop" or not profile:
        reason = "skipped_by_jev" if route != "full_loop" else "profile_not_configured"
        return None, {"status": reason, "profile": profile, "provider": "none", "model": "", "request_id": f"skip-{state.get('run_id', '')}", "latency_ms": 0.0, "retries": 0, "error_type": None, "error": None, "usage": {}}

    visual = context.get("visual", {})
    quality = context.get("quality", {})
    knowledge = context.get("knowledge", [])
    cases = context.get("cases", [])
    process_state = context.get("process_state", {"process_missing": ["process_snapshot"], "alignment_quality": "unknown"})
    visual_summary = {key: visual.get(key) for key in ("source_id", "label", "class_probs", "score", "confidence", "uncertainty", "risk_level", "frame_ids", "key_frame_ids", "temporal_consistency", "model_profile", "metadata") if isinstance(visual, Mapping) and key in visual}
    quality_summary = {key: quality.get(key) for key in ("quality_status", "issues", "blur_score", "brightness_score", "overexposure_ratio", "target_present", "resolution_status", "cross_frame_stability") if isinstance(quality, Mapping) and key in quality}
    process_summary = {key: process_state.get(key) for key in ("process_regime", "event_risk", "anomaly_score", "change_points", "process_tags", "evidence_window", "process_missing", "alignment_quality", "variable_groups") if isinstance(process_state, Mapping) and key in process_state}
    prompt_context = {
        "route": route,
        "parallel_agent_results": [_agent_prompt_summary(item) for item in bundle if item.get("agent_id") != "llm_gateway"],
        "visual_evidence": visual_summary,
        "image_quality": quality_summary,
        "knowledge_passages": [{key: (str(item.get(key))[:320] if key == "text" else item.get(key)) for key in ("source_id", "title", "source_version", "text", "applicability", "risk_level") if item.get(key) is not None} for item in knowledge[:3] if isinstance(item, Mapping)],
        "approved_case_matches": [{key: (str(item.get(key))[:320] if key == "conclusion" else item.get(key)) for key in ("case_id", "similarity", "applicability", "conclusion", "source_id") if item.get(key) is not None} for item in cases[:2] if isinstance(item, Mapping)],
        "process_state_prior": process_summary,
    }
    # Keep local vision models responsive while preserving all identifiers used
    # for evidence validation. Large raw RAG/process payloads belong in the
    # trace, not in the model's generation context.
    compact_context = json.dumps(prompt_context, ensure_ascii=False, default=str)
    context_budget = 1800 if profile == "ollama_local" else 6500
    if len(compact_context) > context_budget:
        compact_context = compact_context[:context_budget] + "...[context_truncated]"
    prompt_variant = str(context.get("prompt_variant", "diagnosis-aggregator-v2"))
    prompt_opening = {
        "diagnosis-aggregator-v2": "先按视觉证据、知识引用和过程先验逐项核对，再生成简洁研判草案。",
        "diagnosis-evidence-first-v1": "先列出可核验的视觉证据与证据缺口，再生成简洁研判草案。",
    }.get(prompt_variant, "先按视觉证据、知识引用和过程先验逐项核对，再生成简洁研判草案。")
    message = {
        "role": "user",
        "content": (
            "你是铝电解火眼视觉研判 Agent。" + prompt_opening
            + "根据输入图像和并行 Agent 的结构化证据，生成简洁研判草案。"
            "按 JSON 返回 diagnosis, diagnosis_class, risk_level(low/medium/high/unknown), confidence, summary, "
            "findings[], recommendations[], inspection_items[], open_questions[], source_ids[], requires_human_review。"
            "引用 source_ids 时只能选择输入中的真实 source_id。C/N/H 是视觉代理类别，不是实测过热度。"
            "过程状态与视频没有可靠对齐时，只能作为独立先验。不能输出摄氏温度、虚构工艺参数或生产控制指令。"
            "结论应聚焦火眼图像和可追溯工艺知识，不输出与铝电解无关内容。必须保留人工复核。输入证据：" + compact_context
        ),
    }
    started = time.perf_counter()
    response = LLMGateway().chat(
        profile,
        [message],
        images=list(context.get("gateway_images", []))[:4],
        response_schema=DiagnosisDraftPayload,
        trace_context={"run_id": state.get("run_id", ""), "thread_id": state.get("thread_id", ""), "agent_id": "diagnosis_aggregator"},
    )
    primary_status = response.status
    primary_error = response.error
    primary_latency = response.latency_ms
    # Qwen-VL models on constrained local hardware can exhaust a long
    # multimodal context without emitting JSON. Retry once with a bounded
    # evidence-only prompt; the original failure stays in metadata.
    if response.status != "completed" and profile == "ollama_local" and response.error == "provider_empty_response":
        source_id = str(visual.get("source_id") or context.get("media_source_id") or "visual_evidence")
        fallback_message = {
            "role": "user",
            "content": (
                "Return a compact JSON DiagnosisDraft for this aluminum electrolysis fire-eye image. "
                "Describe visible evidence only. Do not infer temperature, chemistry, process values, or control actions. "
                f"Use source_ids=[\"{source_id}\"], risk_level=unknown, requires_human_review=true, "
                "and include diagnosis, diagnosis_class, summary, findings, recommendations, inspection_items, open_questions. "
            ),
        }
        response = LLMGateway().chat(
            profile,
            [fallback_message],
            images=list(context.get("gateway_images", []))[:1],
            response_schema=DiagnosisDraftPayload,
            trace_context={"run_id": state.get("run_id", ""), "thread_id": state.get("thread_id", ""), "agent_id": "diagnosis_aggregator_fallback"},
        )
    metadata = {
        "status": response.status,
        "profile": response.profile,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "latency_ms": response.latency_ms,
        "retries": response.retries,
        "error_type": response.error_type,
        "error": response.error,
        "usage": response.usage,
        "capabilities": list(response.capabilities),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "prompt_variant": prompt_variant,
    }
    if primary_status != response.status or primary_error:
        metadata.update({"primary_status": primary_status, "primary_error": primary_error, "primary_latency_ms": primary_latency, "fallback_prompt": True})
    return response.content if response.status == "completed" and isinstance(response.content, Mapping) else None, metadata


def _model_draft(
    payload: Mapping[str, Any],
    state: DiagnosisGraphState,
    route_risk: str,
    route_confidence: float,
    gateway_meta: Mapping[str, Any],
) -> DiagnosisDraft:
    context = state.get("context", {})
    knowledge_ids = {str(item.get("source_id")) for item in context.get("knowledge", []) if isinstance(item, Mapping) and item.get("source_id")}
    visual_ids = {str(context.get("visual", {}).get("source_id")), str(context.get("media_source_id", ""))}
    case_ids = {str(item.get("case_id")) for item in context.get("cases", []) if isinstance(item, Mapping) and item.get("case_id")}
    source_metadata = {
        str(item.get("source_id")): item for item in context.get("knowledge", [])
        if isinstance(item, Mapping) and item.get("source_id")
    }
    source_ids = _strings(payload.get("source_ids"))
    refs: list[EvidenceRef] = []
    for source_id in source_ids:
        if source_id not in {item.source_id for item in refs}:
            kind = EvidenceKind.VISUAL_ROI.value if source_id in visual_ids else EvidenceKind.KNOWLEDGE.value if source_id in knowledge_ids else EvidenceKind.CASE.value if source_id in case_ids else EvidenceKind.UNKNOWN.value
            locator = str(source_metadata.get(source_id, {}).get("locator", "model_citation"))
            refs.append(EvidenceRef(kind=kind, source_id=source_id, locator=locator, run_id=state.get("run_id", ""), thread_id=state.get("thread_id", "")))
    for item in payload.get("evidence", []) if isinstance(payload.get("evidence"), list) else []:
        if not isinstance(item, Mapping) or not item.get("source_id"):
            continue
        source_id = str(item["source_id"])
        if source_id not in {ref.source_id for ref in refs}:
            candidate_kind = str(item.get("kind", EvidenceKind.UNKNOWN.value))
            valid_kinds = {item.value for item in EvidenceKind}
            refs.append(EvidenceRef(kind=candidate_kind if candidate_kind in valid_kinds else EvidenceKind.UNKNOWN.value, source_id=source_id, locator=str(item.get("locator", "model_evidence")), run_id=state.get("run_id", ""), thread_id=state.get("thread_id", "")))

    findings = _strings(payload.get("findings")) or _strings(payload.get("visual_evidence"))
    summary = str(payload.get("summary") or payload.get("visual_summary") or "")
    if not summary and findings:
        summary = findings[0]
    if not findings and summary:
        findings = [summary]
    recommendations = _strings(payload.get("recommendations")) or _strings(payload.get("follow_up_checks"))
    inspections = _strings(payload.get("inspection_items"))
    questions = _strings(payload.get("open_questions")) or _strings(payload.get("uncertainties"))
    risk = str(payload.get("risk_level", payload.get("risk_interpretation", route_risk))).lower()
    if risk not in {"low", "medium", "high", "unknown"}:
        risk = route_risk if route_risk in {"low", "medium", "high", "unknown"} else "unknown"
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", route_confidence))))
    except (TypeError, ValueError):
        confidence = min(1.0, max(0.0, float(route_confidence)))
    model_fields = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
    model_fields.update({"llm_status": gateway_meta.get("status", "unknown"), "provider": gateway_meta.get("provider", ""), "model": gateway_meta.get("model", ""), "loop": state.get("route", "full_loop")})
    return DiagnosisDraft(
        diagnosis=str(payload.get("diagnosis", payload.get("diagnosis_class", "visual_state"))),
        diagnosis_class=str(payload.get("diagnosis_class", "visual_state")),
        risk_level=risk,
        confidence=confidence,
        summary=summary or "Evidence aggregation requires operator review.",
        evidence=refs,
        findings=findings,
        recommendations=recommendations,
        inspection_items=inspections,
        open_questions=questions,
        source_ids=source_ids,
        requires_human_review=True,
        metadata=model_fields,
        schema_version="diagnosis-draft-1",
        run_id=state.get("run_id", ""),
        thread_id=state.get("thread_id", ""),
        agent_id="diagnosis_aggregator",
        model_profile=str(gateway_meta.get("profile", "")),
        implementation="agentic.orchestration.diagnosis_graph",
    )


def _aggregate_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    started = time.perf_counter()
    bundle = state.get("evidence_bundle", [])
    failed = [item for item in bundle if item.get("status") == "failed"]
    route_decision = state.get("route_decision") or {}
    risk_level = route_decision.get("risk_level", "unknown") if isinstance(route_decision, Mapping) else route_decision.risk_level
    confidence = route_decision.get("confidence", 0.0) if isinstance(route_decision, Mapping) else route_decision.confidence
    model_payload, gateway_meta = _aggregate_model_call(state, bundle)
    if model_payload is not None:
        try:
            draft = _model_draft(model_payload, state, risk_level, confidence, gateway_meta)
        except Exception as exc:
            model_payload = None
            gateway_meta = {**gateway_meta, "status": "schema_error", "error_type": type(exc).__name__, "error": str(exc)}
    if model_payload is None:
        context = state.get("context", {})
        visual = context.get("visual", {})
        media_source = visual.get("source_id") or context.get("media_source_id")
        evidence = []
        if media_source:
            evidence.append(EvidenceRef(kind=EvidenceKind.VISUAL_ROI.value, source_id=str(media_source), locator="visual_evidence", confidence=float(visual.get("confidence", visual.get("score", 0.0)) or 0.0), run_id=state.get("run_id", ""), thread_id=state.get("thread_id", "")))
        llm_status = str(gateway_meta.get("status", "unknown"))
        fallback_status = f"fallback_{llm_status}" if llm_status != "skipped_by_jev" else llm_status
        normalized_risk = risk_level if risk_level in {"low", "medium", "high", "unknown"} else "unknown"
        draft = DiagnosisDraft(
            run_id=state.get("run_id", ""),
            thread_id=state.get("thread_id", ""),
            diagnosis_class="visual_state",
            risk_level=normalized_risk,
            confidence=min(1.0, max(0.0, float(confidence or 0.0))),
            summary=f"{state.get('route', 'human')} evidence aggregation",
            evidence=evidence,
            findings=[f"evidence_agents={len(bundle)}", f"failed_agents={len(failed)}"],
            recommendations=["Review the visual evidence and cited process knowledge."],
            requires_human_review=True,
            metadata={"loop": state.get("route", "human"), "failed_agents": [item.get("agent_id") for item in failed], "llm_status": fallback_status},
            model_profile=str(gateway_meta.get("profile", "")),
            implementation="agentic.orchestration.diagnosis_graph",
        )
    gateway_status = "completed" if gateway_meta.get("status") == "completed" else "failed" if str(gateway_meta.get("status", "")).startswith(("provider_error", "configuration_error", "schema_error")) else "skipped"
    gateway_result = {"agent_id": "llm_gateway", "status": gateway_status, "output": {"metadata": gateway_meta, "content": dict(model_payload or {})}}
    llm_trace = _event(state, "diagnosis_aggregator_llm", gateway_status, started, provider=gateway_meta.get("provider"), profile=gateway_meta.get("profile"), request_id=gateway_meta.get("request_id"), elapsed_ms=gateway_meta.get("latency_ms", 0.0))
    return {
        "diagnosis": draft.to_dict(),
        "model_metadata": {"llm_gateway": gateway_meta},
        "evidence_bundle": [gateway_result],
        "trace": [llm_trace, _event(state, "diagnosis_aggregator", "completed", started, evidence_count=len(bundle), failed_count=len(failed), model_status=gateway_meta.get("status"))],
    }


def _review_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    started = time.perf_counter()
    context = state.get("context", {})
    known = {str(item.get("source_id")) for item in context.get("knowledge", []) if item.get("source_id")}
    known.update(str(item.get("case_id")) for item in context.get("cases", []) if item.get("case_id"))
    visual_source = context.get("visual", {}).get("source_id") or context.get("media_source_id")
    if visual_source:
        known.add(str(visual_source))
    known.update(str(item) for item in context.get("known_source_ids", []) if item)
    review = review_draft(state.get("diagnosis", {}), known_source_ids=known)
    return {"review": review, "trace": [_event(state, "evidence_review", "completed", started, action=review.get("action"), issue_count=len(review.get("issues", [])))]}


def _finalize_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    started = time.perf_counter()
    route = state.get("route", "human")
    human_decision = state.get("human_decision")
    if human_decision:
        action = human_decision.get("action")
        status = {"approve": "approved", "amend": "approved", "reject": "rejected", "request_retake": "retake_requested"}.get(action, "needs_human_review")
        human_required = False
    else:
        status = "needs_human_review" if route in {"human", "retake"} or not state.get("review", {}).get("approved") else "evidence_review_passed"
        human_required = True
    ticket_status = TicketStatus.COMPLETED.value if human_decision else TicketStatus.WAITING_REVIEW.value
    ticket = _ticket_for_state(state, stage="completed" if human_decision else "human_review", status=ticket_status,
                               worker_id="human_reviewer" if human_decision else "orchestrator", human_decision=human_decision)
    ticket_obj = DiagnosisTicket.from_dict(ticket)
    ticket_obj.report_id = f"report-{state.get('run_id', '')}"
    ticket = ticket_obj.to_dict()
    result = {
        "status": status,
        "route": route,
        "initial_route": (state.get("route_history") or [route])[0],
        "route_history": list(state.get("route_history") or [route]),
        "upgrade_count": int(state.get("upgrade_count", 0) or 0),
        "route_decision": state.get("route_decision").to_dict() if hasattr(state.get("route_decision"), "to_dict") else (state.get("route_decision") or {}),
        "diagnosis": state.get("diagnosis", {}),
        "review": state.get("review", {}),
        "evidence_bundle": state.get("evidence_bundle", []),
        "human_review_required": human_required,
        "ticket": ticket,
    }
    if human_decision:
        result["human_decision"] = jsonable(human_decision)
    _release_workers(state.get("run_id", ""), state.get("thread_id", ""))
    return {"ticket": ticket, "result": result, "trace": [_event(state, "finalize", "completed", started, result_status=result["status"], ticket_id=ticket.get("ticket_id"))]}


def _human_review_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
    """LangGraph interrupt node used when a real checkpointer is supplied."""
    if langgraph_interrupt is None:
        return {"trace": [_event(state, "human_review", "waiting", time.perf_counter(), allowed_actions=list(ALLOWED_HUMAN_ACTIONS))]}
    payload = {
        "run_id": state.get("run_id", ""),
        "thread_id": state.get("thread_id", ""),
        "draft": state.get("diagnosis", {}),
        "evidence": state.get("evidence_bundle", []),
        "issues": state.get("review", {}).get("issues", []),
        "review": state.get("review", {}),
        "route": state.get("route", "human"),
        "model_metadata": state.get("context", {}).get("model_metadata", {}),
        "allowed_actions": list(ALLOWED_HUMAN_ACTIONS),
    }
    decision = langgraph_interrupt(payload)
    values = dict(decision or {}) if isinstance(decision, Mapping) else {}
    action = str(values.get("action", ""))
    if action not in ALLOWED_HUMAN_ACTIONS:
        raise InvalidActionError(f"invalid human review action: {action!r}")
    review_decision = ReviewDecision(
        run_id=state.get("run_id", ""),
        thread_id=state.get("thread_id", ""),
        action=action,
        reviewer_id=str(values.get("reviewer_id", "local_operator")),
        reason=str(values.get("reason", "")),
        amendments=dict(values.get("amendments", {})),
        evidence_ids=list(values.get("evidence_ids", [])),
    )
    review_decision.validate()
    diagnosis = dict(state.get("diagnosis", {}))
    if action == "amend":
        diagnosis.update(review_decision.amendments)
    return {
        "diagnosis": diagnosis,
        "human_decision": review_decision.to_dict(),
        "trace": [_event(state, "human_review", "completed", time.perf_counter(), action=action)],
    }


def _route_target(state: DiagnosisGraphState) -> str:
    route = state.get("route", "human")
    return {"simple_loop": "simple_loop", "full_loop": "full_loop", "retake": "retake", "human": "human"}.get(route, "human")


def _langgraph_route_target(state: DiagnosisGraphState) -> str:
    """Route map for the LangGraph topology, including the Send dispatcher."""
    target = _route_target(state)
    return "full_loop_dispatch" if target == "full_loop" else target


class _FallbackCompiledGraph:
    """Small sequential graph runner for environments without LangGraph."""

    def __init__(self, router: Optional[JevRouter] = None, human_review: bool = False):
        self.router = router
        self.human_review = human_review

    def invoke(self, payload: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state: DiagnosisGraphState = dict(payload)  # type: ignore[assignment]
        def apply(update: Mapping[str, Any]) -> None:
            for key, value in update.items():
                if key == "trace":
                    state["trace"] = list(state.get("trace", [])) + list(value or [])
                elif key == "evidence_bundle" and key in state:
                    state[key] = list(state.get(key, [])) + list(value or [])
                else:
                    state[key] = value
        for fn in (_route_node,):
            update = fn(state) if self.router is None else _route_update(
                state, _route_decision_for_state(state, self.router), time.perf_counter()
            )
            decision = update["route_decision"]
            update["route"] = decision.get("route") if isinstance(decision, Mapping) else decision.route
            apply(update)
        target = _route_target(state)
        node_update = {
            "simple_loop": _simple_node,
            "full_loop": _full_node,
            "retake": lambda value: _empty_node(value, "retake"),
            "human": lambda value: _empty_node(value, "human"),
        }[target](state)
        apply(node_update)
        for fn in (_aggregate_node, _review_node):
            apply(fn(state))
        if _review_next_target(state, human_review=self.human_review) == "simple_upgrade":
            apply(_simple_upgrade_node(state))
            apply(_full_node(state))
            apply(_aggregate_node(state))
            apply(_review_node(state))
        if self.human_review:
            apply(_human_review_node(state))
        apply(_finalize_node(state))
        return dict(state)


def build_diagnosis_graph(*, router: Optional[JevRouter] = None, checkpointer: Any = None, human_review: bool = False):
    if StateGraph is None:
        return _FallbackCompiledGraph(router=router, human_review=human_review)
    graph = StateGraph(DiagnosisGraphState)
    if router is None:
        route_node = _route_node
    else:
        def route_node(state: DiagnosisGraphState) -> DiagnosisGraphState:
            started = time.perf_counter()
            decision = _route_decision_for_state(state, router)
            return _route_update(state, decision, started)
    graph.add_node("route", route_node)
    graph.add_node("simple_loop", _simple_node)
    # LangGraph's native dynamic fan-out is the primary implementation. The
    # ThreadPoolExecutor implementation remains in _FallbackCompiledGraph.
    use_send = Send is not None
    if use_send:
        graph.add_node("full_loop_dispatch", _full_dispatch_node)
        graph.add_node("full_loop_worker", _full_worker_node)
        graph.add_node("full_loop_join", _full_join_node)
    else:  # pragma: no cover - StateGraph is normally paired with Send
        graph.add_node("full_loop", _full_node)
    graph.add_node("retake", lambda state: _empty_node(state, "retake"))
    graph.add_node("human", lambda state: _empty_node(state, "human"))
    graph.add_node("simple_upgrade", _simple_upgrade_node)
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("review", _review_node)
    if human_review:
        graph.add_node("human_review", _human_review_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "route")
    route_map = {"simple_loop": "simple_loop", "full_loop_dispatch": "full_loop_dispatch", "retake": "retake", "human": "human"}
    if not use_send:
        route_map["full_loop"] = "full_loop"
    graph.add_conditional_edges("route", _langgraph_route_target if use_send else _route_target, route_map)
    if use_send:
        graph.add_conditional_edges("full_loop_dispatch", _full_send_targets)
        graph.add_edge("full_loop_worker", "full_loop_join")
        graph.add_edge("full_loop_join", "aggregate")
        route_nodes = ("simple_loop", "retake", "human")
    else:
        route_nodes = ("simple_loop", "full_loop", "retake", "human")
    for node in route_nodes:
        graph.add_edge(node, "aggregate")
    graph.add_edge("aggregate", "review")
    graph.add_conditional_edges(
        "review",
        lambda state: _review_next_target(state, human_review=human_review),
        {"simple_upgrade": "simple_upgrade", "human_review": "human_review" if human_review else "finalize", "finalize": "finalize"},
    )
    if use_send:
        graph.add_conditional_edges("simple_upgrade", lambda state: "full_loop_dispatch", {"full_loop_dispatch": "full_loop_dispatch"})
    else:  # pragma: no cover - fallback graph handles this path directly
        graph.add_edge("simple_upgrade", "full_loop")
    if human_review:
        graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()


def run_diagnosis_graph(
    features: InspectionFeatures,
    *,
    context: Optional[Dict[str, Any]] = None,
    router: Optional[JevRouter] = None,
    checkpointer: Any = None,
    human_review: bool = False,
    hitl_service: Any = None,
    resume: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_context = dict(context or {})
    runtime_context.setdefault("run_id", features.run_id)
    runtime_context.setdefault("thread_id", features.thread_id)
    workers = runtime_context.get("workers")
    if workers:
        _register_workers(features.run_id, features.thread_id, workers)
    payload: DiagnosisGraphState = {
        "run_id": features.run_id,
        "thread_id": features.thread_id,
        "features": features.to_dict(),
        # Keep function objects out of graph state/checkpoints. Send workers
        # resolve them from the short-lived registry by worker id.
        "context": _json_context(runtime_context),
        "evidence_bundle": [],
        "trace": [],
    }
    graph = build_diagnosis_graph(router=router, checkpointer=checkpointer, human_review=human_review)
    invoke_config = {"configurable": {"thread_id": features.thread_id}} if checkpointer is not None else None
    if resume is not None:
        if checkpointer is None or Command is None:
            raise RuntimeError("resume requires LangGraph Command and a checkpointer")
        invoke_input: Any = Command(resume=dict(resume))
    else:
        invoke_input = payload
    result = graph.invoke(invoke_input, config=invoke_config) if invoke_config is not None else graph.invoke(invoke_input)
    result["trace"] = result.get("trace", [])
    if hitl_service is not None and resume is None:
        checkpoint = hitl_service.pause(
            features.run_id,
            thread_id=features.thread_id,
            draft=result.get("diagnosis", {}),
            evidence=result.get("evidence_bundle", []),
            issues=result.get("review", {}).get("issues", []),
            review=result.get("review", {}),
            route=result.get("route", "human"),
            model_metadata=result.get("model_metadata", runtime_context.get("model_metadata", {})),
            context=_json_context(runtime_context),
        )
        result["human_review"] = checkpoint
        result["status"] = "waiting_review"
        result.setdefault("result", {})["status"] = "waiting_review"
        result["result"]["human_review_required"] = True
    return result
