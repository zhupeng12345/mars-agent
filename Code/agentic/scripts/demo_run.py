"""Run a local, auditable end-to-end fire-eye diagnosis demonstration.

The default path uses the CPU F-GMVAE-T module with its explicit smoke
weights (fresh weights when no checkpoint is supplied).  It exercises the
interfaces and produces traceable evidence; it must not be read as a trained
visual accuracy result.  Set ``--no-model`` for a model-free quality/routing
smoke or ``--review-action`` to complete the persisted HITL review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from agentic.agents.image_quality import ImageQualityAgent
from agentic.agents.process_state import ProcessStateAgent
from agentic.agents.visual_evidence import derive_expert_features, run_visual_evidence
from agentic.models.fgmvae_temporal import FGMVAETemporal
from agentic.models.visual_types import ROISequence
from agentic.platform.contracts import DiagnosisTicket, InspectionFeatures, InspectionTask, InputMode, jsonable, utc_now
from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.knowledge import VersionedKnowledgeBase
from agentic.platform.hitl import ALLOWED_HUMAN_ACTIONS, AlreadyResumedError, HumanReviewService, InvalidActionError
from agentic.platform.checkpoint import build_sqlite_checkpointer
from agentic.platform.jev.features import build_inspection_features
from agentic.platform.jev.policy import DEFAULT_PROMPT_VARIANT
from agentic.platform.jev.router import JevRouter, create_jev_router
from agentic.rsi.runtime_policy import resolve_runtime_jev_policy
from agentic.platform.llm_gateway import LLMGateway
from agentic.platform.runtime import InspectionRuntime
from agentic.platform.tickets import TicketStore
from agentic.media.normalize import MediaNormalizer, read_image
from agentic.orchestration.diagnosis_graph import build_diagnosis_graph, run_diagnosis_graph


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _find_visual_window(root: Path, input_path: Path) -> Optional[dict[str, Any]]:
    target = input_path.resolve()
    rows = _read_jsonl(root / "data" / "evaluation" / "visual_manifest.jsonl")
    for row in rows:
        value = row.get("video_path")
        if not value:
            continue
        candidate = Path(str(value))
        try:
            if candidate.resolve() == target or candidate.name == input_path.name:
                if row.get("source") == "video" and row.get("roi_paths"):
                    return row
        except OSError:
            if candidate.name == input_path.name:
                return row
    return None


def _sequence_from_manifest(row: Mapping[str, Any], root: Path) -> Optional[ROISequence]:
    if cv2 is None:
        return None
    paths = [Path(str(item)) for item in row.get("roi_paths", [])]
    images = []
    for path in paths:
        if not path.is_absolute():
            candidate = root / path
            # Existing manifests may be authored from the repository root and
            # therefore carry a leading ``Code/`` component.
            path = candidate if candidate.exists() else root.parent / path
        image = read_image(path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0)
    if not images:
        return None
    frames = torch.stack(images).unsqueeze(0)
    boxes = torch.tensor([row.get("boxes", [[0, 0, 1, 1] for _ in images])], dtype=torch.float32)
    mask = torch.tensor([row.get("frame_mask", [True] * len(images))], dtype=torch.bool)
    quality = torch.tensor([row.get("quality_mask", row.get("frame_mask", [True] * len(images)))], dtype=torch.bool)
    frame_ids = torch.tensor([row.get("frame_ids", list(range(len(images))))], dtype=torch.long)
    return ROISequence.from_frame_tensor(frames, boxes=boxes, frame_mask=mask, quality_mask=quality, source_ids=[str(row.get("source_id", "visual-window"))], frame_ids=frame_ids)


def _gateway_images(row: Optional[Mapping[str, Any]], root: Path, limit: int = 1) -> list[str]:
    if not row:
        return []
    resolved: list[str] = []
    values = []
    if row.get("roi_paths"):
        values.append(row["roi_paths"][0])
    elif row.get("frame_paths"):
        values.append(row["frame_paths"][0])
    for value in values[:limit]:
        path = Path(str(value))
        candidates = [path, root / path, root.parent / path]
        match = next((item.resolve() for item in candidates if item.exists()), None)
        if match is not None:
            resolved.append(str(match))
    return resolved


def _sequence_from_image(path: Path, source_id: str) -> Optional[ROISequence]:
    if cv2 is None:
        return None
    image = read_image(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    frames = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).unsqueeze(0) / 255.0
    return ROISequence.from_frame_tensor(frames, boxes=torch.tensor([[[0.0, 0.0, 224.0, 224.0]]]), source_ids=[source_id], frame_ids=torch.tensor([[0]]))


def _fallback_video_sequence(path: Path, source_id: str, frames_count: int = 8) -> Optional[ROISequence]:
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or frames_count)
    indices = sorted(set(int(item) for item in np.linspace(0, max(0, total - 1), frames_count)))
    images = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if ok and image is not None:
            image = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0)
    capture.release()
    if not images:
        return None
    t = len(images)
    return ROISequence.from_frame_tensor(torch.stack(images).unsqueeze(0), boxes=torch.tensor([[[0.0, 0.0, 224.0, 224.0]] * t]), source_ids=[source_id], frame_ids=torch.tensor([indices[:t]]))


def _visual_payload(sequence: Optional[ROISequence], row: Optional[Mapping[str, Any]], *, run_id: str, thread_id: str, no_model: bool, weights: Optional[Path]) -> tuple[dict[str, Any], Optional[str]]:
    source_id = str(row.get("source_id", "visual-window")) if row else (sequence.source_ids[0] if sequence and sequence.source_ids else "visual")
    frame_ids = (row.get("frame_ids") if row else None) or (sequence.frame_ids[0].tolist() if sequence and sequence.frame_ids is not None else [0])
    if sequence is None:
        return {"has_target": False, "confidence": 0.0, "class_probabilities": {}, "risk_level": "unknown", "model_profile": "visual_unavailable", "frame_ids": frame_ids}, None
    if no_model:
        return {"has_target": bool(sequence.effective_mask.any()), "confidence": 0.75 if bool(sequence.effective_mask.any()) else 0.0, "class_probabilities": {}, "risk_level": "unknown", "temporal_consistency": float(sequence.effective_mask.float().mean()), "frame_ids": frame_ids, "model_profile": "model_free_visual_smoke"}, None
    model_config: dict[str, Any] = {"model_profile": "fgmvae-t-smoke-untrained-v1"}
    checkpoint_scope = "untrained_smoke"
    checkpoint_payload: Any = None
    if weights and weights.exists():
        try:
            checkpoint_payload = torch.load(weights, map_location="cpu")
            if isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model_config"), Mapping):
                model_config.update(dict(checkpoint_payload["model_config"]))
            checkpoint_scope = str(checkpoint_payload.get("scope", "image_label_supervised") if isinstance(checkpoint_payload, Mapping) else "checkpoint")
        except Exception:
            checkpoint_payload = None
    model = FGMVAETemporal(**model_config)
    weight_status = "fresh_smoke_weights"
    if checkpoint_payload is not None:
        try:
            model.load_state_dict(checkpoint_payload.get("state_dict", checkpoint_payload), strict=False)
            weight_status = f"checkpoint:{weights.name}"
        except Exception:
            weight_status = "checkpoint_load_failed_fresh_weights"
    framewise_pooling = "image_label_supervised" in checkpoint_scope
    static_features, dynamic_features = derive_expert_features(sequence)
    evidence = run_visual_evidence(
        model, sequence, source_id=source_id, frame_ids=frame_ids,
        run_id=run_id, thread_id=thread_id, static_features=static_features,
        dynamic_features=dynamic_features, framewise_pooling=framewise_pooling,
    )
    output = evidence.to_dict()
    output.update({
        "has_target": bool(sequence.effective_mask.any()),
        "confidence": float(output.get("score", 0.0) or 0.0),
        "class_probabilities": output.get("class_probs", {}),
        "weights_status": weight_status,
        "training_scope": checkpoint_scope,
        "aggregation": "mean_frame_probabilities" if framewise_pooling else "temporal_model",
        # No closed-set OOD calibration is available yet; uncertainty is an
        # observable novelty proxy and is explicitly labelled as such.
        "novelty_score": float(min(1.0, output.get("uncertainty", 1.0) or 1.0)),
        "novelty_method": "uncertainty_proxy_uncalibrated",
        "flow_validation_only": True,
    })
    return output, weight_status


def _knowledge_query(
    quality: Mapping[str, Any],
    visual: Mapping[str, Any],
    process: Any,
    extra_terms: tuple[str, ...] | list[str] = (),
) -> str:
    """Build a stable retrieval query from observable demo signals."""
    terms = ["visual", "fire_eye", "process", "superheat", "report", "review"]
    terms.extend(str(item) for item in extra_terms if isinstance(item, str) and item.strip())
    terms.append(str(quality.get("quality_status", "unknown")))
    terms.append(str(visual.get("class_code", visual.get("class_label", "unknown"))))
    if process is not None:
        snapshot = process.to_dict() if hasattr(process, "to_dict") else dict(process or {})
        terms.append(str(snapshot.get("process_regime", "unknown")))
    return " ".join(item for item in terms if item and item != "unknown")


def _case_features(quality: Mapping[str, Any], visual: Mapping[str, Any]) -> dict[str, float]:
    """Return numeric, observable features shared by case retrieval and audit."""
    return {
        "confidence": float(visual.get("confidence", visual.get("score", 0.0)) or 0.0),
        "target_present": 1.0 if bool(visual.get("has_target", quality.get("target_present", False))) else 0.0,
        "quality_score": float(quality.get("quality_score", quality.get("brightness_score", 0.0)) or 0.0),
        "temporal_consistency": float(visual.get("temporal_consistency", quality.get("cross_frame_consistency", 0.0)) or 0.0),
    }


def _gateway_metadata(response: Any) -> dict[str, Any]:
    """Keep gateway audit metadata small enough for graph and HITL state."""
    return {
        "status": response.status,
        "profile": response.profile,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "latency_ms": response.latency_ms,
        "retries": response.retries,
        "error_type": response.error_type,
        "error": response.error,
        "capabilities": list(response.capabilities),
    }


def run_demo(
    root: Path,
    input_path: Path,
    *,
    output: Path,
    no_model: bool = False,
    weights: Optional[Path] = None,
    review_action: Optional[str] = None,
    assets_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    retake_of_run_id: Optional[str] = None,
    knowledge_path: Optional[Path] = None,
    case_memory_root: Optional[Path] = None,
    llm_profile: Optional[str] = None,
    llm_gateway: Optional[LLMGateway] = None,
    runtime_dir: Optional[Path] = None,
    native_hitl: bool = False,
    jev_backend: Optional[str] = None,
    laya_model: Optional[str] = None,
    jev_router: Optional[JevRouter] = None,
    jev_policy_registry_root: Optional[Path] = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    input_path = Path(input_path).resolve()
    run_id = run_id or f"demo-{uuid.uuid4().hex[:12]}"
    if parent_run_id and retake_of_run_id and parent_run_id != retake_of_run_id:
        raise ValueError("parent_run_id and retake_of_run_id must identify the same parent run")
    parent_run_id = parent_run_id or retake_of_run_id
    retake_of_run_id = retake_of_run_id or parent_run_id
    if parent_run_id and not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", parent_run_id):
        raise ValueError("parent_run_id contains unsupported characters")
    llm_profile = llm_profile or os.environ.get("MARS_AGENT_LLM_PROFILE", "fake_json")
    jev_backend = jev_backend or os.environ.get("MARS_AGENT_JEV_BACKEND", "deterministic_rule")
    laya_model = laya_model or os.environ.get("MARS_AGENT_LAYA_MODEL", "typed-decisions")
    thread_id = f"inspection_demo_default_{run_id}"
    manifest = MediaNormalizer().normalize(input_path, run_id=run_id, thread_id=thread_id)
    runtime_root = Path(runtime_dir or os.environ.get("MARS_AGENT_RUNTIME_DIR", str(root / "runtime_data"))).expanduser()
    if weights is None:
        default_checkpoint = runtime_root / "visual_training" / "fgmvae_t_image_supervised.pth"
        if default_checkpoint.exists():
            weights = default_checkpoint
    runtime = InspectionRuntime(runtime_root)
    input_mode = InputMode.VIDEO_WINDOW.value if manifest.get("media_type") == "video" else InputMode.IMAGE.value
    task = InspectionTask(
        input_path=str(input_path),
        run_id=run_id,
        thread_id=thread_id,
        input_mode=input_mode,
        metadata={
            "entrypoint": "demo_run",
            "flow_validation_only": True,
            "parent_run_id": parent_run_id,
            "retake_of_run_id": retake_of_run_id,
        },
    )
    runtime.start(task)
    quality = ImageQualityAgent().run(input_path, manifest=manifest, run_id=run_id, thread_id=thread_id)
    row = _find_visual_window(root, input_path)
    if row:
        sequence = _sequence_from_manifest(row, root)
    elif manifest.get("media_type") == "image":
        sequence = _sequence_from_image(input_path, manifest.get("source_id", "image"))
    else:
        sequence = _fallback_video_sequence(input_path, manifest.get("source_id", "video"))
    # A failed quality gate is a terminal low-cost route for this run.  Do not
    # spend model time, or create a misleading visual classification, before
    # the operator supplies a usable image/video window.
    if quality.get("quality_status") in {"retake", "human"}:
        visual = {
            "has_target": bool(quality.get("target_present", False)),
            "confidence": 0.0,
            "class_probabilities": {},
            "risk_level": "unknown",
            "temporal_consistency": quality.get("cross_frame_consistency"),
            "frame_ids": [],
            "model_profile": "quality_gate_skipped",
            "weights_status": "not_called",
            "flow_validation_only": True,
        }
        weight_status = "quality_gate_skipped"
    else:
        visual, weight_status = _visual_payload(sequence, row, run_id=run_id, thread_id=thread_id, no_model=no_model, weights=weights)
    if row:
        # Keep the sequence provenance in the run JSON so the UI and PPT
        # asset builder can replay the exact ROI/key-frame evidence.
        for key in ("contact_sheet", "frame_paths", "roi_paths", "boxes", "frame_mask", "missing_count", "track_quality", "key_frame_ids"):
            if key in row:
                visual[key] = row[key]
    process_path = root / "data" / "v1" / "step2_ew_202111" / "dataset.npz"
    process = ProcessStateAgent().run(process_path, sample_index=0, run_id=run_id, thread_id=thread_id) if process_path.exists() else ProcessStateAgent().run(None, run_id=run_id, thread_id=thread_id)
    knowledge_path = Path(knowledge_path) if knowledge_path is not None else root / "agentic" / "knowledge" / "rules_v1.jsonl"
    configured_registry = jev_policy_registry_root or os.environ.get("MARS_AGENT_RSI_REGISTRY")
    rsi_runtime_policy = resolve_runtime_jev_policy(configured_registry) if jev_router is None else None
    rsi_query_terms = rsi_runtime_policy.knowledge_query_terms if rsi_runtime_policy else ()
    prompt_variant = rsi_runtime_policy.prompt_variant if rsi_runtime_policy else DEFAULT_PROMPT_VARIANT
    knowledge_base = VersionedKnowledgeBase(knowledge_path)
    knowledge = knowledge_base.search_bm25(_knowledge_query(quality, visual, process, rsi_query_terms), top_k=5)
    case_root = Path(case_memory_root) if case_memory_root is not None else runtime_root
    case_memory = SpecialCaseMemory(case_root)
    features_for_case = _case_features(quality, visual)
    cases = case_memory.search(features_for_case, applicability=("visual", "fire_eye", "process", "report"), top_k=5)
    # Build the low-cost feature contract and make the JEV decision before a
    # provider call. The graph recomputes the same deterministic decision as
    # its first node, while this pre-route keeps expensive diagnosis work
    # downstream of the routing gate.
    features = build_inspection_features(
        run_id=run_id,
        thread_id=thread_id,
        quality=quality,
        visual=visual,
        knowledge=knowledge,
        cases=cases,
        process_snapshot=process,
    )
    selected_router = jev_router or create_jev_router(
        jev_backend,
        laya_model=laya_model,
        policy=rsi_runtime_policy.policy,
        policy_version=rsi_runtime_policy.version_id,
        policy_hash=rsi_runtime_policy.policy_hash,
    )
    pre_route = selected_router.route(features)
    ticket_store = TicketStore(runtime_root)
    ticket = ticket_store.create(task, title="铝电解火眼多智能体诊断", route=pre_route.route, model_profile=llm_profile)

    # The graph calls the provider only for full_loop, after specialist fan-in.
    gateway_output: dict[str, Any] = {}
    deferred = pre_route.route == "full_loop" and quality.get("quality_status") not in {"retake", "human"}
    gateway_meta = {
        "status": "pending_aggregator" if deferred else "skipped_by_jev",
        "profile": llm_profile,
        "provider": "none",
        "model": "deferred" if deferred else "route-policy",
        "request_id": f"deferred-{uuid.uuid4().hex[:12]}" if deferred else f"skip-{uuid.uuid4().hex[:12]}",
        "latency_ms": 0.0,
        "retries": 0,
        "error_type": None if deferred else "not_called_by_route",
        "error": None if deferred else "no_provider_call_for_selected_route",
        "capabilities": [],
    }
    candidate = None
    knowledge_versions = sorted({str(item.get("source_version", "")) for item in knowledge if item.get("source_version")})
    service = HumanReviewService(runtime_root)
    graph_context = {
            "quality": quality,
            "visual": visual,
            "knowledge": knowledge,
            "cases": cases,
            "case_candidate": candidate,
            "llm_profile": llm_profile,
            "gateway_images": _gateway_images(row, root),
            "media_source_id": str(manifest.get("source_id", "")),
            "pre_route_decision": pre_route.to_dict(),
            "pre_route_features": features.to_dict(),
            "ticket": ticket.to_dict(),
            "ticket_store_root": str(runtime_root),
            "process_state": process.to_dict(),
            "rsi_knowledge_query_terms": list(rsi_query_terms),
            "prompt_variant": prompt_variant,
            "knowledge_path": str(knowledge_path),
            "case_memory_root": str(case_root),
            "case_features": features_for_case,
            "model_metadata": {
                "model_profile": visual.get("model_profile", "model_free"),
                "prompt_version": prompt_variant,
                "rsi_policy_version": rsi_runtime_policy.version_id if rsi_runtime_policy else getattr(jev_router, "policy_version", "base"),
                "rsi_policy_hash": rsi_runtime_policy.policy_hash if rsi_runtime_policy else getattr(jev_router, "policy_hash", "builtin"),
                "llm_gateway": gateway_meta,
                "knowledge_versions": knowledge_versions,
                "knowledge_version": knowledge_versions[-1] if knowledge_versions else "unknown",
                "parent_run_id": parent_run_id,
                "retake_of_run_id": retake_of_run_id,
            },
            "parent_run_id": parent_run_id,
            "retake_of_run_id": retake_of_run_id,
        }
    checkpointer = None
    if native_hitl:
        checkpointer = build_sqlite_checkpointer(runtime_root / "graph_checkpoints.sqlite")
        if checkpointer is None:
            raise RuntimeError("native Streamlit review requires langgraph-checkpoint-sqlite")
    try:
        graph = run_diagnosis_graph(
            features,
            context=graph_context,
            router=selected_router,
            checkpointer=checkpointer,
            human_review=native_hitl,
            hitl_service=service,
        )
    finally:
        if checkpointer is not None:
            checkpointer.conn.close()
    gateway_evidence = next((item for item in graph.get("evidence_bundle", []) if item.get("agent_id") == "llm_gateway"), {})
    gateway_output = gateway_evidence.get("output", {}).get("content", {}) if isinstance(gateway_evidence, Mapping) else {}
    gateway_meta = gateway_evidence.get("output", {}).get("metadata", gateway_meta) if isinstance(gateway_evidence, Mapping) else gateway_meta
    graph_ticket = graph.get("ticket") or graph.get("result", {}).get("ticket")
    if isinstance(graph_ticket, Mapping):
        try:
            ticket = ticket_store.update(
                DiagnosisTicket.from_dict(graph_ticket),
                stage=str(graph_ticket.get("stage", "completed")),
                status=str(graph_ticket.get("status", "waiting_review")),
                health=str(graph_ticket.get("health", "healthy")),
                worker_id=str(graph_ticket.get("worker_id", "orchestrator")),
                retry_count=int(graph_ticket.get("retry_count", 0) or 0),
                report_id=str(graph_ticket.get("report_id", "")),
                human_decision=graph_ticket.get("human_decision"),
                payload={"route": graph_ticket.get("route"), "report_id": graph_ticket.get("report_id", "")},
            )
        except (TypeError, ValueError):
            # Older/custom graph adapters may return a ticket without the new
            # fields; preserve their result while keeping the run auditable.
            pass
    if candidate is None and pre_route.route in {"simple_loop", "full_loop"} and quality.get("quality_status") == "pass":
        candidate = case_memory.add_candidate(
            source_id=str(manifest.get("source_id", input_path.name)),
            features=features_for_case,
            conclusion={"visual": visual, "diagnosis": graph.get("diagnosis", {}), "route": pre_route.to_dict()},
            run_id=run_id,
            applicable_when=("visual", "fire_eye"),
        )
    review = None
    if review_action:
        review = service.resume(run_id, action=review_action, reviewer_id="demo_operator", reason="P7 demo review")
        # A candidate is provisional until the operator has decided. Keep its
        # conclusion aligned with the durable final diagnosis, and retire it
        # when the task is rejected or sent back for a retake.
        if candidate is not None:
            reviewed_result = review.get("result") if isinstance(review, Mapping) else None
            final_diagnosis = (reviewed_result or {}).get("diagnosis") if isinstance(reviewed_result, Mapping) else None
            if review_action in {"approve", "amend"} and isinstance(final_diagnosis, Mapping):
                candidate = case_memory.update_candidate_conclusion(
                    str(candidate["case_id"]),
                    {"visual": visual, "diagnosis": dict(final_diagnosis), "route": pre_route.to_dict()},
                    reviewer_id="demo_operator",
                    reason="human review finalized candidate conclusion",
                )
            elif review_action in {"reject", "request_retake"} and candidate.get("state") == "candidate":
                candidate = case_memory.transition(
                    str(candidate["case_id"]),
                    "deprecated",
                    reviewer_id="demo_operator",
                    reason=f"human review action: {review_action}",
                )
    result = {
        "schema_version": "demo-run-1", "run_id": run_id, "thread_id": thread_id,
        "parent_run_id": parent_run_id, "retake_of_run_id": retake_of_run_id,
        "input": manifest, "quality": quality, "visual": visual, "process_state": process.to_dict(),
        "features": features.to_dict(), "graph": graph, "human_review": review or service.pending(run_id),
        "knowledge": knowledge, "cases": cases, "case_candidate": candidate,
        "case_memory_root": str(case_root),
        "ticket": ticket.to_dict(),
        "pre_route_decision": pre_route.to_dict(),
        "jev_router": {
            "configured_backend": jev_backend,
            "decision_backend": pre_route.router_backend,
            "model_profile": pre_route.model_profile,
            "gate_version": pre_route.gate_version,
            "policy_version": pre_route.policy_version,
            "policy_hash": pre_route.policy_hash,
            "minimum_laya_confidence": selected_router.policy.minimum_laya_confidence,
        },
        "rsi_runtime": rsi_runtime_policy.to_dict() if rsi_runtime_policy else {"status": "custom_router", "version_id": pre_route.policy_version, "policy_hash": pre_route.policy_hash},
        "knowledge_versions": knowledge_versions,
        "knowledge_version": knowledge_versions[-1] if knowledge_versions else "unknown",
        "prompt_version": prompt_variant,
        "llm_gateway": {**gateway_meta, "content": gateway_output},
        "human_review_backend": "langgraph_sqlite_interrupt" if native_hitl else "service_checkpoint",
        "claims": {"flow_validation_only": True, "expert_video_labels_available": False, "real_superheat_measurement": False, "process_video_alignment": "unknown"},
    }
    # Store observable node events and the final graph result for the system
    # health evaluator.  Trace summaries contain no private model reasoning.
    for event in graph.get("trace", []):
        runtime.record_event(event)
    runtime_result = dict(graph.get("result", graph)) if isinstance(graph.get("result", graph), Mapping) else {"result": graph.get("result", graph)}
    # Persist runtime policy provenance alongside the graph result so scoped
    # evaluation can audit which RSI surfaces were actually consumed.
    runtime_result.update({
        "rsi_runtime": result["rsi_runtime"],
        "prompt_version": result["prompt_version"],
        "knowledge_query_terms": result["rsi_runtime"].get("knowledge_query_terms", []),
        "parent_run_id": parent_run_id,
        "retake_of_run_id": retake_of_run_id,
    })
    runtime.finish(task, runtime_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if native_hitl:
        _write_demo_result(runtime_root, result)
    if assets_dir and row:
        assets_dir.mkdir(parents=True, exist_ok=True)
        contact = row.get("contact_sheet")
        if contact:
            contact_path = Path(str(contact))
            if not contact_path.is_absolute():
                candidate = root / contact_path
                contact_path = candidate if candidate.exists() else root.parent / contact_path
            if contact_path.exists():
                asset_id = f"asset-{run_id}-contact-sheet"
                target = assets_dir / f"{asset_id}.jpg"
                shutil.copy2(contact_path, target)
                metadata = {"asset_id": asset_id, "title": "真实视频多帧 ROI 证据", "source_files": [str(input_path)], "run_id": run_id, "model_profile": visual.get("model_profile", ""), "knowledge_version": result.get("knowledge_version", "unknown"), "script_version": "demo-run-1", "generated_at": result["input"].get("created_at", ""), "width": None, "height": None, "status": "flow_validation"}
                metadata["script"] = "agentic.scripts.demo_run"
                metadata["width"] = result["input"].get("width")
                metadata["height"] = result["input"].get("height")
                (assets_dir / f"{asset_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _demo_result_path(runtime_dir: Path | str, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", str(run_id)):
        raise ValueError("run_id contains unsupported characters")
    return Path(runtime_dir) / "modern_demos" / f"{run_id}.json"


def _write_demo_result(runtime_dir: Path | str, result: Mapping[str, Any]) -> Path:
    path = _demo_result_path(runtime_dir, str(result.get("run_id", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_demo_run(runtime_dir: Path | str, run_id: str) -> dict[str, Any]:
    """Reload one persisted modern demo for review after a UI rerun/restart."""
    path = _demo_result_path(runtime_dir, run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise ValueError("persisted demo run_id does not match the requested run")
    return payload


def resume_demo_review(
    runtime_dir: Path | str,
    run_id: str,
    *,
    action: str,
    reviewer_id: str = "local_operator",
    reason: str = "",
    amendments: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Resume the native graph, then commit its idempotent service audit row."""
    if action not in ALLOWED_HUMAN_ACTIONS:
        raise InvalidActionError(f"invalid human review action: {action!r}")
    runtime_root = Path(runtime_dir)
    demo = load_demo_run(runtime_root, run_id)
    if demo.get("human_review_backend") != "langgraph_sqlite_interrupt":
        raise ValueError("run is not backed by a native LangGraph interrupt")
    thread_id = str(demo.get("thread_id", ""))
    features = InspectionFeatures.from_dict(demo.get("features") or {})
    if features.run_id != run_id or features.thread_id != thread_id:
        raise ValueError("persisted feature contract does not match this run/thread")
    service = HumanReviewService(runtime_root)
    pending = service.get(run_id, thread_id=thread_id)
    checkpointer = build_sqlite_checkpointer(runtime_root / "graph_checkpoints.sqlite")
    if checkpointer is None:
        raise RuntimeError("native Streamlit review requires langgraph-checkpoint-sqlite")
    decision = {
        "action": action,
        "reviewer_id": reviewer_id,
        "reason": reason,
        "amendments": dict(amendments or {}),
    }
    try:
        compiled = build_diagnosis_graph(checkpointer=checkpointer, human_review=True)
        snapshot = compiled.get_state({"configurable": {"thread_id": thread_id}})
        saved_values = dict(getattr(snapshot, "values", {}) or {})
        next_nodes = tuple(getattr(snapshot, "next", ()) or ())
        saved_decision = saved_values.get("human_decision") or {}
        if "human_review" in next_nodes:
            graph = run_diagnosis_graph(
                features,
                checkpointer=checkpointer,
                human_review=True,
                resume=decision,
            )
        elif saved_decision:
            if saved_decision.get("action") != action:
                raise AlreadyResumedError(
                    f"LangGraph run {run_id} already resumed with action {saved_decision.get('action')}"
                )
            graph = saved_values
        else:
            raise RuntimeError("LangGraph checkpoint is not waiting at human_review")
    finally:
        checkpointer.conn.close()

    runtime = InspectionRuntime(runtime_root)
    outcome = runtime.resume_review(
        run_id,
        {"action": action, "reviewer_id": reviewer_id, "reason": reason, "amendments": dict(amendments or {})},
        reviewer_id=reviewer_id,
        reason=reason,
        amendments=amendments,
        thread_id=thread_id,
    )
    demo["graph"] = graph
    demo["graph"]["status"] = (graph.get("result") or {}).get("status", outcome.get("result", {}).get("status"))
    demo["graph"].setdefault("result", {})["review"] = outcome
    demo["human_review"] = outcome
    candidate = demo.get("case_candidate")
    case_root = Path(str(demo.get("case_memory_root") or runtime_root))
    if candidate and isinstance(outcome.get("result"), Mapping):
        case_memory = SpecialCaseMemory(case_root)
        final_diagnosis = outcome["result"].get("diagnosis")
        if action in {"approve", "amend"} and isinstance(final_diagnosis, Mapping):
            candidate = case_memory.update_candidate_conclusion(
                str(candidate.get("case_id", "")),
                {
                    "visual": demo.get("visual", {}),
                    "diagnosis": dict(final_diagnosis),
                    "route": demo.get("pre_route_decision", {}),
                },
                reviewer_id=reviewer_id,
                reason="human review finalized candidate conclusion",
            )
        elif action in {"reject", "request_retake"} and candidate.get("state") == "candidate":
            candidate = case_memory.transition(
                str(candidate.get("case_id", "")),
                "deprecated",
                reviewer_id=reviewer_id,
                reason=f"human review action: {action}",
            )
        demo["case_candidate"] = candidate
    ticket_data = graph.get("ticket") or (graph.get("result") or {}).get("ticket")
    if isinstance(ticket_data, Mapping):
        demo["ticket"] = dict(ticket_data)
        current_ticket = TicketStore(runtime_root).by_run(run_id)
        if current_ticket is not None and current_ticket.status != "completed":
            completed_ticket = DiagnosisTicket.from_dict(ticket_data)
            TicketStore(runtime_root).update(
                completed_ticket,
                stage="completed",
                status="completed",
                worker_id="human_reviewer",
                human_decision=graph.get("human_decision") or outcome.get("decision"),
                payload={"review_action": action},
            )
    known_events = {
        (str(item.get("run_id", "")), str(item.get("node", "")), str(item.get("status", "")))
        for item in runtime.events.read_all()
    }
    for event in graph.get("trace", []):
        key = (str(event.get("run_id", "")), str(event.get("node", "")), str(event.get("status", "")))
        if key[0] == run_id and key[1] in {"human_review", "finalize"} and key not in known_events:
            runtime.record_event(event)
            known_events.add(key)
    final_result = dict(graph.get("result") or {})
    final_status = str(final_result.get("status", outcome.get("result", {}).get("status", "completed")))
    existing_run_records = [item for item in runtime.runs.read_all() if item.get("run_id") == run_id]
    if not existing_run_records or existing_run_records[-1].get("status") != final_status:
        runtime.runs.append({
            "run_id": run_id,
            "status": final_status,
            "finished_at": utc_now(),
            "error": None,
            "result": final_result,
        })
    _write_demo_result(runtime_root, demo)
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--llm-profile", choices=("fake_json", "ollama_local", "qwen_api", "fallback_human"), default=None)
    parser.add_argument("--jev-backend", choices=("deterministic_rule", "laya_adapter"), default=None)
    parser.add_argument("--laya-model", choices=("typed-decisions",), default=None)
    parser.add_argument("--jev-policy-registry", type=Path, default=None, help="Explicit approved RSI registry root for bounded JEV/prompt/query runtime surfaces.")
    parser.add_argument("--review-action", choices=("approve", "amend", "reject", "request_retake"), default=None)
    parser.add_argument("--assets-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or (args.root / "runtime_data" / f"demo_{args.input.stem}.json")
    result = run_demo(args.root, args.input, output=output, no_model=args.no_model, weights=args.weights, review_action=args.review_action, assets_dir=args.assets_dir, llm_profile=args.llm_profile, jev_backend=args.jev_backend, laya_model=args.laya_model, jev_policy_registry_root=args.jev_policy_registry)
    print(json.dumps({"output": str(output), "run_id": result["run_id"], "route": result["graph"].get("route"), "status": result["graph"].get("status"), "review": (result.get("human_review") or {}).get("status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
