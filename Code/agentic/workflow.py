"""Evidence-first local multi-agent workflow for fire-eye snapshots."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict
from uuid import uuid4

import cv2
import requests
from langgraph.graph import END, START, StateGraph

CODE_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = Path(__file__).resolve().parent / "knowledge"
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from web_interface.algo_web_bridge import FireEyeAnalyzer, CLASS_NAMES
from agentic.platform import InspectionRuntime
from agentic.platform.contracts import InspectionTask
from agentic.platform.retrieval import LocalKnowledgeRetriever
from agentic.media.normalize import read_image


class DiagnosisState(TypedDict, total=False):
    input_path: str
    image_b64: str
    visual: Dict[str, Any]
    knowledge: List[Dict[str, str]]
    diagnosis: Dict[str, Any]
    review: Dict[str, Any]
    result: Dict[str, Any]


TraceCallback = Callable[[Dict[str, Any]], None]


def workflow_spec() -> Dict[str, Any]:
    """Return a renderer-friendly static description of the graph."""
    return {
        "nodes": [
            {"id": "perception_agent", "label": "视觉感知 Agent", "kind": "vision"},
            {"id": "knowledge_agent", "label": "工艺知识 Agent", "kind": "retrieval"},
            {"id": "diagnosis_agent", "label": "多模态诊断 Agent", "kind": "reasoning"},
            {"id": "review_agent", "label": "证据复核 Agent", "kind": "verification"},
            {"id": "finalize", "label": "人工复核门控", "kind": "gate"},
        ],
        "edges": [
            {"source": "perception_agent", "target": "knowledge_agent"},
            {"source": "knowledge_agent", "target": "diagnosis_agent"},
            {"source": "diagnosis_agent", "target": "review_agent"},
            {"source": "review_agent", "target": "finalize"},
        ],
    }


def _trace_summary(node: str, update: Dict[str, Any]) -> Dict[str, Any]:
    """Keep trace payload small and free of image bytes/prompts."""
    if node == "perception_agent":
        visual = update.get("visual", {})
        return {
            "has_target": visual.get("has_target"),
            "class_label": visual.get("class_label"),
            "confidence": visual.get("confidence"),
            "inference_ms": visual.get("inference_ms"),
        }
    if node == "knowledge_agent":
        sources = update.get("knowledge", [])
        return {
            "source_count": len(sources),
            "source_ids": [item.get("source_id") for item in sources[:5]],
        }
    if node == "diagnosis_agent":
        diagnosis = update.get("diagnosis", {})
        return {
            "llm_status": diagnosis.get("llm_status", "not_applicable"),
            "risk_interpretation": diagnosis.get("risk_interpretation"),
            "visual_evidence_count": len(diagnosis.get("visual_evidence", [])),
            "knowledge_evidence_count": len(diagnosis.get("knowledge_evidence", [])),
        }
    if node == "review_agent":
        review = update.get("review", {})
        return {
            "approved": review.get("approved"),
            "issue_count": len(review.get("issues", [])),
            "issues": review.get("issues", [])[:3],
        }
    if node == "finalize":
        result = update.get("result", {})
        return {"status": result.get("status"), "human_review_required": result.get("human_review_required")}
    return {}


def _timed_node(node: str, fn: Callable[[DiagnosisState], DiagnosisState], callback: Optional[TraceCallback]):
    """Wrap a graph node for UI tracing without changing its returned state."""
    def invoke(state: DiagnosisState) -> DiagnosisState:
        started = time.perf_counter()
        started_at = time.time()
        if callback:
            callback({"node": node, "status": "running", "started_at": started_at})
        try:
            update = fn(state)
        except Exception as exc:
            if callback:
                callback({
                    "node": node,
                    "status": "error",
                    "started_at": started_at,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": str(exc),
                })
            raise
        if callback:
            callback({
                "node": node,
                "status": "completed",
                "started_at": started_at,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "summary": _trace_summary(node, update),
            })
        return update
    return invoke


def _read_representative_frame(input_path: str):
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = read_image(path, cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None

    if frame is None or frame.size == 0:
        raise ValueError(f"无法从输入中解码图像: {path}")
    return frame


def _image_base64(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("图像 JPEG 编码失败")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _perception_agent(state: DiagnosisState, vision_device: str) -> DiagnosisState:
    frame = _read_representative_frame(state["input_path"])
    analyzer = FireEyeAnalyzer(device=vision_device)
    output = analyzer.analyze_frame(frame, immediate=True)
    roi = output.get("roi_bgr")
    model_image = roi if roi is not None and roi.size else frame

    probabilities = output.get("cls_probs") or []
    visual = {
        "has_target": bool(output.get("has_target")),
        "class_code": output.get("cls_result", "NoTarget"),
        "class_label": output.get("cls_result_cn", "未识别"),
        "confidence": float(output.get("cls_conf", 0.0)),
        "class_probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
            if index < len(probabilities)
        },
        "boxes": output.get("boxes", {}),
        "static_features": {
            "gray_mean": output.get("gray_mean"),
            "contrast": output.get("contrast"),
            "saturation": output.get("saturation"),
        },
        "roi_size": output.get("roi_size"),
        "inference_ms": output.get("infer_ms"),
    }
    return {"visual": visual, "image_b64": _image_base64(model_image)}


def _knowledge_agent(visual: Dict[str, Any], knowledge_dir: Path) -> List[Dict[str, str]]:
    return LocalKnowledgeRetriever(Path(knowledge_dir)).search(visual, top_k=5)


def _ollama_json(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    request_timeout = float(timeout or os.getenv("MARS_AGENT_TIMEOUT", "90"))
    fast_mode = os.getenv("MARS_AGENT_FAST", "1").strip().lower() not in {"0", "false", "no"}
    session = requests.Session()
    # Ollama is local; ignore inherited HTTP(S)_PROXY values, which may contain
    # malformed enterprise proxy settings and should never receive image data.
    session.trust_env = False
    response = session.post(
        base_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你是严谨的铝电解火眼视觉分析助手。只按证据作答。"},
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096 if fast_mode else 8192,
                "num_predict": 768 if fast_mode else 2048,
            },
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    message = response.json().get("message", {})
    content = message.get("content", "")
    if not str(content).strip():
        raise ValueError("本地模型未生成可见 JSON")
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        # Some local VLM builds wrap otherwise valid JSON in a short preamble.
        # Decode the first complete object without accepting arbitrary prose.
        text_content = str(content or "").strip()
        start = text_content.find("{")
        if start < 0:
            raise ValueError(f"本地模型没有返回 JSON 对象（返回内容: {text_content[:120]!r}）")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text_content[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("本地模型返回的 JSON 不完整") from exc
    # Ollama's JSON mode may serialize the assistant JSON one extra time.
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError("本地模型返回的 JSON 字符串无法解码") from exc
    if not isinstance(parsed, dict):
        raise ValueError("本地模型返回的 JSON 顶层必须是对象")
    return parsed


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _diagnosis_agent(state: DiagnosisState, base_url: str, model: str) -> DiagnosisState:
    visual = state["visual"]
    if not visual["has_target"]:
        return {"diagnosis": {
            "visual_summary": "现有检测器未找到可分析的火眼目标。",
            "visual_evidence": [],
            "knowledge_evidence": [],
            "risk_interpretation": "unknown",
            "uncertainties": ["当前帧没有有效火眼 ROI。"],
            "follow_up_checks": ["更换清晰且包含火眼区域的图像后重试。"],
        }}

    # Keep the multimodal prompt compact; the full retrieved passages remain
    # in the audit result, while the VLM receives the top three snippets.
    sources = state.get("knowledge", [])[:3]
    prompt = (
        "Return exactly one JSON object. You are a cautious aluminum-electrolysis fire-eye inspection assistant. "
        "Describe only visible evidence in this image and the supplied detector output. "
        "C/N/H is a visual proxy label, never a temperature measurement. Do not output Celsius superheat, dosage, voltage, or control commands. "
        "Use only the supplied source_id values for knowledge_evidence; if unsupported, return an empty array. "
        "Use Chinese for the text values. Keys: visual_summary (string), visual_evidence (array of strings), "
        "knowledge_evidence (array of {source_id, claim}), risk_interpretation (low|medium|high|unknown), "
        "uncertainties (array), follow_up_checks (array).\n"
        f"Detector output: {json.dumps(visual, ensure_ascii=False)}\n"
        f"Knowledge passages: {json.dumps(sources, ensure_ascii=False)}"
    )
    llm_error = None
    try:
        raw = _ollama_json(base_url, model, prompt, state["image_b64"])
    except Exception as exc:
        # Keep the graph auditable when a local VLM exhausts its thinking budget
        # or violates the output contract. The fallback never claims a process
        # measurement and is always sent to human review.
        llm_error = str(exc)
        code = visual.get("class_code", "NoTarget")
        raw = {
            "visual_summary": f"传统视觉模型识别为 {visual.get('class_label', code)}，当前仅形成视觉状态草稿。",
            "visual_evidence": [
                f"检测到火眼目标: {visual.get('has_target')}",
                f"分类标签: {visual.get('class_label', code)}，置信度 {visual.get('confidence', 0.0):.3f}",
            ],
            "knowledge_evidence": [],
            "risk_interpretation": "unknown",
            "uncertainties": ["本地视觉语言模型没有返回可校验的 JSON。", "C/N/H 不是实测温度。"],
            "follow_up_checks": ["请人工复核火眼 ROI，并补充温度与电解质成分测量。"],
        }
    evidence = raw.get("knowledge_evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    normalized_evidence = []
    for item in evidence:
        if isinstance(item, dict):
            normalized_evidence.append({
                "source_id": str(item.get("source_id", "")),
                "claim": str(item.get("claim", "")),
            })
    risk = str(raw.get("risk_interpretation", "unknown")).lower()
    if risk not in {"low", "medium", "high", "unknown"}:
        risk = "unknown"
    diagnosis = {
        "visual_summary": str(raw.get("visual_summary", "")),
        "visual_evidence": _string_list(raw.get("visual_evidence")),
        "knowledge_evidence": normalized_evidence,
        "risk_interpretation": risk,
        "uncertainties": _string_list(raw.get("uncertainties")),
        "follow_up_checks": _string_list(raw.get("follow_up_checks")),
    }
    if llm_error:
        diagnosis["llm_status"] = "fallback_visual_draft"
        diagnosis["llm_error"] = llm_error
    else:
        diagnosis["llm_status"] = "qwen_vl_json"
    return {"diagnosis": diagnosis}


def _review_agent(
    state: DiagnosisState,
    base_url: str,
    model: str,
    llm_review: bool = False,
) -> DiagnosisState:
    if not state["visual"]["has_target"]:
        return {"review": {"approved": False, "issues": ["没有可复核的火眼目标。"]}}

    sources = state.get("knowledge", [])
    prompt = (
        "Return exactly one JSON object with keys approved (boolean) and issues (array of strings). "
        "Audit whether the diagnosis is supported by the image and supplied sources. Reject if it treats C/N/H as measured temperature, "
        "invents a source or root cause, or gives an actual process-control command. "
        "A statement of uncertainty and a suggestion to perform a human check are compliant and should not be rejected. Use Chinese for issues.\n"
        f"Detector output: {json.dumps(state['visual'], ensure_ascii=False)}\n"
        f"Sources: {json.dumps(sources, ensure_ascii=False)}\n"
        f"Diagnosis: {json.dumps(state['diagnosis'], ensure_ascii=False)}"
    )
    if llm_review:
        try:
            raw = _ollama_json(base_url, model, prompt, state["image_b64"])
            approved = raw.get("approved") is True
            issues = _string_list(raw.get("issues"))
        except Exception as exc:
            approved = False
            issues = [f"审核 Agent 调用失败: {exc}"]
    else:
        # The default reviewer is deterministic. This keeps the visible
        # multi-agent workflow responsive by avoiding a second VLM generation.
        # Set MARS_AGENT_LLM_REVIEW=true for an experimental LLM reviewer.
        diagnosis = state["diagnosis"]
        approved = True
        issues = []
        if diagnosis.get("llm_status") == "fallback_visual_draft":
            approved = False
            issues.append("诊断 Agent 使用了降级草稿，不能自动通过。")
        if not diagnosis.get("visual_evidence"):
            approved = False
            issues.append("诊断没有提供可复核的视觉证据。")

    known_ids = {item["source_id"] for item in sources}
    cited_ids = {
        item.get("source_id", "")
        for item in state["diagnosis"].get("knowledge_evidence", [])
    }
    invalid_ids = sorted(cited_ids - known_ids - {""})
    if invalid_ids:
        approved = False
        issues.append("诊断引用了未检索到的知识来源: " + ", ".join(invalid_ids))

    serialized = json.dumps(state["diagnosis"], ensure_ascii=False)
    if re.search(r"\d+(?:\.\d+)?\s*(?:°C|℃|摄氏度)|ΔT\s*[=:]", serialized, re.IGNORECASE):
        approved = False
        issues.append("诊断包含未由输入测量支持的数值过热度。")
    return {"review": {"approved": approved, "issues": issues}}


def _finalize(state: DiagnosisState) -> DiagnosisState:
    visual = state["visual"]
    review = state["review"]
    if not visual["has_target"]:
        status = "insufficient_visual_evidence"
    elif review["approved"]:
        status = "evidence_review_passed"
    else:
        status = "needs_human_review"
    return {"result": {
        "flow_version": "legacy-single-frame",
        "status": status,
        "input": Path(state["input_path"]).name,
        "visual": visual,
        "knowledge_sources": state.get("knowledge", []),
        "diagnosis": state["diagnosis"],
        "review": review,
        "human_review_required": True,
        "input_mode": "single representative frame; no temporal features passed to agents",
        "claims": {
            "legacy_compatibility_flow": True,
            "modern_multiframe_flow": False,
            "process_state_used": False,
            "jev_route_used": False,
            "native_hitl_used": False,
            "rsi_runtime_used": False,
            "flow_validation_only": True,
        },
    }}


def build_workflow(
    model: str = "qwen3-vl:latest",
    base_url: str = "http://localhost:11434",
    vision_device: str = "cpu",
    knowledge_dir: Path = KNOWLEDGE_ROOT,
    trace_callback: Optional[TraceCallback] = None,
    llm_review: Optional[bool] = None,
):
    """Build the deterministic graph: perception -> retrieval -> diagnosis -> review."""
    graph = StateGraph(DiagnosisState)
    graph.add_node("perception_agent", _timed_node(
        "perception_agent", lambda state: _perception_agent(state, vision_device), trace_callback
    ))
    graph.add_node("knowledge_agent", _timed_node(
        "knowledge_agent",
        lambda state: {"knowledge": _knowledge_agent(state["visual"], Path(knowledge_dir))},
        trace_callback,
    ))
    graph.add_node("diagnosis_agent", _timed_node(
        "diagnosis_agent", lambda state: _diagnosis_agent(state, base_url, model), trace_callback
    ))
    if llm_review is None:
        llm_review = os.getenv("MARS_AGENT_LLM_REVIEW", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
    graph.add_node("review_agent", _timed_node(
        "review_agent",
        lambda state: _review_agent(state, base_url, model, llm_review=bool(llm_review)),
        trace_callback,
    ))
    graph.add_node("finalize", _timed_node("finalize", _finalize, trace_callback))
    graph.add_edge(START, "perception_agent")
    graph.add_edge("perception_agent", "knowledge_agent")
    graph.add_edge("knowledge_agent", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "review_agent")
    graph.add_edge("review_agent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_diagnosis(
    input_path: str,
    model: str = "qwen3-vl:latest",
    base_url: str = "http://localhost:11434",
    vision_device: str = "cpu",
    knowledge_dir: Path = KNOWLEDGE_ROOT,
    llm_review: Optional[bool] = None,
    run_id: Optional[str] = None,
    tenant_id: str = "tenant_default",
    session_id: str = "default",
    operator_id: str = "local_operator",
    runtime_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = run_diagnosis_trace(
        input_path=input_path,
        model=model,
        base_url=base_url,
        vision_device=vision_device,
        knowledge_dir=knowledge_dir,
        llm_review=llm_review,
        run_id=run_id,
        tenant_id=tenant_id,
        session_id=session_id,
        operator_id=operator_id,
        runtime_dir=runtime_dir,
    )
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload["result"]


def run_diagnosis_trace(
    input_path: str,
    model: str = "qwen3-vl:latest",
    base_url: str = "http://localhost:11434",
    vision_device: str = "cpu",
    knowledge_dir: Path = KNOWLEDGE_ROOT,
    trace_callback: Optional[TraceCallback] = None,
    llm_review: Optional[bool] = None,
    run_id: Optional[str] = None,
    tenant_id: str = "tenant_default",
    session_id: str = "default",
    operator_id: str = "local_operator",
    runtime_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the workflow and return the final result plus UI-safe node events."""
    events: List[Dict[str, Any]] = []
    task = InspectionTask(
        input_path=str(input_path),
        run_id=run_id or f"run-{uuid4().hex[:12]}",
        tenant_id=tenant_id,
        session_id=session_id,
        operator_id=operator_id,
    )
    runtime = InspectionRuntime(runtime_dir)
    runtime.start(task)

    def record_event(event: Dict[str, Any]) -> None:
        event = {"run_id": task.run_id, **event}
        events.append(event)
        runtime.record_event(event)
        if trace_callback:
            trace_callback(event)

    try:
        graph = build_workflow(
            model=model,
            base_url=base_url,
            vision_device=vision_device,
            knowledge_dir=knowledge_dir,
            llm_review=llm_review,
            trace_callback=record_event,
        )
        state = graph.invoke({"input_path": str(input_path)})
    except Exception as exc:
        runtime.finish(task, None, error=str(exc))
        return {
            "result": None,
            "trace": events,
            "workflow": workflow_spec(),
            "flow_version": "legacy-single-frame",
            "run_id": task.run_id,
            "error": str(exc),
        }
    result = dict(state["result"])
    result["run_id"] = task.run_id
    result["thread_id"] = task.thread_id
    runtime.finish(task, result)
    return {
        "result": result,
        "trace": events,
        "workflow": workflow_spec(),
        "flow_version": "legacy-single-frame",
        "claims": result.get("claims", {}),
        "run_id": task.run_id,
    }
