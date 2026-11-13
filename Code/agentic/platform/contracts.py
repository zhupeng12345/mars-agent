"""Versioned, JSON-safe contracts shared by inspection agents.

The contracts deliberately contain only observable state. In particular,
private model reasoning is rejected from metadata, outputs and trace events.
The module has no third-party dependencies so it can be used by the runtime,
UI and offline evaluators alike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, TypeVar
from uuid import uuid4

T = TypeVar("T", bound="Contract")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContractEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AgentStatus(ContractEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABSTAINED = "abstained"
    HUMAN_REVIEW = "human_review"


class MediaType(ContractEnum):
    IMAGE = "image"
    VIDEO = "video"
    FRAME_DIRECTORY = "frame_directory"
    UNKNOWN = "unknown"


class InputMode(ContractEnum):
    IMAGE = "image"
    VIDEO_WINDOW = "video_window"
    FRAME_DIRECTORY = "frame_directory"
    SINGLE_REPRESENTATIVE_FRAME = "single_representative_frame"


class EvidenceKind(ContractEnum):
    VISUAL_ROI = "visual_roi"
    DETECTOR_OUTPUT = "detector_output"
    KNOWLEDGE = "knowledge"
    PROCESS_STATE = "process_state"
    HUMAN_NOTE = "human_note"
    UNKNOWN = "unknown"


class QualityStatus(ContractEnum):
    PASS = "pass"
    RETAKE = "retake"
    HUMAN = "human"


class Route(ContractEnum):
    SIMPLE_LOOP = "simple_loop"
    FULL_LOOP = "full_loop"
    RETAKE = "retake"
    HUMAN = "human"


class RiskLevel(ContractEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class AlignmentQuality(ContractEnum):
    EXACT = "exact"
    INTERVAL = "interval"
    SEGMENT = "segment"
    WEAK = "weak"
    UNKNOWN = "unknown"


class ReviewAction(ContractEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_RETAKE = "request_retake"
    AMEND = "amend"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    HUMAN_REVIEW = "human_review"
    PASS = "pass"


class TicketStatus(ContractEnum):
    OPEN = "open"
    ASSIGNED = "assigned"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssignmentStatus(ContractEnum):
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_FORBIDDEN_REASONING_KEYS = {
    "chain_of_thought", "cot", "thoughts", "private_reasoning",
    "internal_reasoning", "reasoning_trace", "hidden_reasoning",
    "thought_process", "analysis", "reasoning",
}


def _default_thread(run_id: str) -> str:
    return f"inspection_default_default_{run_id}" if run_id else ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Contract):
        return value.to_dict()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _check_reasoning(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_REASONING_KEYS:
                raise ValueError(f"private reasoning field is not allowed: {path}{key}")
            _check_reasoning(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _check_reasoning(item, f"{path}[{index}].")
    elif isinstance(value, Contract):
        for item in fields(value):
            _check_reasoning(getattr(value, item.name), f"{path}{item.name}.")
    elif is_dataclass(value):
        for item in fields(value):
            _check_reasoning(getattr(value, item.name), f"{path}{item.name}.")


def _required_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _source_id(value: Any, name: str = "source_id") -> None:
    _required_text(value, name)
    if any(char.isspace() for char in value) or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain whitespace or control characters")


def _bounded_score(value: Any, name: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")


class Contract:
    """Mixin implementing common validation and JSON round trips."""

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        _check_reasoning(self)
        for name in ("schema_version", "run_id", "thread_id", "created_at"):
            _required_text(getattr(self, name), name)
        if not (getattr(self, "model_profile", "") or getattr(self, "implementation", "")):
            raise ValueError("model_profile or implementation is required")
        if hasattr(self, "agent_id"):
            _required_text(getattr(self, "agent_id"), "agent_id")
        return True

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _jsonable({item.name: getattr(self, item.name) for item in fields(self)})

    @classmethod
    def from_dict(cls: Type[T], payload: Mapping[str, Any]) -> T:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects a mapping")
        names = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in names}
        values = _coerce_nested(cls, values)
        result = cls(**values)
        result.validate()
        return result


def _coerce_nested(cls: Type[Contract], values: Dict[str, Any]) -> Dict[str, Any]:
    nested = {
        "evidence": EvidenceRef,
        "evidence_refs": EvidenceRef,
        "visual_evidence": VisualEvidence,
        "process_snapshot": ProcessStateSnapshot,
        "diagnosis_draft": DiagnosisDraft,
        "review_decision": ReviewDecision,
        "assignments": AgentAssignment,
    }
    for name, contract_cls in nested.items():
        if name not in values or values[name] is None:
            continue
        if name in {"evidence", "evidence_refs", "visual_evidence", "assignments"}:
            values[name] = [
                item if isinstance(item, contract_cls) else contract_cls.from_dict(item)
                for item in values[name]
            ]
        elif isinstance(values[name], Mapping):
            values[name] = contract_cls.from_dict(values[name])
    return values


@dataclass
class EvidenceRef(Contract):
    evidence_id: str = field(default_factory=lambda: f"ev-{uuid4().hex[:12]}")
    kind: str = EvidenceKind.UNKNOWN.value
    source_id: str = ""
    locator: str = ""
    quote: str = ""
    confidence: Optional[float] = None
    schema_version: str = "evidence-ref-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "evidence"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.evidence_id, "evidence_id")
        _source_id(self.source_id)
        if self.kind not in {item.value for item in EvidenceKind}:
            raise ValueError(f"invalid evidence kind: {self.kind}")
        _bounded_score(self.confidence, "confidence", allow_none=True)
        if known_source_ids is not None and self.source_id not in set(known_source_ids):
            raise ValueError(f"unknown evidence source_id: {self.source_id}")
        return True


@dataclass
class AgentResult(Contract):
    agent_id: str
    status: str = AgentStatus.COMPLETED.value
    output: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceRef] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "agent-result-1"
    run_id: str = ""
    thread_id: str = ""
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        if self.status not in {item.value for item in AgentStatus}:
            raise ValueError(f"invalid agent status: {self.status}")
        if not isinstance(self.output, dict) or not isinstance(self.metadata, dict):
            raise ValueError("output and metadata must be mappings")
        if not isinstance(self.evidence, list):
            raise ValueError("evidence must be a list")
        for item in self.evidence:
            if not isinstance(item, EvidenceRef):
                raise ValueError("evidence entries must be EvidenceRef objects")
            if not item.run_id:
                item.run_id = self.run_id
            if not item.thread_id:
                item.thread_id = self.thread_id
            if item.run_id != self.run_id or item.thread_id != self.thread_id:
                raise ValueError("evidence must belong to the same run/thread as the result")
            item.validate(known_source_ids=known_source_ids)
        if known_source_ids is not None:
            _check_sources(self.evidence, known_source_ids)
        return True


def _check_sources(refs: Any, known: Iterable[str]) -> None:
    known_set = set(known)
    for ref in refs:
        if isinstance(ref, EvidenceRef) and ref.source_id not in known_set:
            raise ValueError(f"unknown evidence source_id: {ref.source_id}")


@dataclass
class InspectionTask(Contract):
    input_path: str
    run_id: str = field(default_factory=lambda: f"run-{uuid4().hex[:12]}")
    tenant_id: str = "tenant_default"
    session_id: str = "default"
    operator_id: str = "local_operator"
    input_mode: str = InputMode.SINGLE_REPRESENTATIVE_FRAME.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "inspection-task-1"
    thread_id: str = ""
    agent_id: str = "orchestrator"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.workflow"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = f"inspection_{self.tenant_id}_{self.session_id}_{self.run_id}"

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.input_path, "input_path")
        if self.input_mode not in {item.value for item in InputMode}:
            raise ValueError(f"invalid input_mode: {self.input_mode}")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        return True


@dataclass
class MediaManifest(Contract):
    media_id: str
    source_id: str
    path: str = ""
    media_type: str = MediaType.UNKNOWN.value
    sha256: str = ""
    file_hash: str = ""
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    frame_count: Optional[int] = None
    read_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "media-manifest-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "input_normalizer"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.media_id, "media_id")
        _source_id(self.source_id)
        if known_source_ids is not None and self.source_id not in set(known_source_ids):
            raise ValueError(f"unknown media source_id: {self.source_id}")
        if self.media_type not in {item.value for item in MediaType}:
            raise ValueError(f"invalid media_type: {self.media_type}")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.size_bytes is not None and (isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0):
            raise ValueError("size_bytes must be a non-negative integer")
        for name in ("fps", "duration_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{name} must be a non-negative number")
        return True


@dataclass
class VisualEvidence(Contract):
    evidence_id: str = field(default_factory=lambda: f"ve-{uuid4().hex[:12]}")
    source_id: str = "visual"
    media_id: str = ""
    frame_ids: List[int] = field(default_factory=list)
    bbox: Optional[List[float]] = None
    label: str = ""
    score: Optional[float] = None
    uncertainty: Optional[float] = None
    observation: str = ""
    quality_status: str = QualityStatus.PASS.value
    # Stable fields used by the temporal visual baseline.
    class_probs: Dict[str, float] = field(default_factory=dict)
    risk_level: str = RiskLevel.UNKNOWN.value
    temporal_consistency: Optional[float] = None
    key_frame_ids: List[int] = field(default_factory=list)
    static_features: Dict[str, float] = field(default_factory=dict)
    dynamic_features: Dict[str, float] = field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    schema_version: str = "visual-evidence-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "visual_evidence"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)
        if self.source_id == "visual" and self.evidence_refs:
            self.source_id = self.evidence_refs[0].source_id

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.evidence_id, "evidence_id")
        _source_id(self.source_id)
        if known_source_ids is not None and self.source_id not in set(known_source_ids):
            raise ValueError(f"unknown visual evidence source_id: {self.source_id}")
        if self.quality_status not in {item.value for item in QualityStatus}:
            raise ValueError(f"invalid quality_status: {self.quality_status}")
        if self.risk_level not in {item.value for item in RiskLevel}:
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        if self.score is not None:
            _bounded_score(self.score, "score")
        if self.uncertainty is not None:
            _bounded_score(self.uncertainty, "uncertainty")
        if self.bbox is not None and (len(self.bbox) != 4 or any(not isinstance(item, (int, float)) for item in self.bbox)):
            raise ValueError("bbox must contain four numeric values")
        if not isinstance(self.frame_ids, list) or any(not isinstance(item, int) or item < 0 for item in self.frame_ids):
            raise ValueError("frame_ids must contain non-negative integers")
        if not isinstance(self.key_frame_ids, list) or any(not isinstance(item, int) or item < 0 for item in self.key_frame_ids):
            raise ValueError("key_frame_ids must contain non-negative integers")
        if self.temporal_consistency is not None:
            _bounded_score(self.temporal_consistency, "temporal_consistency")
        if not isinstance(self.class_probs, dict) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1
            for value in self.class_probs.values()
        ):
            raise ValueError("class_probs must map labels to scores between 0 and 1")
        if not isinstance(self.static_features, dict) or not isinstance(self.dynamic_features, dict):
            raise ValueError("static_features and dynamic_features must be mappings")
        for ref in self.evidence_refs:
            if not isinstance(ref, EvidenceRef):
                raise ValueError("evidence_refs must contain EvidenceRef objects")
            if not ref.run_id:
                ref.run_id = self.run_id
            if not ref.thread_id:
                ref.thread_id = self.thread_id
            ref.validate(known_source_ids=known_source_ids)
        return True


@dataclass
class ProcessStateSnapshot(Contract):
    process_missing: List[str] = field(default_factory=list)
    alignment_quality: str = AlignmentQuality.UNKNOWN.value
    process_window: Any = None
    process_regime: str = "unknown"
    event_risk: Optional[float] = None
    anomaly_score: Optional[float] = None
    change_points: List[Any] = field(default_factory=list)
    process_tags: List[str] = field(default_factory=list)
    evidence_window: Any = None
    variables: Dict[str, Any] = field(default_factory=dict)
    missing_mask: Dict[str, bool] = field(default_factory=dict)
    schema_version: str = "process-state-snapshot-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "process_state"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        if not isinstance(self.process_missing, list):
            raise ValueError("process_missing must be a list")
        if self.alignment_quality not in {item.value for item in AlignmentQuality}:
            raise ValueError(f"invalid alignment_quality: {self.alignment_quality}")
        for name in ("event_risk", "anomaly_score"):
            value = getattr(self, name)
            if value is not None:
                _bounded_score(value, name)
        if not isinstance(self.process_tags, list) or not isinstance(self.variables, dict) or not isinstance(self.missing_mask, dict):
            raise ValueError("process_tags, variables and missing_mask must have valid container types")
        return True


@dataclass
class InspectionFeatures(Contract):
    image_quality: Optional[float] = None
    roi_coverage: Optional[float] = None
    visual_confidence: Optional[float] = None
    cross_frame_consistency: Optional[float] = None
    novelty_score: Optional[float] = None
    risk_score: Optional[float] = None
    knowledge_coverage: Optional[float] = None
    case_similarity: Optional[float] = None
    evidence_conflict: Optional[float] = None
    process_snapshot: Optional[ProcessStateSnapshot] = None
    evidence_ids: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "inspection-features-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "feature_builder"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        for name in ("image_quality", "roi_coverage", "visual_confidence", "cross_frame_consistency", "novelty_score", "risk_score", "knowledge_coverage", "case_similarity", "evidence_conflict"):
            value = getattr(self, name)
            if value is not None:
                _bounded_score(value, name)
        if self.process_snapshot is not None:
            self.process_snapshot.validate(known_source_ids=known_source_ids)
        if not isinstance(self.evidence_ids, list) or any(not isinstance(item, str) or not item for item in self.evidence_ids):
            raise ValueError("evidence_ids must contain non-empty strings")
        if known_source_ids is not None:
            missing = set(self.evidence_ids) - set(known_source_ids)
            if missing:
                raise ValueError(f"unknown feature evidence source_id: {sorted(missing)[0]}")
        if not isinstance(self.features, dict):
            raise ValueError("features must be a mapping")
        return True


@dataclass
class JevRouteDecision(Contract):
    route: str = Route.HUMAN.value
    difficulty_score: float = 0.0
    novelty_score: float = 0.0
    confidence: float = 0.0
    risk_level: str = RiskLevel.UNKNOWN.value
    similar_case_ids: List[str] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    process_context_used: bool = False
    alignment_quality: str = AlignmentQuality.UNKNOWN.value
    # Identifies the backend that actually produced this decision, separately
    # from model_profile, which identifies the configured router profile.
    router_backend: str = "deterministic_rule"
    model_profile: str = "deterministic_alu_router_v1"
    gate_version: str = "alu-jev-safety-0.1"
    policy_version: str = "base"
    policy_hash: str = "builtin"
    schema_version: str = "jev-route-decision-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "jev_router"
    created_at: str = field(default_factory=utc_now)
    implementation: str = "agentic.contracts"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        if self.route not in {item.value for item in Route}:
            raise ValueError(f"invalid route: {self.route}")
        if self.risk_level not in {item.value for item in RiskLevel}:
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        if self.alignment_quality not in {item.value for item in AlignmentQuality}:
            raise ValueError(f"invalid alignment_quality: {self.alignment_quality}")
        if self.router_backend not in {"deterministic_rule", "laya_adapter", "laya_adapter_fallback", "safety_fallback"}:
            raise ValueError(f"invalid router_backend: {self.router_backend}")
        for name in ("difficulty_score", "novelty_score", "confidence"):
            _bounded_score(getattr(self, name), name)
        if not isinstance(self.process_context_used, bool):
            raise ValueError("process_context_used must be a boolean")
        for name in ("similar_case_ids", "evidence_gaps", "reasons"):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"{name} must be a list")
        _required_text(self.gate_version, "gate_version")
        _required_text(self.policy_version, "policy_version")
        _required_text(self.policy_hash, "policy_hash")
        return True


@dataclass
class DiagnosisDraftPayload:
    """Only the model-authored fields of a draft; audit IDs are injected locally."""

    diagnosis: str
    diagnosis_class: str
    risk_level: str
    confidence: float
    summary: str
    findings: List[str]
    recommendations: List[str]
    inspection_items: List[str]
    open_questions: List[str]
    source_ids: List[str]
    requires_human_review: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiagnosisDraftPayload":
        payload = cls(**dict(value))
        payload.validate()
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        properties: Dict[str, Any] = {
            "diagnosis": {"type": "string"},
            "diagnosis_class": {"type": "string"},
            "risk_level": {"type": "string", "enum": [item.value for item in RiskLevel]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "inspection_items": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "source_ids": {"type": "array", "items": {"type": "string"}},
            "requires_human_review": {"type": "boolean"},
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    @classmethod
    def fake_payload(cls) -> Dict[str, Any]:
        return {
            "diagnosis": "flow-validation draft",
            "diagnosis_class": "visual_state",
            "risk_level": RiskLevel.UNKNOWN.value,
            "confidence": 0.0,
            "summary": "Fake structured output for gateway validation.",
            "findings": [],
            "recommendations": [],
            "inspection_items": [],
            "open_questions": [],
            "source_ids": [],
            "requires_human_review": True,
        }

    def validate(self) -> bool:
        for name in (
            "diagnosis", "diagnosis_class", "summary", "findings", "recommendations",
            "inspection_items", "open_questions", "source_ids", "requires_human_review",
        ):
            value = getattr(self, name)
            if name in {"findings", "recommendations", "inspection_items", "open_questions", "source_ids"}:
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise ValueError(f"{name} must be a list of strings")
            elif name == "requires_human_review":
                if not isinstance(value, bool):
                    raise ValueError("requires_human_review must be a boolean")
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if self.risk_level not in {item.value for item in RiskLevel}:
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        _bounded_score(self.confidence, "confidence")
        return True


@dataclass
class DiagnosisDraft(Contract):
    diagnosis: str = ""
    diagnosis_class: str = ""
    risk_level: str = RiskLevel.UNKNOWN.value
    confidence: float = 0.0
    summary: str = ""
    evidence: List[EvidenceRef] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    inspection_items: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    requires_human_review: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "diagnosis-draft-1"
    run_id: str = ""
    thread_id: str = ""
    agent_id: str = "diagnosis"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "agentic.contracts"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        if self.risk_level not in {item.value for item in RiskLevel}:
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        _bounded_score(self.confidence, "confidence")
        if not isinstance(self.requires_human_review, bool):
            raise ValueError("requires_human_review must be a boolean")
        for item in self.evidence:
            if not isinstance(item, EvidenceRef):
                raise ValueError("evidence entries must be EvidenceRef objects")
            if not item.run_id:
                item.run_id = self.run_id
            if not item.thread_id:
                item.thread_id = self.thread_id
            if item.run_id != self.run_id or item.thread_id != self.thread_id:
                raise ValueError("evidence must belong to the same run/thread as the draft")
            item.validate(known_source_ids=known_source_ids)
        for source_id in self.source_ids:
            _source_id(source_id)
        return True


@dataclass
class ReviewDecision(Contract):
    run_id: str
    action: str
    reviewer_id: str
    reason: str = ""
    amendments: Dict[str, Any] = field(default_factory=dict)
    decided_at: str = field(default_factory=utc_now)
    schema_version: str = "review-decision-1"
    thread_id: str = ""
    agent_id: str = "human_reviewer"
    created_at: str = field(default_factory=utc_now)
    model_profile: str = ""
    implementation: str = "human_review"
    evidence_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.reviewer_id, "reviewer_id")
        if self.action not in {item.value for item in ReviewAction}:
            raise ValueError(f"invalid review action: {self.action}")
        if not isinstance(self.amendments, dict):
            raise ValueError("amendments must be a mapping")
        if not isinstance(self.evidence_ids, list):
            raise ValueError("evidence_ids must be a list")
        if known_source_ids is not None:
            for source_id in self.evidence_ids:
                if source_id not in set(known_source_ids):
                    raise ValueError(f"unknown review evidence source_id: {source_id}")
        return True


@dataclass
class TraceEvent(Contract):
    run_id: str
    node: str
    status: str
    started_at: str = field(default_factory=utc_now)
    elapsed_ms: Optional[float] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    schema_version: str = "trace-event-1"
    created_at: str = field(default_factory=utc_now)
    thread_id: str = ""
    ticket_id: str = ""
    agent_id: str = "orchestrator"
    stage: str = ""
    finished_at: Optional[str] = None
    model_profile: str = ""
    implementation: str = "agentic.runtime"
    attempt: int = 1
    error_code: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.node, "node")
        _required_text(self.status, "status")
        if self.elapsed_ms is not None and (isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)) or self.elapsed_ms < 0):
            raise ValueError("elapsed_ms must be a non-negative number")
        if not isinstance(self.summary, dict) or not isinstance(self.evidence_ids, list):
            raise ValueError("summary must be a mapping and evidence_ids must be a list")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return True


@dataclass
class DiagnosisTicket(Contract):
    ticket_id: str = field(default_factory=lambda: f"ticket-{uuid4().hex[:12]}")
    run_id: str = ""
    thread_id: str = ""
    status: str = TicketStatus.OPEN.value
    priority: str = "normal"
    title: str = ""
    # DeepSeek-style ticket fields are explicit so a ticket can be resumed or
    # inspected without unpacking opaque metadata.
    stage: str = "created"
    attempt: int = 1
    route: str = "human"
    worker_id: str = "orchestrator"
    health: str = "healthy"
    retry_count: int = 0
    report_id: str = ""
    human_decision: Optional[Dict[str, Any]] = None
    evidence_ids: List[str] = field(default_factory=list)
    diagnosis_draft: Optional[DiagnosisDraft] = None
    assignments: List["AgentAssignment"] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    agent_id: str = "orchestrator"
    model_profile: str = ""
    implementation: str = "agentic.contracts"
    schema_version: str = "diagnosis-ticket-1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.ticket_id, "ticket_id")
        if self.status not in {item.value for item in TicketStatus}:
            raise ValueError(f"invalid ticket status: {self.status}")
        if self.route not in {item.value for item in Route}:
            raise ValueError(f"invalid ticket route: {self.route}")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        _required_text(self.stage, "stage")
        _required_text(self.worker_id, "worker_id")
        _required_text(self.health, "health")
        if self.human_decision is not None and not isinstance(self.human_decision, dict):
            raise ValueError("human_decision must be a mapping or null")
        if not isinstance(self.evidence_ids, list) or not isinstance(self.assignments, list):
            raise ValueError("evidence_ids and assignments must be lists")
        if known_source_ids is not None:
            missing = set(self.evidence_ids) - set(known_source_ids)
            if missing:
                raise ValueError(f"unknown ticket evidence source_id: {sorted(missing)[0]}")
        if self.diagnosis_draft is not None:
            self.diagnosis_draft.validate(known_source_ids=known_source_ids)
        for assignment in self.assignments:
            if not isinstance(assignment, AgentAssignment):
                raise ValueError("assignments must contain AgentAssignment objects")
            if not assignment.run_id:
                assignment.run_id = self.run_id
            if not assignment.thread_id:
                assignment.thread_id = self.thread_id
            if assignment.run_id != self.run_id or assignment.thread_id != self.thread_id:
                raise ValueError("assignment must belong to the same run/thread as the ticket")
            assignment.validate()
        return True


@dataclass
class AgentAssignment(Contract):
    assignment_id: str = field(default_factory=lambda: f"assign-{uuid4().hex[:12]}")
    ticket_id: str = ""
    agent_id: str = ""
    role: str = ""
    status: str = AssignmentStatus.ASSIGNED.value
    model_profile: str = ""
    implementation: str = "agentic.contracts"
    attempt: int = 1
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    schema_version: str = "agent-assignment-1"
    run_id: str = ""
    thread_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.assignment_id, "assignment_id")
        _required_text(self.ticket_id, "ticket_id")
        _required_text(self.agent_id, "agent_id")
        _required_text(self.role, "role")
        if self.status not in {item.value for item in AssignmentStatus}:
            raise ValueError(f"invalid assignment status: {self.status}")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return True


@dataclass
class TaskEvent(Contract):
    """Observable source/thread event used by the Ticket/Worker boundary.

    This is deliberately a small event envelope.  It records what happened
    and where it belongs without carrying private model reasoning.
    """

    event_id: str = field(default_factory=lambda: f"event-{uuid4().hex[:12]}")
    event_type: str = "task.created"
    run_id: str = ""
    thread_id: str = ""
    ticket_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    agent_id: str = "orchestrator"
    model_profile: str = ""
    implementation: str = "agentic.tickets"
    schema_version: str = "task-event-1"

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = _default_thread(self.run_id)

    def validate(self, known_source_ids: Optional[Iterable[str]] = None) -> bool:
        super().validate()
        _required_text(self.event_id, "event_id")
        _required_text(self.event_type, "event_type")
        _required_text(self.ticket_id, "ticket_id")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a mapping")
        return True


def jsonable(value: Any) -> Any:
    return _jsonable(value)


__all__ = [
    "AgentAssignment", "AgentResult", "AgentStatus", "AlignmentQuality",
    "AssignmentStatus", "DiagnosisDraft", "DiagnosisDraftPayload", "DiagnosisTicket", "EvidenceKind",
    "EvidenceRef", "InspectionFeatures", "InspectionTask", "InputMode",
    "JevRouteDecision", "MediaManifest", "MediaType", "ProcessStateSnapshot",
    "QualityStatus", "ReviewAction", "ReviewDecision", "RiskLevel", "Route",
    "TaskEvent", "TicketStatus", "TraceEvent", "VisualEvidence", "Contract", "ContractEnum",
    "jsonable", "utc_now",
]
