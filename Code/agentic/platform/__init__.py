"""Reusable runtime pieces adapted from the local EduAgent reference project.

The platform layer deliberately has no dependency on the education domain. It
provides contracts, run tracing, session isolation, and controlled case memory
for the aluminum-electrolysis agents.
"""

from .contracts import (
    AgentAssignment,
    AgentResult,
    DiagnosisDraft,
    DiagnosisDraftPayload,
    DiagnosisTicket,
    EvidenceRef,
    InspectionFeatures,
    InspectionTask,
    JevRouteDecision,
    MediaManifest,
    ProcessStateSnapshot,
    ReviewDecision,
    TraceEvent,
    VisualEvidence,
)
from .memory import build_thread_config, get_memory_saver
from .memory import JsonCheckpointStore, LocalCheckpointStore, PersistentCheckpointStore, SQLiteCheckpointStore
from .llm_gateway import GatewayResponse, LLMGateway
from .knowledge import KnowledgeRule, VersionedKnowledgeBase
from .case_memory import SpecialCaseMemory
from .retrieval import LocalKnowledgeRetriever
from .runtime import InspectionRuntime
from .hitl import (
    ALLOWED_HUMAN_ACTIONS,
    AlreadyResumedError,
    HumanReviewError,
    HumanReviewService,
    InvalidActionError,
    ReviewNotFoundError,
)

__all__ = [
    "AgentResult",
    "AgentAssignment",
    "DiagnosisDraft",
    "DiagnosisDraftPayload",
    "DiagnosisTicket",
    "EvidenceRef",
    "InspectionFeatures",
    "InspectionTask",
    "JevRouteDecision",
    "MediaManifest",
    "ProcessStateSnapshot",
    "ReviewDecision",
    "TraceEvent",
    "VisualEvidence",
    "GatewayResponse",
    "LLMGateway",
    "KnowledgeRule",
    "VersionedKnowledgeBase",
    "SpecialCaseMemory",
    "InspectionRuntime",
    "JsonCheckpointStore",
    "SQLiteCheckpointStore",
    "PersistentCheckpointStore",
    "LocalCheckpointStore",
    "ALLOWED_HUMAN_ACTIONS",
    "AlreadyResumedError",
    "HumanReviewError",
    "HumanReviewService",
    "InvalidActionError",
    "ReviewNotFoundError",
    "build_thread_config",
    "get_memory_saver",
    "LocalKnowledgeRetriever",
]
