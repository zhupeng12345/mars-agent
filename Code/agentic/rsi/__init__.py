"""Controlled, offline-only RSI (recursive self-improvement) primitives.

The package treats changes as data.  A candidate is reviewed, replayed against
immutable splits, gated, and explicitly published before it can become active.
"""

from .candidate import (
    ALLOWED_MUTABLE_SURFACES,
    Candidate,
    CandidateDiff,
    CandidateReferenceError,
    ForbiddenMutationError,
    MUTABLE_SURFACES,
    build_candidate,
    build_candidate_from_records,
    diff_snapshot,
)
from .evaluator import OfflineReplayer, ReplayBundle, ReplayResult, evaluate_candidate, replay_candidate
from .gate import Gate, GateConfig, GateResult, evaluate_gate
from .lineage import LineageStore, VersionSnapshot
from .publisher import Approval, Publisher
from .records import RsiRecordStore

__all__ = [
    "ALLOWED_MUTABLE_SURFACES", "MUTABLE_SURFACES", "Approval", "Candidate", "CandidateDiff",
    "CandidateReferenceError", "ForbiddenMutationError", "Gate", "GateConfig",
    "GateResult", "LineageStore", "OfflineReplayer", "Publisher",
    "ReplayBundle", "ReplayResult", "RsiRecordStore", "VersionSnapshot",
    "build_candidate", "build_candidate_from_records", "diff_snapshot", "evaluate_candidate", "evaluate_gate", "replay_candidate",
]
