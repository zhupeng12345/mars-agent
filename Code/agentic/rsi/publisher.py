"""Human-gated candidate approval, publication, and rollback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from agentic.platform.jev.policy import JevRuntimePolicy

from .candidate import Candidate
from .gate import GateResult
from .lineage import LineageStore, VersionSnapshot
from .records import jsonable, stable_hash, utc_now


@dataclass(frozen=True)
class Approval:
    candidate_id: str
    reviewer_id: str
    reason: str
    candidate_hash: str = ""
    gate_hash: str = ""
    approved_at: str = field(default_factory=utc_now)
    gate_version: str = "rsi-gate-1"
    schema_version: str = "rsi-approval-1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class Publisher:
    """Local registry; no method mutates the online runtime or policy files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lineage = LineageStore(self.root)
        self.approvals_path = self.root / "approvals.jsonl"
        self.events_path = self.root / "publish_events.jsonl"
        self.candidates_path = self.root / "candidates.jsonl"
        self._candidates: dict[str, Candidate] = {}

    def register(self, candidate: Candidate) -> Candidate:
        candidate.validate()
        self._candidates[candidate.candidate_id] = candidate
        with self.candidates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return candidate

    def _resolve(self, candidate: Union[Candidate, str]) -> Candidate:
        if isinstance(candidate, Candidate):
            self._candidates[candidate.candidate_id] = candidate
            return candidate
        try:
            return self._candidates[str(candidate)]
        except KeyError:
            if self.candidates_path.exists():
                for line in reversed(self.candidates_path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("candidate_id") == str(candidate):
                        value = Candidate.from_dict(row)
                        self._candidates[value.candidate_id] = value
                        return value
            raise KeyError(candidate)

    def _append_event(self, event: Mapping[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(dict(event)), ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _validated_gate(candidate: Candidate, gate_result: Union[GateResult, Mapping[str, Any], None]) -> GateResult:
        if isinstance(gate_result, Mapping):
            gate = GateResult.from_dict(gate_result)
        elif isinstance(gate_result, GateResult):
            gate = gate_result
        else:
            raise ValueError("a complete accepted gate result is required")
        gate.validate_for(candidate)
        return gate

    def approve(
        self,
        candidate: Union[Candidate, str],
        reviewer_id: str,
        reason: str,
        gate_result: Optional[Union[GateResult, Mapping[str, Any]]] = None,
    ) -> Approval:
        value = self._resolve(candidate)
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required for publication approval")
        gate = self._validated_gate(value, gate_result)
        approval = Approval(
            value.candidate_id,
            reviewer_id.strip(),
            reason.strip(),
            candidate_hash=value.diff_hash,
            gate_hash=stable_hash(gate.to_dict()),
            gate_version=gate.gate_version,
        )
        with self.approvals_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(approval.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        value.status = "approved"
        self._append_event({"event": "approved", "candidate_id": value.candidate_id, "reviewer_id": reviewer_id, "at": approval.approved_at})
        return approval

    def _has_approval(self, candidate: Candidate, gate: GateResult) -> bool:
        if not self.approvals_path.exists():
            return False
        for line in self.approvals_path.read_text(encoding="utf-8").splitlines():
            try:
                approval = json.loads(line)
                if (
                    approval.get("candidate_id") == candidate.candidate_id
                    and approval.get("candidate_hash") == candidate.diff_hash
                    and approval.get("gate_hash") == stable_hash(gate.to_dict())
                    and approval.get("gate_version") == gate.gate_version
                ):
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def publish(
        self,
        candidate: Union[Candidate, str],
        gate_result: Union[GateResult, Mapping[str, Any]],
        *,
        reviewer_id: Optional[str] = None,
    ) -> VersionSnapshot:
        value = self._resolve(candidate)
        gate = self._validated_gate(value, gate_result)
        if reviewer_id and not self._has_approval(value, gate):
            self.approve(value, reviewer_id, "approval supplied at publish", gate)
        if not self._has_approval(value, gate):
            raise ValueError("manual approval is required before publication")
        current = self.lineage.active()
        parent = current.version_id if current else (value.parent_version if value.parent_version != "base" else None)
        surfaces = value.apply(current.surfaces if current else value.base_snapshot)
        if "jev_policies" in value.surfaces:
            JevRuntimePolicy.from_mapping(surfaces.get("jev_policies"))
        version = self.lineage.record(
            surfaces,
            parent_version=parent,
            candidate_id=value.candidate_id,
            metadata={"gate": gate.to_dict()},
        )
        self.lineage.set_active(version.version_id)
        value.status = "published"
        self._append_event({"event": "published", "candidate_id": value.candidate_id, "version_id": version.version_id, "at": utc_now()})
        return version

    def rollback(self, version_id: Optional[str] = None) -> VersionSnapshot:
        current = self.lineage.active()
        if current is None:
            raise ValueError("no active version to roll back")
        target_id = version_id or current.parent_version
        if not target_id:
            versions = self.lineage.list()
            prior = [item for item in versions if item.version_id != current.version_id]
            target_id = prior[-1].version_id if prior else None
        if not target_id:
            raise ValueError("no previous version available")
        target = self.lineage.set_active(target_id)
        self._append_event({"event": "rollback", "from_version": current.version_id, "to_version": target.version_id, "at": utc_now()})
        return target

    def current(self) -> Optional[VersionSnapshot]:
        return self.lineage.active()


__all__ = ["Approval", "Publisher"]
