"""Resolve an explicitly configured, human-approved RSI snapshot for runtime use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic.platform.jev.policy import (
    DEFAULT_PROMPT_VARIANT,
    JevRuntimePolicy,
    parse_knowledge_query_surface,
    parse_prompt_surface,
)

from .lineage import LineageStore, VersionSnapshot
from .publisher import Publisher
from .records import stable_hash


@dataclass(frozen=True)
class RuntimeJevPolicy:
    policy: JevRuntimePolicy
    knowledge_query_terms: tuple[str, ...] = ()
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    version_id: str = "base"
    policy_hash: str = "builtin"
    status: str = "disabled"
    reason: str = "registry_not_configured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "version_id": self.version_id,
            "policy_hash": self.policy_hash,
            "policy": self.policy.to_dict(),
            "knowledge_query_terms": list(self.knowledge_query_terms),
            "prompt_variant": self.prompt_variant,
        }


def _approved_active_snapshot(root: Path) -> VersionSnapshot | None:
    pointer = root / "active_version.json"
    if not pointer.exists():
        return None
    import json

    payload = json.loads(pointer.read_text(encoding="utf-8"))
    version_id = payload.get("version_id") if isinstance(payload, dict) else None
    if not isinstance(version_id, str) or not version_id:
        raise ValueError("active version pointer is invalid")
    lineage = LineageStore(root)
    snapshot = lineage.get(version_id)
    if snapshot is None:
        raise ValueError("active version snapshot is missing")
    if snapshot.content_hash != stable_hash(snapshot.surfaces):
        raise ValueError("active version content hash mismatch")
    if not snapshot.candidate_id:
        raise ValueError("active version is not bound to a candidate")

    publisher = Publisher(root)
    candidate = publisher._resolve(snapshot.candidate_id)
    gate = publisher._validated_gate(candidate, snapshot.metadata.get("gate"))
    if not publisher._has_approval(candidate, gate):
        raise ValueError("active version has no matching human approval")
    parent = lineage.get(snapshot.parent_version) if snapshot.parent_version else None
    expected = candidate.apply(parent.surfaces if parent else candidate.base_snapshot)
    if expected != snapshot.surfaces:
        raise ValueError("active version surfaces do not match its approved candidate")
    return snapshot


def resolve_runtime_jev_policy(registry_root: Path | str | None) -> RuntimeJevPolicy:
    """Load only the bounded JEV policy from an explicitly selected registry.

    Missing, malformed, unapproved, or unsupported snapshots use the built-in
    policy. RSI demo registries are never discovered implicitly.
    """
    default = JevRuntimePolicy()
    if registry_root is None or not str(registry_root).strip():
        return RuntimeJevPolicy(default)
    root = Path(registry_root).expanduser()
    try:
        snapshot = _approved_active_snapshot(root)
        if snapshot is None:
            return RuntimeJevPolicy(default, status="disabled", reason="no_active_version")
        allowed = {"jev_policies", "knowledge_queries", "prompts"}
        unsupported = sorted(key for key in snapshot.surfaces if key not in allowed and snapshot.surfaces[key])
        if unsupported:
            return RuntimeJevPolicy(default, status="fallback", reason=f"unsupported_surfaces:{','.join(unsupported)}")
        policy = JevRuntimePolicy.from_mapping(snapshot.surfaces.get("jev_policies", {}))
        query_terms = parse_knowledge_query_surface(snapshot.surfaces.get("knowledge_queries"))
        prompt_variant = parse_prompt_surface(snapshot.surfaces.get("prompts"))
        policy_hash = stable_hash({
            "version_id": snapshot.version_id,
            "policy": policy.to_dict(),
            "knowledge_query_terms": list(query_terms),
            "prompt_variant": prompt_variant,
        })
        return RuntimeJevPolicy(
            policy,
            query_terms,
            prompt_variant,
            snapshot.version_id,
            policy_hash,
            "active",
            "approved_snapshot_loaded",
        )
    except Exception as exc:
        return RuntimeJevPolicy(default, status="fallback", reason=f"invalid_registry:{type(exc).__name__}")


__all__ = ["RuntimeJevPolicy", "resolve_runtime_jev_policy"]
