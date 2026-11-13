"""Strict, monotonic runtime policy surface for the optional Laya router."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_PROMPT_VARIANT = "diagnosis-aggregator-v2"
ALLOWED_PROMPT_VARIANTS = frozenset({
    DEFAULT_PROMPT_VARIANT,
    "diagnosis-evidence-first-v1",
})


def parse_prompt_surface(value: Any) -> str:
    """Validate an RSI prompt surface against code-owned variants only."""
    if value is None:
        return DEFAULT_PROMPT_VARIANT
    if not isinstance(value, Mapping):
        raise ValueError("prompts must be an object")
    unknown = set(value) - {"variant"}
    if unknown:
        raise ValueError(f"unsupported prompt surface keys: {sorted(unknown)}")
    variant = value.get("variant", DEFAULT_PROMPT_VARIANT)
    if not isinstance(variant, str) or variant not in ALLOWED_PROMPT_VARIANTS:
        raise ValueError("prompts.variant is not a code-owned prompt variant")
    return variant


def parse_knowledge_query_surface(value: Any) -> tuple[str, ...]:
    """Validate bounded query terms; RSI cannot inject arbitrary prompts."""
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("knowledge_queries must be an object")
    unknown = set(value) - {"extra_terms"}
    if unknown:
        raise ValueError(f"unsupported knowledge query keys: {sorted(unknown)}")
    terms = value.get("extra_terms", [])
    if not isinstance(terms, (list, tuple)) or len(terms) > 4:
        raise ValueError("knowledge_queries.extra_terms must contain at most four terms")
    normalized: list[str] = []
    for term in terms:
        if not isinstance(term, str):
            raise ValueError("knowledge query terms must be strings")
        cleaned = " ".join(term.split())
        if not cleaned or len(cleaned) > 40 or any(ord(char) < 32 for char in cleaned):
            raise ValueError("knowledge query terms must be short printable strings")
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


@dataclass(frozen=True)
class JevRuntimePolicy:
    """RSI may tighten Laya proposal acceptance, never lower the safety bar."""

    minimum_laya_confidence: float = 0.7

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "JevRuntimePolicy":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("jev_policies must be an object")
        unknown = set(value) - {"minimum_laya_confidence"}
        if unknown:
            raise ValueError(f"unsupported JEV runtime policy keys: {sorted(unknown)}")
        confidence = value.get("minimum_laya_confidence", 0.7)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("minimum_laya_confidence must be a number")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.7 <= confidence <= 1.0:
            raise ValueError("minimum_laya_confidence must be within [0.7, 1.0]")
        return cls(minimum_laya_confidence=confidence)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


__all__ = [
    "ALLOWED_PROMPT_VARIANTS", "DEFAULT_PROMPT_VARIANT", "JevRuntimePolicy",
    "parse_knowledge_query_surface", "parse_prompt_surface",
]
