"""Versioned knowledge rules and auditable retrieval helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                terms.add(token)
            else:
                terms.update(token[index:index + 2] for index in range(len(token) - 1))
        else:
            terms.add(token)
    return terms


@dataclass(frozen=True)
class KnowledgeRule:
    rule_id: str
    title: str
    source: str
    source_version: str
    effective_date: str
    preconditions: tuple[str, ...]
    applicability: tuple[str, ...]
    risk_level: str
    allowed_output: tuple[str, ...]
    text: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeRule":
        required = ["rule_id", "title", "source", "source_version", "effective_date", "risk_level", "text"]
        missing = [name for name in required if not str(payload.get(name, "")).strip()]
        if missing:
            raise ValueError(f"knowledge rule missing fields: {missing}")
        return cls(
            rule_id=str(payload["rule_id"]), title=str(payload["title"]), source=str(payload["source"]),
            source_version=str(payload["source_version"]), effective_date=str(payload["effective_date"]),
            preconditions=tuple(str(item) for item in payload.get("preconditions", [])),
            applicability=tuple(str(item) for item in payload.get("applicability", [])),
            risk_level=str(payload["risk_level"]), allowed_output=tuple(str(item) for item in payload.get("allowed_output", [])),
            text=str(payload["text"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "title": self.title, "source": self.source, "source_version": self.source_version, "effective_date": self.effective_date, "preconditions": list(self.preconditions), "applicability": list(self.applicability), "risk_level": self.risk_level, "allowed_output": list(self.allowed_output), "text": self.text}


class VersionedKnowledgeBase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rules: list[KnowledgeRule] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        try:
                            self.rules.append(KnowledgeRule.from_dict(payload))
                        except (TypeError, ValueError):
                            continue

    def search(self, query: str, *, top_k: int = 5, as_of: Optional[str] = None, applicability: Iterable[str] = ()) -> list[dict[str, Any]]:
        terms = _terms(query)
        allowed = set(applicability)
        cutoff = date.fromisoformat(as_of) if as_of else None
        rows = []
        for rule in self.rules:
            try:
                effective = date.fromisoformat(rule.effective_date)
            except ValueError:
                continue
            if cutoff and effective > cutoff:
                continue
            if allowed and rule.applicability and not allowed.intersection(rule.applicability):
                continue
            text = _terms(" ".join((rule.title, rule.text, *rule.preconditions, *rule.applicability)))
            score = len(terms.intersection(text))
            if score:
                rows.append((score / max(1, len(terms)), rule))
        rows.sort(key=lambda item: (-item[0], item[1].rule_id))
        return [{
            "source_id": rule.rule_id,
            "source": rule.source,
            "source_version": rule.source_version,
            "effective_date": rule.effective_date,
            "applicability": list(rule.applicability),
            "risk_level": rule.risk_level,
            "score": score,
            "locator": f"rule:{rule.rule_id}:paragraph:1",
            "text": rule.text,
        } for score, rule in rows[:max(1, top_k)]]

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        as_of: Optional[str] = None,
        applicability: Iterable[str] = (),
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        """Dependency-free BM25 over versioned rule passages.

        This is the local lexical retrieval tier. It preserves the same
        provenance fields as ``search`` so a vector/reranker adapter can be
        swapped in without changing agent contracts.
        """
        if not math.isfinite(float(k1)) or float(k1) <= 0:
            raise ValueError("k1 must be a positive finite number")
        if not math.isfinite(float(b)) or not 0 <= float(b) <= 1:
            raise ValueError("b must be a finite number in [0, 1]")
        terms = _terms(query)
        allowed = set(applicability)
        cutoff = date.fromisoformat(as_of) if as_of else None
        candidates: list[KnowledgeRule] = []
        for rule in self.rules:
            try:
                effective = date.fromisoformat(rule.effective_date)
            except ValueError:
                continue
            if cutoff and effective > cutoff:
                continue
            if allowed and rule.applicability and not allowed.intersection(rule.applicability):
                continue
            candidates.append(rule)
        if not candidates or not terms:
            return []
        documents = [_terms(" ".join((rule.title, rule.text, *rule.preconditions, *rule.applicability))) for rule in candidates]
        average_length = sum(len(document) for document in documents) / max(len(documents), 1)
        document_frequency: dict[str, int] = {}
        for document in documents:
            for term in terms.intersection(document):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        scored: list[tuple[float, KnowledgeRule]] = []
        for rule, document in zip(candidates, documents):
            length = len(document)
            score = 0.0
            for term in terms.intersection(document):
                frequency = sum(1 for token in document if token == term)
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + float(k1) * (1.0 - float(b) + float(b) * length / max(average_length, 1e-9))
                score += idf * (frequency * (float(k1) + 1.0)) / max(denominator, 1e-9)
            if score > 0:
                scored.append((score, rule))
        scored.sort(key=lambda item: (-item[0], item[1].rule_id))
        return [
            {
                "source_id": rule.rule_id,
                "source": rule.source,
                "source_version": rule.source_version,
                "effective_date": rule.effective_date,
                "applicability": list(rule.applicability),
                "risk_level": rule.risk_level,
                "score": score,
                "retrieval_mode": "versioned_bm25",
                "locator": f"rule:{rule.rule_id}:paragraph:1",
                "text": rule.text,
            }
            for score, rule in scored[:max(1, top_k)]
        ]

    def get_rule(self, rule_id: str) -> Optional[KnowledgeRule]:
        """Resolve a citation only against rules already loaded from this pack."""
        wanted = str(rule_id)
        if not wanted or len(wanted) > 200:
            return None
        return next((rule for rule in self.rules if rule.rule_id == wanted), None)
