"""Knowledge retrieval boundary for the aluminum-electrolysis agents.

The initial implementation is dependency-free and mirrors the reference
project's idea of returning ranked chunks with source IDs. A future BGE-M3 /
Milvus / MCP adapter can implement the same ``search`` method.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part[index:index + 2] for index in range(max(1, len(part) - 1)))
        else:
            tokens.append(part)
    return tokens


class LocalKnowledgeRetriever:
    """Rank local Markdown/text passages and return auditable source metadata."""

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = Path(knowledge_dir)

    def _chunks(self) -> list[Dict[str, str]]:
        chunks: list[Dict[str, str]] = []
        for path in sorted(self.knowledge_dir.glob("**/*")):
            if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8-sig")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            for index, paragraph in enumerate(paragraphs, start=1):
                if len(paragraph) < 16:
                    continue
                chunks.append({
                    "source_id": f"{path.stem}:{index}",
                    "source": path.name,
                    "text": paragraph,
                    "source_version": "local",
                    "locator": f"file:{path.name}:paragraph:{index}",
                })
        return chunks

    def search(self, visual: Dict[str, Any], top_k: int = 5) -> list[Dict[str, str]]:
        chunks = self._chunks()
        if not chunks:
            return []
        query = (
            "铝电解 火眼 过热度 电解质温度 初晶温度 图像状态 C N H "
            + str(visual.get("class_label", ""))
        )
        query_terms = set(_tokens(query))
        doc_terms = [set(_tokens(item["text"])) for item in chunks]
        frequencies: Dict[str, int] = {}
        for terms in doc_terms:
            for term in terms & query_terms:
                frequencies[term] = frequencies.get(term, 0) + 1

        scored: list[tuple[float, Dict[str, str]]] = []
        doc_count = len(chunks)
        for item, terms in zip(chunks, doc_terms):
            overlap = terms & query_terms
            if not overlap:
                continue
            score = sum(math.log(1 + doc_count / (1 + frequencies[t])) for t in overlap)
            score /= math.sqrt(max(len(terms), 1))
            enriched = dict(item)
            enriched["score"] = f"{score:.6f}"
            scored.append((score, enriched))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[:max(1, top_k)] if score > 0]


class RetrieverProtocol:
    """Documentation-only protocol for future Milvus/MCP implementations."""

    def search(self, visual: Dict[str, Any], top_k: int = 5) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
