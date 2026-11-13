"""Append-only records used by the offline RSI loop.

Records deliberately copy runtime and review data.  They do not grant the RSI
writer access to the online runtime, frozen manifests, safety policy, or model
weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_rsi_dir() -> Path:
    configured = os.getenv("MARS_AGENT_RSI_DIR", "")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "runtime_data" / "rsi"


@dataclass(frozen=True)
class RsiRecord:
    """One immutable input or lifecycle event."""

    record_type: str
    payload: dict[str, Any]
    record_id: str = field(default_factory=lambda: f"rsi-{uuid4().hex[:16]}")
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "rsi-record-1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class CritiqueRecord:
    """Structured Reflexion memory derived from observable failures."""

    run_id: str
    agent_id: str
    failure_type: str
    observation: str
    root_cause: str
    candidate_fix: str
    repeat_count: int = 1
    eligible_for_rsi: bool = False
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "rsi-critique-1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


def critique_from_run(run: Mapping[str, Any], *, repeat_count: int = 1) -> CritiqueRecord:
    result = run.get("result", run)
    result = result if isinstance(result, Mapping) else {}
    review = result.get("review", {})
    issues = list(review.get("issues", [])) if isinstance(review, Mapping) else []
    issue_text = "; ".join(str(item) for item in issues)
    failure = "invalid_citation" if "citation" in issue_text or "来源" in issue_text else "missing_evidence" if issues else "format_error"
    observation = issue_text or "No deterministic review failure was supplied"
    return CritiqueRecord(
        run_id=str(run.get("run_id", result.get("run_id", "unknown"))),
        agent_id="diagnosis_aggregator",
        failure_type=failure,
        observation=observation,
        root_cause="Structured review found an unsupported or incomplete evidence handoff",
        candidate_fix="Require source_ids and traceable evidence references in the diagnosis template",
        repeat_count=max(1, int(repeat_count)),
        eligible_for_rsi=bool(issues) and int(repeat_count) >= 2,
    )


class RsiRecordStore:
    """Small JSONL store with deterministic reads and append-only writes."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_rsi_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "records.jsonl"
        self._lock = Lock()

    def append(self, record_type: str, payload: Mapping[str, Any]) -> RsiRecord:
        record = RsiRecord(record_type=str(record_type), payload=copy.deepcopy(dict(payload)))
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return record

    def read(self, record_type: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and (record_type is None or row.get("record_type") == record_type):
                rows.append(row)
        return rows

    def collect_runtime(self, runtime_dir: Path) -> int:
        """Copy observable runtime inputs into the RSI record log.

        The source files remain untouched.  Missing files are simply omitted so
        a partially populated local runtime can still produce a candidate.
        """
        runtime_dir = Path(runtime_dir)
        count = 0
        mapping = {
            "runs.jsonl": "runtime_run",
            "events.jsonl": "runtime_event",
            "review_decisions.jsonl": "human_review",
            "case_candidates.jsonl": "case_candidate",
            "approved_cases.jsonl": "approved_case",
        }
        for filename, record_type in mapping.items():
            source = runtime_dir / filename
            if not source.exists():
                continue
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self.append(record_type, payload)
                    count += 1
        return count

    def collect_evaluation(self, evaluation_dir: Path, filenames: Iterable[str]) -> int:
        count = 0
        for filename in filenames:
            path = Path(evaluation_dir) / filename
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                rows = path.read_text(encoding="utf-8").splitlines()
            else:
                rows = [path.read_text(encoding="utf-8")]
            for line in rows:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self.append("evaluation_input", {"file": filename, "row": payload})
                    count += 1
        return count

    def collect(
        self,
        *,
        runtime_dir: Optional[Path] = None,
        evaluation_dir: Optional[Path] = None,
        evaluation_files: Iterable[str] = ("route_manifest.jsonl", "knowledge_queries.jsonl", "hitl_cases.jsonl"),
    ) -> int:
        """Convenience ingestion entry point used by scripts and notebooks."""
        count = self.collect_runtime(Path(runtime_dir)) if runtime_dir else 0
        if evaluation_dir:
            count += self.collect_evaluation(Path(evaluation_dir), evaluation_files)
        return count


__all__ = ["CritiqueRecord", "RsiRecord", "RsiRecordStore", "critique_from_run", "default_rsi_dir", "jsonable", "stable_hash", "utc_now"]
