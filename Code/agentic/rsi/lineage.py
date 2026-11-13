"""Version snapshots and parent/rollback lineage for published candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from .records import jsonable, stable_hash, utc_now


@dataclass
class VersionSnapshot:
    version_id: str
    parent_version: Optional[str]
    surfaces: dict[str, Any]
    candidate_id: Optional[str] = None
    status: str = "active"
    created_at: str = field(default_factory=utc_now)
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "rsi-version-1"

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = stable_hash(self.surfaces)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionSnapshot":
        return cls(**dict(payload))


class LineageStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lineage.jsonl"
        self.active_path = self.root / "active_version.json"

    def _read(self) -> list[VersionSnapshot]:
        if not self.path.exists():
            return []
        rows: list[VersionSnapshot] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(VersionSnapshot.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return rows

    def record(
        self,
        surfaces: Mapping[str, Any],
        *,
        parent_version: Optional[str],
        candidate_id: Optional[str],
        status: str = "active",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VersionSnapshot:
        version = VersionSnapshot(
            version_id=f"v-{uuid4().hex[:16]}",
            parent_version=parent_version,
            surfaces=dict(surfaces),
            candidate_id=candidate_id,
            status=status,
            metadata=dict(metadata or {}),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(version.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return version

    def get(self, version_id: str) -> Optional[VersionSnapshot]:
        matches = [item for item in self._read() if item.version_id == version_id]
        return matches[-1] if matches else None

    def list(self) -> list[VersionSnapshot]:
        return self._read()

    def set_active(self, version_id: str) -> VersionSnapshot:
        version = self.get(version_id)
        if version is None:
            raise KeyError(version_id)
        self.active_path.write_text(json.dumps({"version_id": version_id}, sort_keys=True), encoding="utf-8")
        return version

    def active(self) -> Optional[VersionSnapshot]:
        if not self.active_path.exists():
            versions = self._read()
            return versions[-1] if versions else None
        try:
            version_id = json.loads(self.active_path.read_text(encoding="utf-8")).get("version_id")
        except (json.JSONDecodeError, OSError):
            return None
        return self.get(str(version_id)) if version_id else None


__all__ = ["LineageStore", "VersionSnapshot"]
