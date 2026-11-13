"""Session memory helpers and a small durable checkpoint store.

The online graph can use LangGraph's checkpointers when that optional package
is installed.  The HITL service also needs to work in the repository's local
minimal environment, so :class:`JsonCheckpointStore` provides a dependency
free, atomic file-backed checkpoint with the same thread-oriented semantics.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional

from .contracts import jsonable

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover - keeps contracts importable without LangGraph
    MemorySaver = None  # type: ignore[assignment,misc]


_savers: Dict[str, Any] = {}
_lock = Lock()


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout: float = 30.0):
    """Small cross-process lock for the JSON fallback on Windows and POSIX."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > timeout * 4:
                    lock_path.unlink()
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring checkpoint lock: {lock_path}")
            time.sleep(0.02)
    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


class JsonCheckpointStore:
    """Durable thread checkpoints stored in one atomically replaced JSON file.

    Checkpoints contain only JSON-safe, observable state.  Writes use a temp
    file in the same directory followed by ``os.replace`` so a process crash
    cannot leave a partially written checkpoint.  The in-process lock also
    makes compare-and-write operations safe for concurrent review callbacks.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        self._file_lock = Lock()

    def _read(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {str(key): item for key, item in value.items() if isinstance(item, dict)}

    def _write(self, records: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(jsonable(records), handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._file_lock, _exclusive_file_lock(self.path):
            value = self._read().get(str(thread_id))
            return dict(value) if value is not None else None

    def put(self, thread_id: str, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if not thread_id:
            raise ValueError("thread_id is required")
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must be a mapping")
        with self._file_lock, _exclusive_file_lock(self.path):
            records = self._read()
            records[str(thread_id)] = dict(checkpoint)
            self._write(records)
        return dict(checkpoint)

    def update_if(self, thread_id: str, expected_status: Optional[str], checkpoint: Dict[str, Any]) -> bool:
        """Atomically update a row when its current status matches.

        ``expected_status=None`` means the row must not already exist.  This
        is used by resume to guarantee that only one concurrent reviewer can
        transition a pending checkpoint.
        """
        if not thread_id:
            raise ValueError("thread_id is required")
        with self._file_lock, _exclusive_file_lock(self.path):
            records = self._read()
            current = records.get(str(thread_id))
            current_status = current.get("status") if current else None
            if current_status != expected_status:
                return False
            records[str(thread_id)] = dict(checkpoint)
            self._write(records)
            return True

    def list(self, *, status: Optional[str] = None) -> list[Dict[str, Any]]:
        with self._file_lock, _exclusive_file_lock(self.path):
            rows = list(self._read().values())
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        return [dict(row) for row in rows]

    def delete(self, thread_id: str) -> bool:
        with self._file_lock, _exclusive_file_lock(self.path):
            records = self._read()
            if str(thread_id) not in records:
                return False
            records.pop(str(thread_id), None)
            self._write(records)
            return True


class SQLiteCheckpointStore:
    """Standard-library SQLite checkpoint store for multi-process safety.

    This is intentionally independent from ``langgraph-checkpoint-sqlite``;
    callers can use it when they want a transactional local backend while the
    JSON implementation remains the zero-dependency default.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @staticmethod
    def _decode(row: Any) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, dict) else None

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            return self._decode(connection.execute("SELECT payload FROM checkpoints WHERE thread_id=?", (str(thread_id),)).fetchone())

    def put(self, thread_id: str, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(jsonable(checkpoint), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute("INSERT INTO checkpoints(thread_id,payload) VALUES(?,?) ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload", (str(thread_id), payload))
        return dict(checkpoint)

    def update_if(self, thread_id: str, expected_status: Optional[str], checkpoint: Dict[str, Any]) -> bool:
        payload = json.dumps(jsonable(checkpoint), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM checkpoints WHERE thread_id=?", (str(thread_id),)).fetchone()
            current = self._decode(row)
            status = current.get("status") if current else None
            if status != expected_status:
                connection.execute("ROLLBACK")
                return False
            connection.execute("INSERT INTO checkpoints(thread_id,payload) VALUES(?,?) ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload", (str(thread_id), payload))
            connection.execute("COMMIT")
            return True

    def list(self, *, status: Optional[str] = None) -> list[Dict[str, Any]]:
        query = "SELECT payload FROM checkpoints"
        values: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE json_extract(payload, '$.status')=?"
            values = (status,)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [decoded for row in rows if (decoded := self._decode(row)) is not None]

    def delete(self, thread_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM checkpoints WHERE thread_id=?", (str(thread_id),))
            return cursor.rowcount > 0


# Names used by integrations that do not need to know which local file format
# is selected.  JSON remains the dependency-free default.
PersistentCheckpointStore = JsonCheckpointStore
LocalCheckpointStore = JsonCheckpointStore


def get_memory_saver(agent_id: str = "inspection"):
    """Return an isolated in-process checkpointer for one agent role."""

    if MemorySaver is None:
        raise RuntimeError("LangGraph is required for session memory")
    with _lock:
        if agent_id not in _savers:
            _savers[agent_id] = MemorySaver()
        return _savers[agent_id]


def build_thread_config(
    tenant_id: str,
    session_id: str,
    run_id: str,
) -> Dict[str, Dict[str, str]]:
    """Build the same configurable thread shape used by LangGraph agents."""

    return {
        "configurable": {
            "thread_id": f"inspection_{tenant_id}_{session_id}_{run_id}",
        }
    }
