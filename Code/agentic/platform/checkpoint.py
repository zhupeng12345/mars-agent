"""Checkpoint factory for native LangGraph and local service persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional


def build_sqlite_checkpointer(path: Path | str) -> Optional[Any]:
    """Return LangGraph's native SQLite saver when its optional package exists.

    The base install keeps the dependency optional.  The JSON/stdlib SQLite
    stores in ``memory.py`` remain the service-level fallback; callers can
    inspect the return value and report the provider explicitly.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    except ImportError:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteSaver(connection)
    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()
    return saver


__all__ = ["build_sqlite_checkpointer"]
