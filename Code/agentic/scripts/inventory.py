"""Create a reproducible inventory of the local MARS-Agent assets.

This module intentionally uses only the Python standard library.  It must be
usable before the optional vision and LangGraph dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CODE_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield files below *root* in stable relative-path order."""

    if not root.exists():
        return
    paths = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            paths.append(path)
    yield from sorted(paths, key=lambda item: item.relative_to(root).as_posix().lower())


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_record(path: Path, code_root: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.relative_to(code_root).as_posix(),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = _sha256(path)
    return record


def _group_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    by_suffix: dict[str, int] = {}
    total_bytes = 0
    for item in files:
        suffix = item["suffix"] or "<no_suffix>"
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        total_bytes += int(item["bytes"])
    return {"file_count": len(files), "total_bytes": total_bytes, "by_suffix": dict(sorted(by_suffix.items()))}


def build_inventory(code_root: Path = DEFAULT_CODE_ROOT, include_hash: bool = False) -> dict[str, Any]:
    """Return a JSON-safe inventory for the repository's research assets."""

    code_root = Path(code_root).resolve()
    roots = {
        "visual_videos": code_root / "data" / "video",
        "fire_eye_images": code_root / "data" / "火眼分类图片帧",
        "process_data_v1": code_root / "data" / "v1",
        "knowledge": code_root / "agentic" / "knowledge",
        "agentic_code": code_root / "agentic",
        "model_code": code_root / "model",
        "weights": code_root / "weights",
        "report_assets": code_root / "report_assets",
    }
    groups: dict[str, Any] = {}
    all_files: list[dict[str, Any]] = []
    for name, root in roots.items():
        files = [_asset_record(path, code_root, include_hash) for path in _iter_files(root)]
        groups[name] = {"root": root.relative_to(code_root).as_posix(), **_group_summary(files), "files": files}
        all_files.extend(files)

    return {
        "schema_version": "asset-inventory-1",
        "generated_at": _utc_now(),
        "project_root": str(code_root),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "hashes_included": include_hash,
        "summary": _group_summary(all_files),
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory MARS-Agent data, code, knowledge and report assets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_CODE_ROOT, help="Code/ directory to scan.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument("--hash", action="store_true", dest="include_hash", help="Include SHA-256 for every file.")
    args = parser.parse_args(argv)
    payload = build_inventory(args.root, include_hash=args.include_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
