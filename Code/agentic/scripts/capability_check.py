"""Report optional provider and framework capabilities without network writes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def _module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def capabilities() -> dict[str, Any]:
    result = {
        "schema_version": "capability-report-1",
        "python": _module("sys"),
        "langgraph": _module("langgraph"),
        "laya": _module("laya"),
        "torch": _module("torch"),
        "ollama_client": _module("ollama"),
        "vector_options": {name: _module(name) for name in ("sentence_transformers", "pymilvus", "rank_bm25")},
        "providers": {
            "ollama_local": {"configured": bool(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))},
            "qwen_api": {"configured": bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"))},
        },
    }
    try:
        import torch
        result["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        result["torch"]["device_count"] = int(torch.cuda.device_count())
    except Exception:
        pass
    try:
        importlib.import_module("langgraph.checkpoint.sqlite")
        result["langgraph_sqlite_checkpoint"] = {"available": True}
    except Exception as exc:
        result["langgraph_sqlite_checkpoint"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        result["laya"]["distribution_version"] = importlib.metadata.version("laya")
    except importlib.metadata.PackageNotFoundError:
        pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = capabilities()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "langgraph": report["langgraph"]["available"], "laya": report["laya"]["available"], "cuda": report["torch"].get("cuda_available", False), "sqlite_checkpoint": report["langgraph_sqlite_checkpoint"]["available"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
