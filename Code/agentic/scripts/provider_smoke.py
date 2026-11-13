"""Run structured provider checks without storing private model reasoning."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic.agents.review_guard import review_draft
from agentic.platform.contracts import DiagnosisDraftPayload
from agentic.platform.llm_gateway import LLMGateway


def _runtime_dir(root: Path) -> Path:
    return Path(os.environ.get("MARS_AGENT_RUNTIME_DIR", str(root / "runtime_data"))).expanduser()


def _default_image(root: Path) -> Path:
    manifests = root / "data" / "evaluation" / "visual_manifest.jsonl"
    if manifests.exists():
        for line in manifests.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            paths = row.get("roi_paths", [])
            if row.get("source") == "video" and paths:
                path = Path(paths[0])
                if not path.is_absolute():
                    candidates = (root / path, root.parent / path)
                else:
                    candidates = (path,)
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
    raise FileNotFoundError("no video ROI frame is available in visual_manifest.jsonl")


def run_provider_smoke(root: Path, image: Path, output: Path, profiles: tuple[str, ...] = ("ollama_local", "qwen_api")) -> dict[str, Any]:
    image = image.resolve()
    source_id = f"provider-smoke-{image.stem}"
    run_id = f"provider-smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    thread_id = f"inspection_provider_{run_id}"
    message = {
        "role": "user",
        "content": (
            "Return a small JSON diagnosis for the attached aluminum electrolysis fire-eye ROI. "
            "Describe visible evidence only; do not infer temperature, chemistry, process values, or control actions. "
            f"Use source_ids=[\"{source_id}\"], risk_level=unknown, confidence between 0 and 1, and requires_human_review=true. "
            "Include diagnosis, diagnosis_class, summary, findings, recommendations, inspection_items, open_questions, "
            "and requires_human_review=true."
        ),
    }
    gateway = LLMGateway()
    results: dict[str, Any] = {}
    for profile in profiles:
        prompt = dict(message)
        prompt["content"] = str(prompt["content"]).replace("{profile}", profile)
        images = [str(image)] if profile != "fake_json" else []
        response = gateway.chat(
            profile,
            [prompt],
            images=images,
            response_schema=DiagnosisDraftPayload,
            trace_context={"run_id": run_id, "thread_id": thread_id, "agent_id": "provider_smoke"},
        )
        review = review_draft(response.content, known_source_ids={source_id}) if response.status == "completed" and isinstance(response.content, dict) else None
        results[profile] = {
            "status": response.status,
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "latency_ms": response.latency_ms,
            "retries": response.retries,
            "usage": response.usage,
            "error_type": response.error_type,
            "error": response.error,
            "content": response.content,
            "review": review,
        }
    report = {
        "schema_version": "provider-smoke-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "thread_id": thread_id,
        "image": str(image),
        "source_id": source_id,
        "results": results,
        "raw_model_reasoning_stored": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--profiles", nargs="+", default=["ollama_local", "qwen_api"])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    image = args.image or _default_image(root)
    output = args.output or (_runtime_dir(root) / "provider_smoke.json")
    report = run_provider_smoke(root, image, output, tuple(args.profiles))
    print(json.dumps({"output": str(output), "results": {key: {"status": item["status"], "provider": item["provider"], "model": item["model"], "latency_ms": item["latency_ms"], "review_action": (item.get("review") or {}).get("action")} for key, item in report["results"].items()}}, ensure_ascii=False))
    return 0 if any(item["status"] == "completed" for item in report["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
