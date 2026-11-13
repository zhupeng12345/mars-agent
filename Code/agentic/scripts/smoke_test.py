"""Dependency-light P0.1 smoke checks for media and model configuration.

The smoke check validates the inputs and fallback decision without invoking a
vision model or making a network request.  Model execution belongs to later
plan stages and should not make the repository baseline non-reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def inspect_input(path: Path) -> dict[str, Any]:
    """Inspect one media path and return a stable, UI-safe result."""

    path = Path(path)
    if not path.exists():
        return {"status": "invalid_input", "reason": "path_not_found", "path": str(path)}
    if not path.is_file():
        return {"status": "invalid_input", "reason": "not_a_file", "path": str(path)}
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        kind = "image"
    elif suffix in VIDEO_SUFFIXES:
        kind = "video"
    else:
        return {
            "status": "invalid_input",
            "reason": "unsupported_media_suffix",
            "path": str(path),
            "suffix": suffix,
        }
    stat = path.stat()
    return {
        "status": "valid_input",
        "kind": kind,
        "path": str(path),
        "suffix": suffix,
        "bytes": stat.st_size,
    }


def model_configuration(simulate_no_model: bool = False) -> dict[str, Any]:
    """Report the selected local/API fallback without contacting a provider."""

    if simulate_no_model:
        return {
            "status": "configuration_error",
            "reason": "no_vision_model_configured",
            "fallback_route": "human",
        }
    local_model = os.getenv("MARS_AGENT_MODEL", "qwen3-vl:latest").strip()
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    api_key_env = os.getenv("MARS_LLM_API_KEY_ENV", "DASHSCOPE_API_KEY").strip()
    return {
        "status": "local_profile_available",
        "provider": "ollama",
        "model": local_model,
        "base_url": ollama_url,
        "cloud_api_key_present": bool(os.getenv(api_key_env, "").strip()),
        "cloud_api_key_env": api_key_env,
        "network_checked": False,
        "fallback_route": "human",
    }


def run_smoke(image: Path, video: Path, invalid: Path, simulate_no_model: bool = False) -> dict[str, Any]:
    results = {
        "image": inspect_input(image),
        "video": inspect_input(video),
        "invalid": inspect_input(invalid),
        "model_configuration": model_configuration(simulate_no_model=simulate_no_model),
    }
    checks = {
        "image_valid": results["image"].get("status") == "valid_input"
        and results["image"].get("kind") == "image",
        "video_valid": results["video"].get("status") == "valid_input"
        and results["video"].get("kind") == "video",
        "invalid_rejected": results["invalid"].get("status") == "invalid_input",
        "fallback_visible": results["model_configuration"].get("fallback_route") == "human",
    }
    results["checks"] = checks
    results["passed"] = all(checks.values())
    return results


def _default_media(code_root: Path) -> tuple[Path, Path]:
    image_root = code_root / "data" / "火眼分类图片帧"
    images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    videos = sorted(path for path in (code_root / "data" / "video").glob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    image = images[0] if images else image_root / "missing.jpg"
    video = videos[0] if videos else code_root / "data" / "video" / "missing.mp4"
    return image, video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run P0.1 media/config smoke checks without model inference.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="Code/ directory.")
    parser.add_argument("--image", type=Path, help="A valid image to inspect.")
    parser.add_argument("--video", type=Path, help="A valid video to inspect.")
    parser.add_argument("--invalid", type=Path, default=Path("__mars_agent_missing_input__.mp4"))
    parser.add_argument("--simulate-no-model", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    default_image, default_video = _default_media(args.root)
    payload = run_smoke(
        args.image or default_image,
        args.video or default_video,
        args.invalid,
        simulate_no_model=args.simulate_no_model,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
