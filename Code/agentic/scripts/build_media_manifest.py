"""Build JSONL media manifests from files or directories.

Examples::

    python -m agentic.scripts.build_media_manifest data/video --output runtime_data/media.jsonl
    python -m agentic.scripts.build_media_manifest image.jpg --output media.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from agentic.media.normalize import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaNormalizer, manifests_to_jsonl


def _expand_inputs(values: Iterable[str], *, recursive: bool) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            direct_media = sorted(
                item for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            )
            # ``data/video`` is a collection of video files, whereas a folder
            # containing image frames is one frame-directory media item.
            if any(item.suffix.lower() in VIDEO_EXTENSIONS for item in direct_media):
                paths.extend(direct_media)
            elif recursive:
                paths.append(path)
                paths.extend(sorted(
                    item for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
                ))
            else:
                paths.append(path)
        else:
            paths.append(path)
    return paths


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize image, video and frame-directory media")
    parser.add_argument("inputs", nargs="+", help="media paths")
    parser.add_argument("--output", required=True, help="UTF-8 JSONL output path")
    parser.add_argument("--recursive", action="store_true", help="include child image/video files under input directories")
    parser.add_argument("--run-id", default="", help="optional audit run identifier")
    parser.add_argument("--thread-id", default="", help="optional audit thread identifier")
    args = parser.parse_args(argv)
    normalizer = MediaNormalizer()
    manifests = [
        normalizer.normalize(path, run_id=args.run_id, thread_id=args.thread_id)
        for path in _expand_inputs(args.inputs, recursive=args.recursive)
    ]
    manifests_to_jsonl(manifests, args.output)
    print(f"wrote {len(manifests)} manifests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
