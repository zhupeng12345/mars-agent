"""Fill reproducible dimensions in PPT image sidecars from the image bytes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


def enrich(directory: Path) -> int:
    count = 0
    for image in sorted(directory.glob("*")):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        sidecar = image.with_suffix(".json")
        if not sidecar.exists():
            continue
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        probe = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if probe is None:
            continue
        height, width = probe.shape[:2]
        metadata["width"] = int(width)
        metadata["height"] = int(height)
        metadata.setdefault("script_version", "asset-sidecar-enricher-1")
        metadata["dimensions_verified_at"] = datetime.now(timezone.utc).isoformat()
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[2] / "report_assets" / "multi_agent_v2")
    args = parser.parse_args()
    print(json.dumps({"updated": enrich(args.directory)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
