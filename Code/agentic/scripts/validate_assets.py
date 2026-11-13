"""Validate report image sidecars and emit a structured asset status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


REQUIRED = {"asset_id", "title", "source_files", "run_id", "model_profile", "knowledge_version", "generated_at", "width", "height", "status"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _decode_dimensions(path: Path) -> list[int]:
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as decoded:
        return [int(decoded.width), int(decoded.height)]


def validate_assets(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    images = sorted(item for item in directory.glob("*") if item.suffix.lower() in IMAGE_EXTENSIONS) if directory.exists() else []
    rows = []
    for image in images:
        sidecar = image.with_suffix(".json")
        errors = []
        if not sidecar.exists():
            errors.append("missing_sidecar")
            metadata = {}
        else:
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
                errors.append("invalid_sidecar_json")
        if not isinstance(metadata, dict):
            errors.append("sidecar_not_object")
            metadata = {}
        if not (metadata.get("script") or metadata.get("script_version")):
            errors.append("missing_script")
        errors.extend(f"missing_{key}" for key in sorted(REQUIRED - set(metadata)))
        if metadata.get("asset_id") and metadata["asset_id"] != image.stem:
            errors.append("asset_id_mismatch")
        try:
            width, height = int(metadata.get("width", 0)), int(metadata.get("height", 0))
        except (TypeError, ValueError):
            width, height = 0, 0
        if width <= 0 or height <= 0:
            errors.append("invalid_dimensions")
        actual_dimensions = None
        try:
            actual_dimensions = _decode_dimensions(image)
            if actual_dimensions != [width, height]:
                errors.append("dimension_mismatch")
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            errors.append("image_decode_failed")
        rows.append({"path": str(image), "sidecar": str(sidecar), "actual_dimensions": actual_dimensions, "status": "valid" if not errors else "invalid", "errors": errors, "metadata": metadata})
    status = "pass" if rows and all(row["status"] == "valid" for row in rows) else "partial" if not rows else "fail"
    manifest_summary: dict[str, Any] = {"present": False}
    manifest_path = directory / "ppt_asset_manifest.json"
    if manifest_path.exists():
        manifest_summary["present"] = True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest_not_object")
            pages = manifest.get("pages", [])
            if not isinstance(pages, list):
                raise ValueError("manifest_pages_not_list")
            missing_page_images = []
            unresolved_references = []
            resolved_references = []
            for page in pages:
                if not isinstance(page, dict):
                    unresolved_references.append({"page": None, "reference": None, "reason": "page_not_object"})
                    continue
                page_images = page.get("image_assets") or []
                if not page_images:
                    missing_page_images.append(page.get("page"))
                for field in ("image_assets", "supporting_image_assets"):
                    references = page.get(field) or []
                    if not isinstance(references, list):
                        unresolved_references.append({
                            "page": page.get("page"), "field": field,
                            "reference": references, "reason": "references_not_list",
                        })
                        continue
                    for reference in references:
                        if not isinstance(reference, str) or not reference.strip():
                            unresolved_references.append({
                                "page": page.get("page"), "field": field,
                                "reference": reference, "reason": "invalid_reference",
                            })
                            continue
                        path = Path(reference)
                        if not path.is_absolute():
                            path = directory / path
                        path = path.resolve()
                        if path.suffix.lower() not in IMAGE_EXTENSIONS:
                            unresolved_references.append({
                                "page": page.get("page"), "field": field,
                                "reference": reference, "reason": "unsupported_image_extension",
                            })
                        elif not path.is_file():
                            unresolved_references.append({
                                "page": page.get("page"), "field": field,
                                "reference": reference, "reason": "file_missing",
                            })
                        else:
                            try:
                                reference_dimensions = _decode_dimensions(path)
                            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                                unresolved_references.append({
                                    "page": page.get("page"), "field": field,
                                    "reference": reference, "reason": "image_decode_failed",
                                })
                                continue
                            resolved_references.append({
                                "page": page.get("page"), "field": field,
                                "reference": reference,
                                "dimensions": reference_dimensions,
                            })
            manifest_summary.update({
                "page_count": manifest.get("page_count"),
                "image_asset_count": manifest.get("image_asset_count"),
                "pages_with_images": len(pages) - len(missing_page_images),
                "pages_missing_images": missing_page_images,
                "resolved_reference_count": len(resolved_references),
                "unresolved_references": unresolved_references,
                "page_coverage": not missing_page_images and not unresolved_references and len(pages) == 25,
            })
            if unresolved_references:
                status = "fail"
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            manifest_summary.update({"error": str(exc) if str(exc) else "invalid_manifest"})
            status = "fail"
    return {"schema_version": "asset-validation-1", "directory": str(directory), "status": status, "image_count": len(images), "valid_count": sum(row["status"] == "valid" for row in rows), "rows": rows, "manifest": manifest_summary, "claims": {"real_runtime_assets_only": True, "empty_directory_is_pending": not rows}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[2] / "report_assets" / "multi_agent_v2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_assets(args.directory)
    output = args.output or (args.directory / "asset_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "images": report["image_count"], "valid": report["valid_count"]}, ensure_ascii=False))
    return 0 if report["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
