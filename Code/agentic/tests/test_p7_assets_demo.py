import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from agentic.scripts.validate_assets import validate_assets


class P7AssetTests(unittest.TestCase):
    @staticmethod
    def _write_asset(root: Path, *, width: int = 12, height: int = 8, sidecar_width: int | None = None):
        image_path = root / "asset-a.png"
        Image.new("RGB", (width, height), color=(20, 80, 160)).save(image_path)
        metadata = {
            "asset_id": image_path.stem,
            "title": "test asset",
            "source_files": ["sample.mp4"],
            "run_id": "run-test",
            "model_profile": "test",
            "knowledge_version": "v1",
            "generated_at": "2026-11-11T00:00:00Z",
            "width": sidecar_width if sidecar_width is not None else width,
            "height": height,
            "status": "flow_validation",
            "script_version": "test",
        }
        image_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
        return image_path

    def test_empty_asset_directory_is_explicitly_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_assets(Path(directory))
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["claims"]["empty_directory_is_pending"])

    def test_sidecar_schema_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset-a.png").write_bytes(b"fake")
            report = validate_assets(root)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["valid_count"], 0)

    def test_valid_png_decodes_and_matches_sidecar_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = self._write_asset(root)
            report = validate_assets(root)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["valid_count"], 1)
        self.assertEqual(report["rows"][0]["actual_dimensions"], [12, 8])
        self.assertNotIn("image_decode_failed", report["rows"][0]["errors"])

    def test_corrupt_raster_is_reported_as_decode_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "asset-a.png"
            image_path.write_bytes(b"not a png")
            image_path.with_suffix(".json").write_text("{}", encoding="utf-8")
            report = validate_assets(root)
        self.assertEqual(report["status"], "fail")
        self.assertIn("image_decode_failed", report["rows"][0]["errors"])

    def test_sidecar_dimension_must_match_decoded_raster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_asset(root, width=12, height=8, sidecar_width=13)
            report = validate_assets(root)
        self.assertEqual(report["status"], "fail")
        self.assertIn("dimension_mismatch", report["rows"][0]["errors"])
        self.assertEqual(report["rows"][0]["actual_dimensions"], [12, 8])

    def test_manifest_image_reference_must_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = self._write_asset(root)
            manifest = {
                "schema_version": "ppt-asset-manifest-1",
                "page_count": 1,
                "image_asset_count": 1,
                "pages": [{"page": 1, "image_assets": [str(image_path), str(root / "missing.png")]}],
            }
            (root / "ppt_asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = validate_assets(root)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["manifest"]["resolved_reference_count"], 1)
        self.assertEqual(report["manifest"]["unresolved_references"][0]["reason"], "file_missing")


if __name__ == "__main__":
    unittest.main()
