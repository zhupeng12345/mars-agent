from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.scripts.inventory import build_inventory
from agentic.scripts.smoke_test import inspect_input, model_configuration, run_smoke
from agentic.workflow import _finalize


class P01BaselineTests(unittest.TestCase):
    def test_inventory_is_json_safe_and_groups_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "video").mkdir(parents=True)
            (root / "data" / "video" / "sample.mp4").write_bytes(b"video")
            payload = build_inventory(root)
        self.assertEqual(payload["schema_version"], "asset-inventory-1")
        self.assertEqual(payload["groups"]["visual_videos"]["file_count"], 1)
        self.assertEqual(payload["groups"]["visual_videos"]["by_suffix"], {".mp4": 1})

    def test_invalid_input_is_structured(self) -> None:
        result = inspect_input(Path("does-not-exist.jpg"))
        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["reason"], "path_not_found")

    def test_no_model_path_is_visible_and_safe(self) -> None:
        result = model_configuration(simulate_no_model=True)
        self.assertEqual(result["status"], "configuration_error")
        self.assertEqual(result["fallback_route"], "human")

    def test_smoke_checks_all_four_baseline_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            video = root / "sample.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            result = run_smoke(image, video, root / "missing.mp4", simulate_no_model=True)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_legacy_result_is_explicitly_excluded_from_modern_claims(self) -> None:
        result = _finalize({
            "input_path": "sample.jpg",
            "visual": {"has_target": True},
            "review": {"approved": True, "issues": []},
            "diagnosis": {},
            "knowledge": [],
        })["result"]
        self.assertEqual(result["flow_version"], "legacy-single-frame")
        self.assertFalse(result["claims"]["modern_multiframe_flow"])
        self.assertFalse(result["claims"]["jev_route_used"])
        self.assertFalse(result["claims"]["rsi_runtime_used"])


if __name__ == "__main__":
    unittest.main()
