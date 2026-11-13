import tempfile
import unittest
from pathlib import Path

from agentic.scripts.build_ppt_asset_manifest import build_manifest


class P7PptManifestTests(unittest.TestCase):
    def test_manifest_always_contains_25_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime_data").mkdir()
            (root / "runtime_data" / "demo_test.json").write_text("{}", encoding="utf-8")
            payload = build_manifest(root, root / "report_assets" / "multi_agent_v2" / "ppt_asset_manifest.json")
        self.assertEqual(payload["page_count"], 25)
        self.assertEqual(payload["runtime_demo_count"], 1)
        self.assertEqual(payload["pages"][0]["page"], 1)
        self.assertEqual(payload["pages"][-1]["page"], 25)

    def test_structure_assets_are_attached_to_every_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime_data"
            assets = root / "report_assets" / "multi_agent_v2"
            runtime.mkdir()
            assets.mkdir(parents=True)
            (runtime / "demo_test.json").write_text("{}", encoding="utf-8")
            for page in range(1, 26):
                image = assets / f"asset-page-{page:02d}.png"
                image.write_bytes(b"placeholder")
                image.with_suffix(".json").write_text("{}", encoding="utf-8")
            payload = build_manifest(root, assets / "ppt_asset_manifest.json")
        self.assertEqual(payload["image_asset_count"], 25)
        self.assertTrue(all(row["image_assets"] for row in payload["pages"]))
        self.assertTrue(all(row["source_files"] for row in payload["pages"]))


if __name__ == "__main__":
    unittest.main()
