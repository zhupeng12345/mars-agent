import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from agentic.training.train_image_classifier import collect_records, resolve_device, split_records, train_image_classifier


class ImageTrainingTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        for label, value in (("cold", 40), ("hot", 180), ("normal", 100)):
            directory = root / label
            directory.mkdir(parents=True)
            for source in range(4):
                for index in range(32):
                    image = np.full((24, 24, 3), value + index % 3, dtype=np.uint8)
                    cv2.imwrite(str(directory / f"{label}_source{source}_{index}.jpg"), image)
        return root

    def test_group_split_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            records = collect_records(self._dataset(Path(temp)))
            splits = split_records(records, seed=7)
            groups = {name: {item.group_id for item in rows} for name, rows in splits.items()}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])

    def test_cpu_device_is_explicit(self):
        self.assertEqual(str(resolve_device("cpu")), "cpu")

    def test_metadata_source_groups_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._dataset(Path(temp) / "images")
            manifest = Path(temp) / "provenance.jsonl"
            rows = []
            for path in sorted(root.rglob("*.jpg")):
                source = path.stem.split("_")[1]
                rows.append(json.dumps({"path": str(path), "session_id": source}, ensure_ascii=False))
            manifest.write_text("\n".join(rows), encoding="utf-8")
            records = collect_records(root, metadata_path=manifest)
            self.assertTrue(all(item.provenance_status == "metadata" for item in records))
            splits = split_records(records, seed=7)
            groups = {name: {item.group_id for item in rows} for name, rows in splits.items()}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])

    def test_cuda_request_fails_cleanly_when_unavailable(self):
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                resolve_device("cuda")

    def test_training_writes_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._dataset(Path(temp) / "images")
            output = Path(temp) / "out"
            report = train_image_classifier(root, output, epochs=1, batch_size=4, image_size=32)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(Path(report["checkpoint"]).exists())
            self.assertTrue((output / "image_training_report.json").exists())
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["provenance"]["leakage_risk"], "high")


if __name__ == "__main__":
    unittest.main()
