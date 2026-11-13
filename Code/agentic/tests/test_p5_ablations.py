import hashlib
import copy
import tempfile
import unittest
from pathlib import Path

from agentic.evaluation.ablation import ABLATION_SPECS, build_ablation_report


class P5AblationTests(unittest.TestCase):
    @staticmethod
    def _valid_override(root: Path, *, metrics=None, provenance_updates=None):
        files = {
            "paired.jsonl": b'{"sample":"a"}\n',
            "baseline.bin": b"baseline artifact",
            "variant.bin": b"variant artifact",
        }
        for name, content in files.items():
            (root / name).write_bytes(content)
        provenance = {
            "data_version": "visual-v1",
            "split": "sealed-test",
            "seed": 17,
            "model": "temporal-v1",
            "generated_at": "2026-11-09T00:00:00Z",
            "paired_sample_count": 1,
            "paired_data_path": "paired.jsonl",
            "paired_data_sha256": hashlib.sha256(files["paired.jsonl"]).hexdigest(),
            "baseline_artifact_path": "baseline.bin",
            "baseline_artifact_sha256": hashlib.sha256(files["baseline.bin"]).hexdigest(),
            "variant_artifact_path": "variant.bin",
            "variant_artifact_sha256": hashlib.sha256(files["variant.bin"]).hexdigest(),
        }
        provenance.update(provenance_updates or {})
        return {"metrics": metrics if metrics is not None else {"macro_f1": 0.75}, "provenance": provenance}

    def test_required_matrix_is_complete_and_does_not_fabricate_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_ablation_report(root)
        self.assertEqual(len(report["comparisons"]), 9)
        self.assertEqual({row["ablation_id"] for row in report["comparisons"]}, {row[0] for row in ABLATION_SPECS})
        for row in report["comparisons"]:
            self.assertIn(row["status"], {"not_run", "structure_only", "completed"})
            self.assertFalse(row["claimable"])
            self.assertEqual(row["metrics"], {})

    def test_numeric_override_requires_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_ablation_report(root, result_overrides={"without_jev_vs_with_jev": {"metrics": {"route_accuracy": 1.0}}})
        row = next(item for item in report["comparisons"] if item["ablation_id"] == "without_jev_vs_with_jev")
        self.assertEqual(row["status"], "structure_only")
        self.assertEqual(row["metrics"], {})

    def test_invalid_numeric_or_incomplete_provenance_stays_structure_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self._valid_override(root)
            invalid_overrides = []
            for metrics in ({}, {"score": float("nan")}, {"score": float("inf")}, {"score": "0.75"}):
                supplied = copy.deepcopy(valid)
                supplied["metrics"] = metrics
                invalid_overrides.append(supplied)
            for key, value in (("data_version", " "), ("paired_sample_count", 0), ("variant_artifact_sha256", "0" * 64)):
                supplied = copy.deepcopy(valid)
                supplied["provenance"][key] = value
                invalid_overrides.append(supplied)
            for supplied in invalid_overrides:
                with self.subTest(supplied=supplied):
                    report = build_ablation_report(
                        root, result_overrides={"without_jev_vs_with_jev": supplied}
                    )
                    row = next(item for item in report["comparisons"] if item["ablation_id"] == "without_jev_vs_with_jev")
                    self.assertEqual(row["status"], "structure_only")
                    self.assertFalse(row["claimable"])
                    self.assertEqual(row["metrics"], {})

    def test_reproducible_paired_override_with_matching_artifact_hashes_is_claimable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supplied = self._valid_override(root)
            report = build_ablation_report(
                root, result_overrides={"without_jev_vs_with_jev": supplied}
            )
        row = next(item for item in report["comparisons"] if item["ablation_id"] == "without_jev_vs_with_jev")
        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["claimable"])
        self.assertEqual(row["metrics"], {"macro_f1": 0.75})


if __name__ == "__main__":
    unittest.main()
