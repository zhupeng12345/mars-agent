import json
import tempfile
import unittest
from pathlib import Path

from agentic.platform.jev.calibration import fit_route_calibration
from agentic.platform.jev.laya_adapter import LayaAdapter
from agentic.scripts.fit_jev import main as fit_jev_main


def labeled_rows(count=12):
    rows = []
    for index in range(count):
        expert = "simple_loop" if index % 2 == 0 else "full_loop"
        predicted = expert if index < count // 2 else ("full_loop" if expert == "simple_loop" else "simple_loop")
        rows.append({
            "case_id": f"expert-{index:02d}",
            "features": {"visual_confidence": 0.55 + 0.03 * (index % 10)},
            "expert_route": expert,
            "predicted_route": predicted,
        })
    return rows


class JevCalibrationTests(unittest.TestCase):
    def test_route_manifest_nested_schema_and_expert_labels(self):
        result = fit_route_calibration(labeled_rows())
        self.assertEqual(result["row_count"], 12)
        self.assertEqual(result["target"], "observed_route_matches_expert_route")
        self.assertEqual(result["confidence_sources"]["features.visual_confidence_proxy"], 12)
        self.assertEqual(set(result["expert_route_classes"]), {"simple_loop", "full_loop"})

    def test_rejects_insufficient_or_malformed_route_rows(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            fit_route_calibration(labeled_rows(5))

        malformed = labeled_rows()
        del malformed[0]["features"]["visual_confidence"]
        with self.assertRaisesRegex(ValueError, "features.visual_confidence"):
            fit_route_calibration(malformed)

        malformed = labeled_rows()
        malformed[0]["expert_route"] = "normal"
        with self.assertRaisesRegex(ValueError, "expert_route"):
            fit_route_calibration(malformed)

        malformed = labeled_rows()
        del malformed[0]["predicted_route"]
        with self.assertRaisesRegex(ValueError, "predicted_route"):
            fit_route_calibration(malformed)

    def test_rejects_single_outcome_calibration_labels(self):
        rows = labeled_rows()
        for row in rows:
            row["predicted_route"] = row["expert_route"]
        with self.assertRaisesRegex(ValueError, "both correct and incorrect"):
            fit_route_calibration(rows)

    def test_fit_jev_writes_rejection_report_for_fixture_only_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = labeled_rows(5)
            source = root / "route_manifest.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "calibration.json"
            status = fit_jev_main(["--rows", str(source), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(status, 2)
            self.assertEqual(report["status"], "rejected")
            self.assertIn("at least 10", report["errors"][0])

    def test_laya_fit_failure_is_not_silenced(self):
        class BrokenRouter:
            def fit(self, _rows):
                raise ValueError("backend training failure")

        with self.assertRaisesRegex(RuntimeError, "Laya router fitting failed") as raised:
            LayaAdapter(BrokenRouter()).fit(labeled_rows())
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_laya_backend_without_fit_is_explicitly_unsupported(self):
        class PredictOnlyRouter:
            def predict(self, *_args, **_kwargs):
                return {}

        with self.assertRaisesRegex(NotImplementedError, "does not expose fit"):
            LayaAdapter(PredictOnlyRouter()).fit(labeled_rows())


if __name__ == "__main__":
    unittest.main()
