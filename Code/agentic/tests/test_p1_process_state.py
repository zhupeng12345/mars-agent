"""Focused tests for the independent Process State Agent and audit tools."""

from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentic.agents.process_state import (
    DATASET_FEATURE_COLUMNS,
    PROCESS_VARIABLE_GROUPS,
    PROCESS_VARIABLES,
    ProcessStateAgent,
)
from agentic.evaluation.process_state import _read_prediction, evaluate_baselines, evaluate_process_state
from agentic.process.baselines import build_process_baseline
from agentic.process.inference import ProcessCheckpointPredictor
from agentic.scripts.build_process_manifest import build_process_manifest
from agentic.training.train_process import train_process


class ProcessStateTests(unittest.TestCase):
    @staticmethod
    def _write_training_fixture(root: Path) -> tuple[Path, Path]:
        dataset_dir = root / "step2_ew_202111"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        timestamps = np.asarray([
            "2021-11-01T22:00", "2021-11-01T23:58",
            "2021-11-02T00:00", "2021-11-02T11:00",
            "2021-11-03T12:00", "2021-11-03T14:00",
        ], dtype="datetime64[m]")
        days = timestamps.astype("datetime64[D]").astype(str)
        values = np.asarray([0.0, 10.0, 100.0, 200.0, 1000.0, 2000.0], dtype=np.float32)
        x = np.broadcast_to(values[:, None, None], (6, 30, 14)).copy()
        labels = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int64)
        np.savez(
            dataset_dir / "dataset.npz",
            X_seq=x,
            X_stat=np.zeros((6, 70), dtype=np.float32),
            y=labels,
            ts_anchor=timestamps,
            day_anchor=days,
            feature_cols=np.asarray(DATASET_FEATURE_COLUMNS),
            L=np.int32(30),
            H=np.int32(15),
            L_MIN=np.int32(60),
            H_MIN=np.int32(30),
            SAMPLE_MINUTES=np.int32(2),
        )
        split_path = dataset_dir / "split.json"
        split_path.write_text(json.dumps({"train": [0, 1], "val": [2, 3], "test": [4, 5]}), encoding="utf-8")
        return dataset_dir / "dataset.npz", split_path

    @staticmethod
    def _write_prediction(root: Path, indices: list[int], labels: list[int]) -> None:
        baseline_dir = root / "step3_baseline_202111" / "test_model"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        path = baseline_dir / "test_pred.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_idx", "y_true", "y_prob", "anchor_ts"))
            writer.writeheader()
            for index, label in zip(indices, labels):
                ts = (np.datetime64("2021-11-01T22:00", "m") + np.timedelta64(index, "D")).astype("datetime64[s]")
                writer.writerow({"sample_idx": index, "y_true": label, "y_prob": 0.7 if label else 0.3, "anchor_ts": str(ts)})

    def test_train_process_purges_boundary_and_fits_normalizer_on_effective_train_only(self) -> None:
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split = self._write_training_fixture(root)
            report = train_process(dataset, split, root / "run", epochs=1, batch_size=2, seed=3, device="cpu")
            self.assertEqual(report["train_count"], 1)
            self.assertEqual(report["validation_count"], 2)
            self.assertEqual(report["test_count"], 2)
            self.assertEqual(report["split"]["purged_counts"]["train"], 1)
            self.assertEqual(report["split"]["normalization_fit_split"], "train")
            self.assertTrue(report["split"]["dates_disjoint"])
            self.assertFalse(report["split"]["support_overlap"])
            self.assertEqual(report["normalization"]["fitted_on"], "train_only")
            self.assertTrue(np.allclose(report["normalization"]["mean"], np.zeros(14)))
            self.assertIn("event_recall", report["metrics"])
            self.assertIn("test_predictions", report)
            prediction_rows = list(csv.DictReader(Path(report["test_predictions"]).open(encoding="utf-8", newline="")))
            self.assertEqual(len(prediction_rows), report["test_count"])
            self.assertEqual({int(row["sample_idx"]) for row in prediction_rows}, set(checkpoint_index for checkpoint_index in [4, 5]))
            self.assertTrue(all("T" in row["anchor_ts"] for row in prediction_rows))
            checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["schema_version"], "process-checkpoint-2")
            self.assertEqual(checkpoint["model_config"]["input_dim"], 14)
            self.assertEqual(checkpoint["effective_split_indices"]["train"], [0])
            self.assertIn("normalization", checkpoint)
            predictor = ProcessCheckpointPredictor(report["checkpoint"])
            prediction = predictor.predict(np.zeros((30, 14), dtype=np.float32), feature_columns=DATASET_FEATURE_COLUMNS)
            self.assertEqual(prediction["status"], "completed")
            self.assertEqual(prediction["normalization_fit_split"], "train")

    def test_train_process_supports_named_feature_subsets_without_test_evaluation(self) -> None:
        compact = (
            "potVolt", "potCurrent", "fluctDelta", "wavingDelta",
            "slopeData", "sumSlopeData", "actualFeedingInter", "feedingState",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split = self._write_training_fixture(root)
            report = train_process(
                dataset,
                split,
                root / "compact",
                model_name="cnn",
                epochs=1,
                batch_size=2,
                seed=3,
                device="cpu",
                feature_names=compact,
                evaluate=False,
            )
            self.assertIsNone(report["metrics"])
            self.assertIsNone(report["evaluation_split"])
            predictor = ProcessCheckpointPredictor(report["checkpoint"])
            self.assertEqual(predictor.feature_columns, compact)
            prediction = predictor.predict(np.zeros((30, 8), dtype=np.float32), feature_columns=compact)
            self.assertEqual(prediction["status"], "completed")

    def test_baseline_evaluation_rejects_sample_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_training_fixture(root)
            self._write_prediction(root, [3, 5], [0, 0])
            with self.assertRaisesRegex(ValueError, "sample-index mismatch"):
                evaluate_baselines(root)

    def test_baseline_evaluation_rejects_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_training_fixture(root)
            self._write_prediction(root, [4, 5], [0, 1])
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                evaluate_baselines(root)

    def test_prediction_reader_rejects_fractional_sample_index_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            path.write_text("sample_idx,y_true,y_prob\n1.5,0,0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample indices must be finite integers"):
                _read_prediction(path)
            path.write_text("sample_idx,y_true,y_prob\n1,0.5,0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labels must be binary"):
                _read_prediction(path)

    def test_trainable_baseline_contracts_accept_14_variable_windows(self) -> None:
        import torch
        value = torch.randn(2, 30, 14)
        for name in ("cnn", "lstm", "tcn"):
            with self.subTest(name=name):
                logits = build_process_baseline(name)(value)
                self.assertEqual(tuple(logits.shape), (2,))

    def test_groups_cover_fourteen_recorded_variables(self) -> None:
        names = [name for group in PROCESS_VARIABLE_GROUPS.values() for name in group]
        self.assertEqual(len(names), 14)
        self.assertEqual(set(names), set(PROCESS_VARIABLES))
        self.assertEqual(len(DATASET_FEATURE_COLUMNS), 14)

    def test_snapshot_extracts_window_features_and_unknown_alignment(self) -> None:
        x = np.zeros((30, 14), dtype=float)
        x[:, 0] = np.linspace(4025, 4050, 30)
        x[:, 1] = 4000 + np.sin(np.linspace(0, 8, 30)) * 30
        x[15:, 6] += 40
        timestamps = [f"2021-11-01 00:{index * 2:02d}:00" for index in range(30)]
        snapshot = ProcessStateAgent().run(x, timestamps=timestamps, run_id="run-process-test")
        self.assertEqual(snapshot.alignment_quality, "unknown")
        self.assertIn("video_mapping", snapshot.process_missing)
        self.assertIn(snapshot.process_regime, {"stable", "transition", "abnormal"})
        self.assertGreaterEqual(snapshot.event_risk or 0.0, 0.0)
        self.assertLessEqual(snapshot.event_risk or 0.0, 1.0)
        self.assertEqual(set(snapshot.variables["by_name"]), set(DATASET_FEATURE_COLUMNS))
        self.assertTrue(snapshot.change_points)
        payload = snapshot.to_dict()
        self.assertEqual(payload["alignment_quality"], "unknown")
        self.assertFalse(payload.get("video_classification_improvement", False))

    def test_missing_window_and_missing_column_are_explicit(self) -> None:
        missing = ProcessStateAgent().run(run_id="run-empty")
        self.assertIn("process_window", missing.process_missing)
        self.assertIn("video_mapping", missing.process_missing)
        self.assertEqual(missing.alignment_quality, "unknown")
        window = {name: np.ones(30) for name in DATASET_FEATURE_COLUMNS if name != "potVolt"}
        snapshot = ProcessStateAgent().run(window, run_id="run-column-missing")
        self.assertIn("potVolt", snapshot.process_missing)
        self.assertTrue(snapshot.missing_mask["potVolt"])
        self.assertEqual(snapshot.alignment_quality, "unknown")

    def test_metrics_have_process_only_boundary(self) -> None:
        timestamps = [f"2021-11-01T00:{index * 2:02d}:00" for index in range(5)]
        result = evaluate_process_state([0, 1, 1, 0, 1], [0.1, 0.9, 0.8, 0.2, 0.6], timestamps=timestamps)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertIn("event_recall", result)
        self.assertEqual(result["point_precision"], 1.0)
        self.assertEqual(result["point_specificity"], 1.0)
        self.assertEqual(result["confusion_counts"], {"tp": 3, "fp": 0, "fn": 0, "tn": 2})
        self.assertEqual(len(result["event_segments"]), 2)
        self.assertEqual(result["event_segments"][0]["start_index"], 1)
        self.assertIn("not physical event onset", result["lead_time_semantics"])
        self.assertNotIn("video_accuracy", result)

    def test_manifest_audits_fixture_without_inventing_video_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "step2_ew_202111"
            dataset_dir.mkdir()
            x = np.zeros((4, 30, 14), dtype=np.float32)
            stat = np.zeros((4, 70), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.int64)
            ts = np.array(["2021-11-01T00:00", "2021-11-01T00:02", "2021-11-01T00:04", "2021-11-01T00:06"], dtype="datetime64[m]")
            np.savez(
                dataset_dir / "dataset.npz", X_seq=x, X_stat=stat, y=y, ts_anchor=ts,
                day_anchor=np.array(["2021-11-01"] * 4), feature_cols=np.array(DATASET_FEATURE_COLUMNS),
                L=np.int32(30), H=np.int32(15), L_MIN=np.int32(60), H_MIN=np.int32(30),
                SAMPLE_MINUTES=np.int32(2),
            )
            (dataset_dir / "split.json").write_text(json.dumps({"train": [0, 1], "val": [2], "test": [3]}), encoding="utf-8")
            manifest = build_process_manifest(root)
            self.assertEqual(manifest["dataset"]["x_seq_shape"], [4, 30, 14])
            self.assertTrue(manifest["split"]["ordered_without_overlap"])
            self.assertEqual(manifest["video_mapping"]["alignment_quality"], "unknown")
            self.assertEqual(manifest["video_mapping"]["process_missing"], ["video_mapping"])
            self.assertFalse(manifest["claims"]["video_classification_improvement_reported"])


if __name__ == "__main__":
    unittest.main()
