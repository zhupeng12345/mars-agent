from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.evaluation.process_repeats import run_repeated_process_evaluation, summarize_repeated_reports


def report(seed: int, auc: float):
    return {
        "model": "cnn",
        "seed": seed,
        "dataset_sha256": "dataset-hash",
        "split_sha256": "split-hash",
        "test_count": 20,
        "split": {"effective_split_index_sha256": {"test": "test-index-hash"}},
        "checkpoint": f"seed-{seed}.pth",
        "test_predictions": f"seed-{seed}.csv",
        "metrics": {"roc_auc": auc, "pr_auc": auc - 0.05, "event_recall": 0.7, "false_alarm_segment_count": 2, "predicted_positive_rate": 0.5, "point_precision": 0.6, "point_recall": 0.7, "point_f1": 0.64, "point_specificity": 0.4, "brier": 0.2, "calibration": {"ece": 0.1}},
    }


class ProcessRepeatedEvaluationTests(unittest.TestCase):
    def test_summary_reports_seed_variation_with_same_data_and_split(self):
        result = summarize_repeated_reports([report(7, 0.6), report(17, 0.8)])
        self.assertEqual(result["seeds"], [7, 17])
        self.assertEqual(result["test_windows"], 20)
        self.assertAlmostEqual(result["metrics"]["roc_auc"]["mean"], 0.7)
        self.assertAlmostEqual(result["metrics"]["roc_auc"]["std_population"], 0.1)
        self.assertAlmostEqual(result["metrics"]["ece"]["mean"], 0.1)
        self.assertAlmostEqual(result["metrics"]["predicted_positive_rate"]["mean"], 0.5)
        self.assertFalse(result["claims"]["video_process_fusion_evaluated"])

    def test_summary_rejects_changed_dataset_or_test_partition(self):
        changed_dataset = report(17, 0.8) | {"dataset_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "same identified dataset"):
            summarize_repeated_reports([report(7, 0.6), changed_dataset])
        changed_split = report(17, 0.8) | {"split": {"effective_split_index_sha256": {"test": "other"}}}
        with self.assertRaisesRegex(ValueError, "same effective test split"):
            summarize_repeated_reports([report(7, 0.6), changed_split])

    def test_runner_writes_seeded_summary_and_csv(self):
        calls = []

        def fake_train(dataset, split, output_dir, **kwargs):
            calls.append((output_dir, kwargs["seed"]))
            return report(kwargs["seed"], 0.6 + kwargs["seed"] / 1000)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_repeated_process_evaluation(
                root / "data.npz",
                root / "split.json",
                root / "output",
                seeds=(7, 17),
                train_fn=fake_train,
            )
            self.assertTrue((root / "output" / "repeated_process_evaluation.json").exists())
            self.assertTrue((root / "output" / "repeated_process_evaluation_metrics.csv").exists())
        self.assertEqual([seed for _, seed in calls], [7, 17])
        self.assertEqual(result["run_count"], 2)

    def test_seed_list_must_be_unique_and_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unique"):
                run_repeated_process_evaluation(Path("x"), Path("y"), Path(tmp), seeds=(7, 7))
            with self.assertRaisesRegex(ValueError, "nonempty"):
                run_repeated_process_evaluation(Path("x"), Path("y"), Path(tmp), seeds=())


if __name__ == "__main__":
    unittest.main()
