import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentic.evaluation.visual_eval import classification_metrics, evaluate_visual_manifest
from agentic.models.ood import EmbeddingNoveltyScorer, energy_novelty
from agentic.training.train_visual import run_smoke_training, train_manifest


class VisualEvalTrainingTests(unittest.TestCase):
    def test_metrics_and_unlabeled_boundary(self):
        metrics = classification_metrics(["C", "H", "N"], [{"C": .8}, {"H": .8}, {"N": .8}])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(evaluate_visual_manifest([{"window_id": "x", "human_label": None}])["metric_status"], "structure_validation_only")

    def test_labeled_manifest_without_complete_predictions_does_not_invent_metrics(self):
        rows = [{"window_id": "window-1", "human_label": "H"}, {"window_id": "window-2", "human_label": "N"}]
        missing = evaluate_visual_manifest(rows)
        self.assertEqual(missing["metric_status"], "prediction_missing")
        self.assertNotIn("accuracy", missing)
        self.assertEqual(missing["missing_prediction_count"], 2)

        partial = evaluate_visual_manifest(rows, predictions={"window-1": {"C": 0.1, "H": 0.8, "N": 0.1}})
        self.assertEqual(partial["metric_status"], "prediction_missing")
        self.assertEqual(partial["missing_prediction_window_ids"], ["window-2"])

    def test_invalid_probability_vectors_cannot_be_scored(self):
        for row in ({"C": 0.0, "H": 0.0, "N": 0.0}, {"C": float("nan"), "H": 0.0, "N": 0.0}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    classification_metrics(["C"], [row])

    def test_unknown_truth_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown visual label"):
            classification_metrics(["unknown"], [{"C": 0.8, "H": 0.1, "N": 0.1}])

    def test_smoke_training_and_novelty(self):
        result = run_smoke_training(steps=1)
        self.assertEqual(result["status"], "smoke_passed")
        scorer = EmbeddingNoveltyScorer().fit(np.zeros((2, 3)))
        self.assertGreaterEqual(scorer.score(np.ones(3)), 0.0)
        self.assertEqual(len(energy_novelty(np.zeros((2, 3)))), 2)

    def test_manifest_training_reports_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "visual.jsonl"
            manifest.write_text('{"human_label": null}\n', encoding="utf-8")
            result = train_manifest(manifest)
        self.assertEqual(result["status"], "blocked_no_human_video_labels")


if __name__ == "__main__":
    unittest.main()
