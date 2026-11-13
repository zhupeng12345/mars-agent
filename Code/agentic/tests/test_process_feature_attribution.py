from __future__ import annotations

import unittest

import numpy as np

from agentic.agents.process_state import DATASET_FEATURE_COLUMNS, PROCESS_VARIABLE_GROUPS
from agentic.evaluation.process_feature_attribution import permutation_importance, select_top_features


class ProcessFeatureAttributionTests(unittest.TestCase):
    def _synthetic(self):
        rng = np.random.default_rng(44)
        labels = np.asarray([0, 1] * 100, dtype=int)
        windows = rng.normal(0.0, 0.15, size=(len(labels), 30, 14)).astype(np.float32)
        windows[:, :, 0] += labels[:, None] * 2.0
        return windows, labels

    def test_signal_permutation_reduces_score_more_than_irrelevant_feature(self):
        windows, labels = self._synthetic()
        original = windows.copy()

        def predictor(batch):
            logits = batch[:, :, 0].mean(axis=1)
            return 1.0 / (1.0 + np.exp(-logits))

        report = permutation_importance(
            predictor,
            windows,
            labels,
            DATASET_FEATURE_COLUMNS,
            repetitions=4,
            seed=3,
        )
        rows = {row["feature"]: row for row in report["feature_importance"]}
        self.assertGreater(rows["settingVolt"]["pr_auc_drop_mean"], rows["potCurrent"]["pr_auc_drop_mean"] + 0.2)
        self.assertEqual(windows.tobytes(), original.tobytes())
        self.assertEqual(len(report["group_importance"]), 4)
        self.assertGreater(report["group_importance"][0]["pr_auc_drop_mean"], 0.2)
        self.assertEqual(select_top_features(report, top_k=8)[0], "settingVolt")

    def test_validation_attribution_is_deterministic_and_groups_cover_all_fields(self):
        windows, labels = self._synthetic()
        predictor = lambda batch: 1.0 / (1.0 + np.exp(-batch[:, :, 0].mean(axis=1)))
        first = permutation_importance(predictor, windows, labels, DATASET_FEATURE_COLUMNS, repetitions=3, seed=12)
        second = permutation_importance(predictor, windows, labels, DATASET_FEATURE_COLUMNS, repetitions=3, seed=12)
        self.assertEqual(first["feature_importance"], second["feature_importance"])
        self.assertEqual(first["group_importance"], second["group_importance"])
        flattened = [name for members in PROCESS_VARIABLE_GROUPS.values() for name in members]
        self.assertEqual(len(flattened), 14)
        self.assertEqual(set(flattened), set(DATASET_FEATURE_COLUMNS))

    def test_bad_column_mapping_and_probabilities_are_rejected(self):
        windows, labels = self._synthetic()
        with self.assertRaisesRegex(ValueError, "identify each of the 14"):
            bad_columns = DATASET_FEATURE_COLUMNS[:-1] + (DATASET_FEATURE_COLUMNS[0],)
            permutation_importance(lambda batch: np.full(len(batch), 0.5), windows, labels, bad_columns)
        with self.assertRaisesRegex(ValueError, r"probabilities must be in \[0,1\]"):
            permutation_importance(lambda batch: np.full(len(batch), 1.5), windows, labels, DATASET_FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
