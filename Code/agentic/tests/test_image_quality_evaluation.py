import unittest

from agentic.evaluation.image_quality import evaluate_quality_gate


class ImageQualityEvaluationTests(unittest.TestCase):
    def test_synthetic_failures_are_gated_without_false_retake(self):
        report = evaluate_quality_gate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["evaluation_kind"], "synthetic_quality_gate_regression")
        self.assertFalse(report["claimable_domain_accuracy"])
        self.assertGreaterEqual(report["failure_detection_rate"], 0.9)
        self.assertEqual(report["valid_false_retake_count"], 0)
        self.assertEqual(report["failure_sample_count"], 5)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in report["fixtures"]))

    def test_detection_threshold_is_validated(self):
        with self.assertRaises(ValueError):
            evaluate_quality_gate(min_failure_detection_rate=1.1)


if __name__ == "__main__":
    unittest.main()
