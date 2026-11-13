import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.knowledge import VersionedKnowledgeBase


class KnowledgeCaseTests(unittest.TestCase):
    def test_versioned_search_and_empty_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.jsonl"
            path.write_text(json.dumps({"rule_id":"r1","title":"火眼亮度","source":"manual","source_version":"v1","effective_date":"2026-01-01","preconditions":[],"applicability":["visual"],"risk_level":"low","allowed_output":["observation"],"text":"亮度稳定时记录视觉证据"}, ensure_ascii=False) + "\n", encoding="utf-8")
            kb = VersionedKnowledgeBase(path)
            self.assertEqual(kb.search("亮度", applicability=["visual"])[0]["source_id"], "r1")
            bm25 = kb.search_bm25("亮度", applicability=["visual"])
            self.assertEqual(bm25[0]["source_id"], "r1")
            self.assertEqual(bm25[0]["retrieval_mode"], "versioned_bm25")
            self.assertEqual(kb.search("不存在"), [])
            self.assertEqual(kb.search_bm25("不存在"), [])
            self.assertEqual(kb.get_rule("r1").title, "火眼亮度")
            self.assertIsNone(kb.get_rule("unknown-rule"))
            self.assertIsNone(kb.get_rule("../rules.jsonl"))

    def test_case_lifecycle_requires_review_for_active(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = SpecialCaseMemory(Path(directory))
            row = memory.add_candidate(source_id="media-1", features={"confidence": 0.9}, conclusion={"label":"N"}, run_id="run-1")
            memory.transition(row["case_id"], "eval_only", reviewer_id="expert", reason="evaluate")
            memory.transition(row["case_id"], "approved_special_case", reviewer_id="expert", reason="approved")
            with self.assertRaises(ValueError):
                memory.transition(row["case_id"], "active", reviewer_id="expert", reason="publish", regression_passed=False)
            active = memory.transition(row["case_id"], "active", reviewer_id="expert", reason="publish", regression_passed=True)
            self.assertEqual(active["state"], "active")
            self.assertEqual(len(memory.search({"confidence": 0.9})), 1)

    def test_case_search_rejects_missing_or_low_coverage_features(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = SpecialCaseMemory(Path(directory))
            row = memory.add_candidate(
                source_id="media-2",
                features={"confidence": 0.9},
                conclusion={"label": "N"},
                run_id="run-2",
            )
            memory.transition(row["case_id"], "eval_only", reviewer_id="expert", reason="evaluate")
            memory.transition(row["case_id"], "approved_special_case", reviewer_id="expert", reason="approved")
            memory.transition(row["case_id"], "active", reviewer_id="expert", reason="publish", regression_passed=True)

            self.assertEqual(memory.search({}), [])
            self.assertEqual(memory.search({"risk": 0.1}), [])
            self.assertEqual(
                memory.search({"confidence": 0.9, "risk": 0.1, "conflict": 0.0}),
                [],
            )
            partial = memory.search(
                {"confidence": 0.9, "risk": 0.1, "conflict": 0.0},
                min_feature_coverage=0.3,
            )
            self.assertEqual(len(partial), 1)
            self.assertAlmostEqual(partial[0]["feature_coverage"], 1 / 3)
            self.assertEqual(memory.search({"confidence": float("nan")}), [])

    def test_candidate_conclusion_update_collapses_to_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = SpecialCaseMemory(Path(directory))
            row = memory.add_candidate(
                source_id="media-amend",
                features={"confidence": 0.9},
                conclusion={"diagnosis": {"summary": "draft"}},
                run_id="run-amend",
            )
            updated = memory.update_candidate_conclusion(
                row["case_id"],
                {"diagnosis": {"summary": "human amended"}},
                reviewer_id="operator",
                reason="reviewed",
            )
            self.assertEqual(updated["conclusion"]["diagnosis"]["summary"], "human amended")
            snapshots = [item for item in memory._snapshot() if item["case_id"] == row["case_id"]]
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(memory._snapshot()[-1]["conclusion"]["diagnosis"]["summary"], "human amended")

            memory.transition(row["case_id"], "eval_only", reviewer_id="operator", reason="evaluate")
            self.assertEqual(memory.search({"confidence": 0.9}), [])

    def test_case_state_transition_is_atomic_across_memory_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = SpecialCaseMemory(root)
            row = writer.add_candidate(
                source_id="media-race",
                features={"confidence": 0.9},
                conclusion={"label": "N"},
                run_id="run-race",
            )
            memories = (SpecialCaseMemory(root), SpecialCaseMemory(root))
            start = Barrier(2)

            def transition(memory, target):
                start.wait(timeout=5)
                try:
                    return ("committed", memory.transition(
                        row["case_id"], target, reviewer_id=f"reviewer-{target}", reason="concurrent test"
                    )["state"])
                except ValueError:
                    return ("rejected", target)

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(
                    lambda memory: transition(memory, "eval_only"),
                    memories,
                ))

            self.assertEqual(sum(result == "committed" for result, _ in outcomes), 1)
            self.assertEqual(sum(result == "rejected" for result, _ in outcomes), 1)
            snapshots = [item for item in writer._snapshot() if item.get("case_id") == row["case_id"]]
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(snapshots[-1]["state"], next(value for result, value in outcomes if result == "committed"))
            self.assertEqual(len(snapshots[-1]["history"]), 2)


if __name__ == "__main__":
    unittest.main()
