import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import warnings
from streamlit.testing.v1 import AppTest

from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.hitl import AlreadyResumedError
from agentic.platform.presentation_review import PresentationReviewStore
from agentic.scripts.presentation_demo import build_all


class PresentationDemoTests(unittest.TestCase):
    def test_builds_four_route_cases_and_graph_backed_escalation_case(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            index = build_all(root, Path(directory))
            self.assertEqual(index["case_count"], 5)
            routes = {row["case_id"]: row["route"] for row in index["cases"]}
            self.assertEqual(routes, {
                "stable_case": "simple_loop",
                "conflict_high_risk": "full_loop",
                "bad_quality": "retake",
                "ood_unknown_case": "human",
                "simple_upgrade": "full_loop",
            })
            payload = json.loads((Path(directory) / "conflict_high_risk.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "flow_validation")
            self.assertEqual(payload["route_decision"]["router_backend"], "deterministic_rule")
            self.assertEqual(payload["route_decision"]["model_profile"], "deterministic_alu_router_v1")
            self.assertEqual(len(payload["agents"]), 5)
            self.assertEqual(
                {agent["agent_id"] for agent in payload["agents"]},
                {"visual_review", "evidence_quality", "knowledge_rag", "case_compare", "process_state"},
            )
            self.assertFalse(payload["claims"]["real_superheat_measurement"])
            self.assertTrue((Path(directory) / "conflict_high_risk.sidecar.json").exists())

            upgrade = json.loads((Path(directory) / "simple_upgrade.json").read_text(encoding="utf-8"))
            self.assertEqual(upgrade["status"], "flow_validation")
            self.assertEqual(upgrade["initial_route"], "simple_loop")
            self.assertEqual(upgrade["route_history"], ["simple_loop", "full_loop"])
            self.assertEqual(upgrade["upgrade_count"], 1)
            self.assertEqual(upgrade["fan_out"]["agent_count"], 5)
            self.assertEqual({agent["agent_id"] for agent in upgrade["agents"]}, {
                "visual_review", "evidence_quality", "knowledge_rag", "case_compare", "process_state",
            })
            self.assertIn("simple_loop_upgrade", {event["node"] for event in upgrade["graph_trace"]})
            self.assertIn("full_loop_fanout", {event["node"] for event in upgrade["graph_trace"]})
            self.assertFalse(upgrade["evaluation"]["metrics_claimable"])

    def test_demo_query_fixtures_have_ten_rows_each(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("knowledge_queries_demo.jsonl", "case_queries_demo.jsonl"):
            rows = [line for line in (root / "data" / "evaluation" / name).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 10, name)
            for line in rows:
                payload = json.loads(line)
                self.assertEqual(payload["fixture_type"], "demo_fixture")

    def test_presentation_review_persists_idempotently_with_candidate_only_case(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_all(root, output / "fixtures")
            fixture = json.loads((output / "fixtures" / "conflict_high_risk.json").read_text(encoding="utf-8"))
            store = PresentationReviewStore(output / "runtime")

            reviewed = store.apply(fixture, "approve")
            restored = store.load(fixture["run_id"])
            repeated = store.apply(fixture, "approve")

            self.assertEqual(restored, reviewed)
            self.assertEqual(repeated, reviewed)
            self.assertEqual(reviewed["human_review"]["action"], "approve")
            self.assertEqual(reviewed["graph"]["result"]["review"]["action"], "approve")
            candidate = reviewed["case_candidate"]
            self.assertEqual(candidate["state"], "candidate")
            self.assertEqual(candidate["run_id"], fixture["run_id"])
            self.assertEqual(candidate["applicable_when"], ["presentation_fixture"])
            self.assertEqual(candidate["conclusion"]["diagnosis"], reviewed["graph"]["result"]["diagnosis"])
            audit = [json.loads(line) for line in store.audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["run_id"], candidate["run_id"])
            self.assertEqual(audit[0]["case_id"], candidate["case_id"])
            cases = SpecialCaseMemory(output / "runtime" / "presentation_demo" / "case_memory")._snapshot()
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["state"], "candidate")

            with self.assertRaises(AlreadyResumedError):
                store.apply(fixture, "reject")

    def test_retake_or_reject_does_not_create_a_presentation_case(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_all(root, output / "fixtures")
            fixture = json.loads((output / "fixtures" / "bad_quality.json").read_text(encoding="utf-8"))
            store = PresentationReviewStore(output / "runtime")
            reviewed = store.apply(fixture, "request_retake")
            self.assertEqual(reviewed["human_review"]["status"], "waiting_for_retake")
            self.assertNotIn("case_candidate", reviewed)
            case_path = output / "runtime" / "presentation_demo" / "case_memory" / "special_cases.jsonl"
            self.assertFalse(case_path.exists())

    def test_streamlit_presentation_review_survives_page_recreation(self):
        root = Path(__file__).resolve().parents[2]
        app_path = root / "web_interface" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fixtures = output / "fixtures"
            runtime = output / "runtime"
            build_all(root, fixtures)
            environment = {
                "MARS_AGENT_RUNTIME_DIR": str(runtime),
                "MARS_AGENT_PRESENTATION_DEMO_DIR": str(fixtures),
            }

            def open_app():
                app = AppTest.from_file(str(app_path), default_timeout=60)
                app.session_state["page"] = "agent"
                app.run()
                self.assertFalse(app.exception)
                app.selectbox(key="presentation_case_select").set_value("stable_case").run()
                app.button(key="load_presentation_demo").click().run()
                self.assertFalse(app.exception)
                return app

            with patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                first = open_app()
                first.button(key="modern_review_approve").click().run()
                self.assertFalse(first.exception)
                self.assertEqual(first.session_state["modern_demo"]["human_review"]["action"], "approve")
                self.assertEqual(first.session_state["modern_demo"]["case_candidate"]["state"], "candidate")

                # A new Streamlit script session must restore the persisted decision.
                second = open_app()
                restored = second.session_state["modern_demo"]
                self.assertEqual(restored["human_review"]["action"], "approve")
                self.assertEqual(restored["case_candidate"]["run_id"], restored["run_id"])
                audit = [json.loads(line) for line in (runtime / "presentation_demo" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(audit), 1)
                self.assertEqual(audit[0]["run_id"], restored["run_id"])

    def test_streamlit_evaluation_panel_loads_e3_route_flow_and_delivery_snapshot(self):
        root = Path(__file__).resolve().parents[2]
        app_path = root / "web_interface" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fixtures = output / "fixtures"
            runtime = output / "runtime"
            build_all(root, fixtures)
            runtime.mkdir(parents=True)
            evaluation = {
                "schema_version": "evaluation-report-1",
                "evaluation_freeze": {
                    "status": "invalid_manifests",
                    "frozen": False,
                    "evaluation_data_eligible": False,
                    "claimable": False,
                    "claimability_reason": "Freeze validation failed; metric claims are not eligible.",
                    "manifest_details": {
                        "visual_manifest.jsonl": {
                            "status": "invalid",
                            "record_count": 3,
                            "valid_record_count": 1,
                            "invalid_error_counts": {
                                "annotation_not_expert_reviewed": 1,
                                "human_label_missing_or_invalid": 1,
                            },
                            "errors": ["invalid_rows:2", "duplicate_identities:1"],
                        },
                        "route_manifest.jsonl": {
                            "status": "invalid",
                            "record_count": 1,
                            "valid_record_count": 0,
                            "errors": ["insufficient_records:1<10"],
                        },
                    },
                },
                "knowledge_pack_validation": {
                    "status": "pass",
                    "claimable": False,
                    "record_count": 3,
                    "valid_record_count": 3,
                    "invalid_record_count": 0,
                },
                "retrieval": {
                    "knowledge": {
                        "query_count": 10,
                        "fixture_pass_rate": 1.0,
                        "citation_integrity": {
                            "citation_count": 10,
                            "valid_citation_count": 10,
                            "pass_rate": 1.0,
                            "claimable": False,
                        },
                    },
                    "cases": {"query_count": 10, "fixture_pass_rate": 1.0},
                },
                "jev": {
                    "fixture_count": 5,
                    "fixture_route_match_rate": 1.0,
                    "simple_fixture_recall": 1.0,
                    "complex_fixture_full_recall": 1.0,
                    "retake_fixture_recall": 1.0,
                    "human_fixture_recall": 1.0,
                },
                "route_flow": {
                    "status": "evaluated",
                    "run_count": 3,
                    "upgrade_count": 1,
                    "upgrade_rate": 1 / 3,
                    "transitions": {"simple_loop->full_loop": 1},
                },
                "input_integrity": {"status": "passed", "claimable": True},
                "worker_execution": {
                    "status": "evaluated",
                    "worker_count": 5,
                    "retry_total": 1,
                    "workers_retried": 1,
                    "timeout_worker_count": 1,
                    "policy_coverage_rate": 1.0,
                },
                "system": {
                    "runtime_scope": {"scope": "explicit_run_ids", "claimable": True},
                    "event_count": 10,
                    "node_failure_rate": 0.1,
                    "timeout_rate": 0.05,
                    "retry_rate": 0.2,
                    "fallback_rate": 0.0,
                },
            }
            (runtime / "evaluation.e3_ui.json").write_text(json.dumps(evaluation), encoding="utf-8")
            delivery = {
                "schema_version": "delivery-snapshot-1",
                "status": "passed",
                "selected_run_ids": ["run-ui-1"],
                "finished_run_count": 1,
                "route_fixture_count": 5,
                "errors": [],
            }
            (runtime / "delivery_snapshot.e3_ui.json").write_text(json.dumps(delivery), encoding="utf-8")
            environment = {
                "MARS_AGENT_RUNTIME_DIR": str(runtime),
                "MARS_AGENT_PRESENTATION_DEMO_DIR": str(fixtures),
            }

            with patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                app = AppTest.from_file(str(app_path), default_timeout=60)
                app.session_state["page"] = "agent"
                app.run()
                self.assertFalse(app.exception)
                app.selectbox(key="presentation_case_select").set_value("stable_case").run()
                app.button(key="load_presentation_demo").click().run()
                self.assertFalse(app.exception)

            captions = [str(item.value) for item in app.get("caption")]
            self.assertTrue(any("evaluation.e3_ui.json" in value for value in captions))
            self.assertTrue(any("delivery_snapshot.e3_ui.json" in value for value in captions))
            metrics = {item.label: item.value for item in app.get("metric")}
            self.assertEqual(metrics.get("评测集冻结状态"), "未冻结")
            self.assertEqual(metrics.get("研究结论可声明"), "否")
            self.assertEqual(metrics.get("过程状态"), "演示摘要")
            self.assertEqual(metrics.get("流程状态"), "待人工确认")
            self.assertTrue(any("3 次" in value and "升级 1" in value for value in captions) or any("3" in value and "1" in value for value in captions))
            evaluation_rows = []
            for item in app.get("dataframe"):
                frame = item.value
                if hasattr(frame, "to_dict"):
                    evaluation_rows.extend(frame.to_dict("records"))
            self.assertTrue(any(row.get("评测项") == "知识引用字段完整性" for row in evaluation_rows))
            self.assertTrue(any(row.get("评测项") == "工艺知识包结构" and row.get("结果") == "3/3" for row in evaluation_rows))
            jev_recall_row = next((row for row in evaluation_rows if row.get("评测项") == "JEV 四路 fixture 召回"), None)
            self.assertIsNotNone(jev_recall_row)
            self.assertEqual(jev_recall_row.get("结果"), "100% / 100% / 100% / 100%")
            self.assertEqual(jev_recall_row.get("可声明"), "否 · policy fixture")
            self.assertEqual(set(jev_recall_row), {"评测项", "结果", "性质", "可声明"})
            self.assertTrue(any(row.get("评测项") == "Full-loop Agent 交接" and row.get("可声明") == "否 · 流程完整性" for row in evaluation_rows))
            self.assertTrue(any(
                row.get("必需清单") == "visual_manifest.jsonl"
                and row.get("状态") == "invalid"
                and row.get("记录数") == 3
                and row.get("有效记录") == 1
                and row.get("无效记录") == 2
                and "duplicate_identities:1" in row.get("校验错误", "")
                for row in evaluation_rows
            ))
            self.assertTrue(any(
                row.get("必需清单") == "visual_manifest.jsonl"
                and row.get("字段/审核拒绝原因") == "annotation_not_expert_reviewed: 1；human_label_missing_or_invalid: 1"
                for row in evaluation_rows
            ))
            self.assertTrue(any(
                row.get("必需清单") == "route_manifest.jsonl"
                and row.get("状态") == "invalid"
                and row.get("记录数") == 1
                and row.get("无效记录") == 1
                and "insufficient_records:1<10" in row.get("校验错误", "")
                for row in evaluation_rows
            ))
            self.assertTrue(any("simple_loop->full_loop 1 次" in str(row.get("结果", "")) for row in evaluation_rows))
            self.assertTrue(any(row.get("评测项") == "节点失败 / 超时 / 重试" and row.get("结果") == "10% / 5% / 20%" for row in evaluation_rows))
            self.assertTrue(any(row.get("评测项") == "Worker 策略执行" and row.get("结果") == "5 个 · 重试 1 · 超时 1 · 策略覆盖 100%" for row in evaluation_rows))
            json_panels = []
            for item in app.get("json"):
                value = item.value
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                json_panels.append(value)
            self.assertTrue(any(isinstance(value, dict) and value.get("selected_run_ids") == ["run-ui-1"] for value in json_panels))

    def test_streamlit_presentation_displays_simple_to_full_upgrade_trace(self):
        root = Path(__file__).resolve().parents[2]
        app_path = root / "web_interface" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fixtures = output / "fixtures"
            build_all(root, fixtures)
            environment = {
                "MARS_AGENT_RUNTIME_DIR": str(output / "runtime"),
                "MARS_AGENT_PRESENTATION_DEMO_DIR": str(fixtures),
            }
            with patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                app = AppTest.from_file(str(app_path), default_timeout=60)
                app.session_state["page"] = "agent"
                app.run()
                self.assertFalse(app.exception)
                app.selectbox(key="presentation_case_select").set_value("simple_upgrade").run()
                app.button(key="load_presentation_demo").click().run()
                self.assertFalse(app.exception)

            self.assertEqual(app.session_state["modern_demo"]["graph"]["route"], "full_loop")
            self.assertEqual(app.session_state["modern_demo"]["graph"]["result"]["route_history"], ["simple_loop", "full_loop"])
            captions = [str(item.value) for item in app.get("caption")]
            self.assertTrue(any("simple_loop → full_loop" in value and "升级 1 次" in value for value in captions))
            trace_rows = []
            for item in app.get("dataframe"):
                frame = item.value
                if hasattr(frame, "to_dict"):
                    trace_rows.extend(frame.to_dict("records"))
            self.assertTrue(any(row.get("节点") == "simple_loop_upgrade" for row in trace_rows))
            self.assertTrue(any(row.get("节点") == "full_loop_fanout" for row in trace_rows))
            worker_rows = []
            for item in app.get("dataframe"):
                frame = item.value
                if hasattr(frame, "to_dict"):
                    worker_rows.extend(frame.to_dict("records"))
            self.assertTrue(any(row.get("策略") == "visual-review-v1" and row.get("Agent") == "visual_review" for row in worker_rows))


if __name__ == "__main__":
    unittest.main()
