from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from agentic.platform.hitl import HumanReviewService
from agentic.scripts.demo_run import run_demo


class StreamlitNativeHitlTests(unittest.TestCase):
    def _pending_run(self, root: Path, action: str) -> tuple[Path, dict]:
        code_root = Path(__file__).resolve().parents[2]
        image_path = root / "inspection.png"
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[48:208, 48:208] = (90, 140, 210)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        runtime = root / "runtime"
        demo = run_demo(
            code_root,
            image_path,
            output=root / f"{action}.json",
            no_model=True,
            runtime_dir=runtime,
            run_id=f"ui-{action}",
            llm_profile="fake_json",
            native_hitl=True,
        )
        return runtime, demo

    def _app(self, runtime: Path) -> AppTest:
        app_path = Path(__file__).resolve().parents[2] / "web_interface" / "streamlit_app.py"
        with patch.dict(os.environ, {"MARS_AGENT_RUNTIME_DIR": str(runtime)}):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                app = AppTest.from_file(str(app_path), default_timeout=60)
                app.session_state["page"] = "agent"
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.selectbox(key="modern_jev_backend_select").value, "deterministic_rule")
                self.assertEqual(app.selectbox(key="modern_pending_run_select").label, "待审 run_id")
                self.assertTrue(any(item.label == "恢复待审任务" for item in app.get("expander")))
                self.assertTrue(any(item.label == "加载待审任务" for item in app.get("button")))
                app.button(key="load_pending_modern_demo").click().run()
                self.assertFalse(app.exception)
            return app

    def _exercise_action(self, runtime: Path, demo: dict, action: str, expected_status: str) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"agentic\.platform\.hitl")
            with patch.dict(os.environ, {"MARS_AGENT_RUNTIME_DIR": str(runtime)}):
                app = self._app(runtime)
                self.assertEqual(app.session_state["modern_demo"]["run_id"], demo["run_id"])
                version_panels = []
                for item in app.get("json"):
                    value = item.value
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = None
                    if isinstance(value, dict) and "rsi_runtime" in value:
                        version_panels.append(value)
                self.assertTrue(version_panels)
                runtime_panel = version_panels[-1]["rsi_runtime"]
                self.assertEqual(runtime_panel["status"], demo["rsi_runtime"]["status"])
                self.assertEqual(runtime_panel["version_id"], demo["rsi_runtime"]["version_id"])
                self.assertEqual(runtime_panel["prompt_variant"], demo["rsi_runtime"]["prompt_variant"])
                citations = [item.value for item in app.get("caption") if isinstance(item.value, str)]
                self.assertTrue(any("来源：" in value and "版本：" in value for value in citations))
                if action == "amend":
                    app.text_area(key="modern_amendment").set_value("reviewed evidence").run()
                app.button(key=f"modern_review_{action}").click().run()
                self.assertFalse(app.exception)
                updated = app.session_state["modern_demo"]
                self.assertEqual(updated["graph"]["result"]["status"], expected_status)
                self.assertEqual(updated["human_review"]["decision"]["action"], action)
                self.assertEqual(len(HumanReviewService(runtime).history(run_id=demo["run_id"])), 1)

                downloads = {button.key: button for button in app.get("download_button")}
                for key in ("modern_report_json", "modern_report_md", "modern_report_html", "modern_report_png"):
                    self.assertIn(key, downloads)
                self.assertTrue(any(str(key).startswith("knowledge_source_download_") for key in downloads))
                service = HumanReviewService(runtime)
                json_report = service.export_report(demo["run_id"], "json")
                markdown_report = service.export_report(demo["run_id"], "markdown")
                html_report = service.export_report(demo["run_id"], "html")
                self.assertEqual(json.loads(json_report)["run_id"], demo["run_id"])
                self.assertIn(demo["run_id"], markdown_report)
                self.assertIn(demo["run_id"], html_report)
                png_path = runtime.parent / "reports" / f"{demo['run_id']}.png"
                service.export_report(demo["run_id"], "png", destination=png_path)
                self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_persisted_pending_run_loads_and_all_review_actions_render_reports(self):
        expected = {
            "approve": "approved",
            "amend": "approved",
            "reject": "rejected",
            "request_retake": "retake_requested",
        }
        for action, status in expected.items():
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime, demo = self._pending_run(root, action)
                self._exercise_action(runtime, demo, action, status)

    def test_request_retake_upload_starts_linked_run_and_preserves_parent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, parent = self._pending_run(root, "request_retake")
            with patch.dict(os.environ, {"MARS_AGENT_RUNTIME_DIR": str(runtime)}):
                app = self._app(runtime)
                app.button(key="modern_review_request_retake").click().run()
                self.assertFalse(app.exception)
                service = HumanReviewService(runtime)
                parent_audit_before = service.history(run_id=parent["run_id"])
                self.assertEqual(len(parent_audit_before), 1)
                self.assertEqual(parent_audit_before[0]["action"], "request_retake")

                replacement_path = root / "replacement.png"
                replacement = np.zeros((256, 256, 3), dtype=np.uint8)
                replacement[32:224, 32:224] = (160, 90, 180)
                self.assertTrue(cv2.imwrite(str(replacement_path), replacement))
                app.file_uploader(key="modern_retake_upload").set_value(
                    (replacement_path.name, replacement_path.read_bytes(), "image/png")
                ).run()
                app.button(key="modern_start_retake").click().run()
                self.assertFalse(app.exception)

                child = app.session_state["modern_demo"]
                self.assertNotEqual(child["run_id"], parent["run_id"])
                self.assertEqual(child["parent_run_id"], parent["run_id"])
                self.assertEqual(child["retake_of_run_id"], parent["run_id"])
                persisted_child = json.loads(
                    (runtime / "modern_demos" / f"{child['run_id']}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(persisted_child["parent_run_id"], parent["run_id"])
                self.assertEqual(persisted_child["retake_of_run_id"], parent["run_id"])
                run_rows = [
                    json.loads(line)
                    for line in (runtime / "runs.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                child_finish = next(row for row in reversed(run_rows) if row.get("run_id") == child["run_id"] and "result" in row)
                self.assertEqual(child_finish["result"]["parent_run_id"], parent["run_id"])
                self.assertEqual(child_finish["result"]["retake_of_run_id"], parent["run_id"])

                app.button(key="modern_review_approve").click().run()
                self.assertFalse(app.exception)
                child_audit = service.history(run_id=child["run_id"])
                self.assertEqual(len(child_audit), 1)
                self.assertEqual(child_audit[0]["parent_run_id"], parent["run_id"])
                self.assertEqual(child_audit[0]["retake_of_run_id"], parent["run_id"])
                self.assertEqual(service.history(run_id=parent["run_id"]), parent_audit_before)

    def test_history_page_lists_and_filters_modern_human_review_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            service = HumanReviewService(runtime)
            service.pause(
                "history-audit-1",
                draft={"summary": "draft summary", "risk_level": "high"},
                route="full_loop",
                model_metadata={"model_profile": "fake_json"},
            )
            service.resume(
                "history-audit-1",
                action="amend",
                reviewer_id="history-reviewer",
                amendments={"summary": "approved amended summary"},
            )
            service.pause(
                "history-audit-2",
                draft={"summary": "qwen draft summary", "risk_level": "low"},
                route="simple_loop",
                model_metadata={"model_profile": "qwen_api"},
            )
            service.resume(
                "history-audit-2",
                action="approve",
                reviewer_id="qwen-reviewer",
            )
            app_path = Path(__file__).resolve().parents[2] / "web_interface" / "streamlit_app.py"
            with patch.dict(os.environ, {"MARS_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("sys.path", [str(app_path.parent), *sys.path]):
                    with patch("history_store.load_records", return_value=[]):
                        app = AppTest.from_file(str(app_path), default_timeout=60)
                        app.session_state["page"] = "history"
                        app.run()
                        self.assertFalse(app.exception)
                        self.assertEqual(app.selectbox(key="modern_history_route_filter").value, "全部")
                        self.assertEqual(app.selectbox(key="modern_history_report_pick").value, 0)
                        self.assertEqual(set(app.selectbox(key="modern_history_model_filter").options), {"全部", "fake_json", "qwen_api"})
                        self.assertTrue(any("2 条审核记录" in str(item.value) for item in app.caption))
                        self.assertIn("history-audit-1", str(app.selectbox(key="modern_history_report_pick").options))
                        download_keys = {button.key for button in app.get("download_button")}
                        self.assertTrue({"modern_history_report_json", "modern_history_report_md", "modern_history_report_html"}.issubset(download_keys))

                        app.selectbox(key="modern_history_model_filter").set_value("qwen_api").run()
                        self.assertFalse(app.exception)
                        self.assertTrue(any("1 条审核记录" in str(item.value) for item in app.caption))
                        report_options = str(app.selectbox(key="modern_history_report_pick").options)
                        self.assertIn("history-audit-2", report_options)
                        self.assertNotIn("history-audit-1", report_options)

                        app.text_input(key="modern_history_run_filter").set_value("missing-run").run()
                        self.assertFalse(app.exception)
                        self.assertTrue(any("0 条审核记录" in str(caption.value) for caption in app.caption))

    def test_history_page_renders_compact_mobile_record_cards_with_key_fields(self):
        app_path = Path(__file__).resolve().parents[2] / "web_interface" / "streamlit_app.py"
        records = [{
            "time": "2026-11-11 10:15:00",
            "source": "inspection-camera-east",
            "cell": "9306",
            "result": "H",
            "result_cn": "偏高",
            "conf": 0.91,
            "gray_mean": 142.5,
            "contrast": 31.2,
            "saturation": 0.68,
            "crust_speed": 2.1,
            "node_speed": 0.7,
            "frames": 24,
            "alarm": True,
            "device": "camera-east",
        }]
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            with patch.dict(os.environ, {"MARS_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("sys.path", [str(app_path.parent), *sys.path]):
                    with patch("history_store.load_records", return_value=records):
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                            app = AppTest.from_file(str(app_path), default_timeout=60)
                            app.session_state["page"] = "history"
                            app.run()
            self.assertFalse(app.exception)
            markdown = "\n".join(str(item.value) for item in app.get("markdown"))
            self.assertIn(".history-mobile-list{display:grid!important;}", markdown)
            self.assertIn("history-mobile-list", markdown)
            self.assertIn("2026-11-11 10:15:00", markdown)
            self.assertIn("偏高", markdown)
            self.assertIn("9306", markdown)
            self.assertIn("inspection-camera-east", markdown)
            self.assertIn("91.0%", markdown)


if __name__ == "__main__":
    unittest.main()
