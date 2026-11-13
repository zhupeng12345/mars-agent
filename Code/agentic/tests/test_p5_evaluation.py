import csv
import tempfile
import json
import unittest
from pathlib import Path

from agentic.evaluation.metrics import end_to_end_latency_metrics, evidence_metrics, latency_metrics, multi_agent_flow_metrics, route_flow_metrics, route_metrics, rsi_runtime_policy_metrics, system_health_metrics, worker_execution_metrics
from agentic.evaluation.run_all import run_all, write_report_artifacts
from agentic.scripts.freeze_evaluation import MINIMUM_RECORDS, REQUIRED_MANIFESTS


class P5EvaluationTests(unittest.TestCase):
    @staticmethod
    def _write_freezable_manifests(root: Path) -> None:
        evaluation = root / "data" / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        row_builders = {
            "visual_manifest.jsonl": lambda i: {"window_id": f"w{i}", "source_group": f"video-{i}", "human_label": "N", "split": "test", "annotation_status": "expert_reviewed"},
            "process_manifest.jsonl": lambda i: {"sample_idx": i, "anchor_ts": f"2026-01-01T00:{i % 60:02d}:00", "y_true": i % 2, "split": "test", "annotation_status": "expert_reviewed"},
            "route_manifest.jsonl": lambda i: {"case_id": f"route-{i}", "features": {"run_id": f"run-{i}", "visual_confidence": 0.92, "risk_score": 0.1, "knowledge_coverage": 1.0, "case_similarity": 0.93, "novelty_score": 0.08, "evidence_conflict": 0.02, "features": {"quality_status": "pass", "target_present": True, "case_applicable": True}}, "expert_route": "simple_loop", "annotation_id": f"ann-{i}", "annotation_status": "expert_reviewed"},
            "knowledge_queries.jsonl": lambda i: {"query_id": f"k-{i}", "query": f"query {i}", "expected_source_ids": ["rule-1"], "annotation_status": "expert_reviewed"},
            "case_queries.jsonl": lambda i: {"query_id": f"c-{i}", "query": f"query {i}", "expected_outcome": "match", "annotation_status": "expert_reviewed"},
            "hitl_cases.jsonl": lambda i: {"case_id": f"h-{i}", "expected_action": "approve", "reviewer_id": f"reviewer-{i}", "annotation_status": "expert_reviewed"},
        }
        for name, builder in row_builders.items():
            rows = [builder(index) for index in range(MINIMUM_RECORDS[name])]
            (evaluation / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_system_metrics_include_review_action_distribution(self):
        report = system_health_metrics(
            [],
            [],
            [
                {"action": "approve"},
                {"action": "amend"},
                {"action": "reject"},
                {"action": "request_retake"},
                {"action": "approve"},
            ],
        )
        self.assertEqual(report["human_decision_count"], 5)
        self.assertEqual(report["human_action_counts"], {"approve": 2, "amend": 1, "reject": 1, "request_retake": 1})
        self.assertEqual(report["human_adoption_rate"], 0.6)

    def test_main_evaluation_csv_preserves_comma_and_quotes_in_string_values(self):
        note = 'fixture, with a "quoted" phrase'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = write_report_artifacts(
                root,
                {"schema_version": "evaluation-test-1", "claim_policy": note},
                root / "evaluation.json",
                runtime_dir=root / "runtime",
            )
            with Path(artifacts["csv"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(rows[0], ["metric", "value"])
        self.assertTrue(all(len(row) == 2 for row in rows))
        values = {metric: json.loads(value) for metric, value in rows[1:]}
        self.assertEqual(values["claim_policy"], note)

    def test_route_and_system_metrics(self):
        report = route_metrics([
            {"route": "simple_loop", "expected_route": "simple_loop"},
            {"route": "full_loop", "expected_route": "full_loop"},
            {"route": "retake", "expected_route": "retake"},
            {"route": "human", "expected_route": "human"},
            {"route": "human", "expected_route": "human"},
        ])
        self.assertEqual(report["fixture_route_match_rate"], 1.0)
        self.assertEqual(report["simple_fixture_recall"], 1.0)
        self.assertEqual(report["complex_fixture_full_recall"], 1.0)
        self.assertEqual(report["retake_fixture_recall"], 1.0)
        self.assertEqual(report["human_fixture_recall"], 1.0)
        self.assertEqual(report["evaluation_kind"], "policy_fixture_regression")
        self.assertFalse(report["model_accuracy_claimable"])
        self.assertNotIn("route_accuracy", report)
        self.assertEqual(latency_metrics([{"elapsed_ms": 10}, {"elapsed_ms": 20}])["count"], 2)
        self.assertEqual(evidence_metrics([{"evidence_bundle": [{"agent_id": "x"}]}])["runs"], 1)

    def test_route_flow_audit_preserves_simple_upgrade_without_accuracy_claim(self):
        report = route_flow_metrics([
            {"initial_route": "simple_loop", "route": "full_loop", "route_history": ["simple_loop", "full_loop"], "upgrade_count": 1},
            {"initial_route": "full_loop", "route": "full_loop", "route_history": ["full_loop"], "upgrade_count": 0},
        ])
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(report["transitions"], {"full_loop->full_loop": 1, "simple_loop->full_loop": 1})
        self.assertEqual(report["upgrade_count"], 1)
        self.assertEqual(report["upgrade_rate"], 0.5)
        self.assertFalse(report["accuracy_claimable"])
        self.assertFalse(report["efficiency_gain_claimable"])

    def test_rsi_runtime_policy_audit_counts_versions_without_claiming_gain(self):
        report = rsi_runtime_policy_metrics([
            {"run_id": "r1", "finished_at": "now", "result": {"route": "simple_loop", "route_decision": {"policy_version": "v-1", "policy_hash": "hash-1", "router_backend": "laya_adapter"}}},
            {"run_id": "r2", "finished_at": "now", "result": {"route": "full_loop", "route_decision": {"policy_version": "v-1", "policy_hash": "hash-1", "router_backend": "laya_adapter_fallback"}}},
            {"run_id": "r3", "finished_at": "now", "result": {"route": "full_loop", "route_decision": {"policy_version": "base", "policy_hash": "builtin", "router_backend": "deterministic_rule"}}},
            {"run_id": "r4", "finished_at": "now", "result": {"route": "human"}},
            {"run_id": "r5", "finished_at": None, "result": None},
        ])
        self.assertEqual(report["input_run_record_count"], 5)
        self.assertEqual(report["selected_completed_run_count"], 4)
        self.assertEqual(report["incomplete_run_record_count"], 1)
        self.assertEqual(report["runs_with_policy_version"], 3)
        self.assertEqual(report["runs_missing_policy_version"], 1)
        self.assertEqual(report["versions"]["v-1"]["routes"], {"simple_loop": 1, "full_loop": 1})
        self.assertEqual(report["runs_with_prompt_variant"], 0)
        self.assertEqual(report["runs_with_query_policy"], 0)
        self.assertFalse(report["accuracy_claimable"])
        self.assertFalse(report["behavior_gain_claimable"])

    def test_rsi_runtime_audit_records_consumed_prompt_and_query_surfaces(self):
        report = rsi_runtime_policy_metrics([
            {
                "run_id": "r1",
                "finished_at": "now",
                "result": {
                    "route": "full_loop",
                    "route_decision": {"policy_version": "v-2", "policy_hash": "hash-2"},
                    "rsi_runtime": {
                        "prompt_variant": "diagnosis-evidence-first-v1",
                        "knowledge_query_terms": ["thermal trend"],
                    },
                },
            },
        ])
        self.assertEqual(report["runs_with_prompt_variant"], 1)
        self.assertEqual(report["runs_with_query_policy"], 1)
        self.assertEqual(report["prompt_variants"], {"diagnosis-evidence-first-v1": 1})
        self.assertEqual(report["query_term_sets"], {"[\"thermal trend\"]": 1})

    def test_evidence_metrics_resolve_diagnosis_evidence_refs_without_counting_handoffs(self):
        results = [
            {
                "route": "full_loop",
                "visual": {"source_id": "frame-001"},
                "diagnosis": {
                    "diagnosis": "Observed a localized blue flame.",
                    "evidence": [{"source_id": "frame-001", "kind": "visual_roi"}],
                    "source_ids": [],
                },
            },
            {
                "route": "full_loop",
                "diagnosis": {"diagnosis": "Potential anomaly.", "source_ids": ["missing-source"]},
            },
            {
                "route": "retake",
                "diagnosis": {
                    "diagnosis": "",
                    "summary": "retake evidence aggregation",
                    "findings": ["evidence_agents=0", "failed_agents=0"],
                },
            },
        ]
        report = evidence_metrics(results)
        self.assertEqual(report["eligible_conclusions"], 2)
        self.assertEqual(report["excluded_nonconclusions"], 1)
        self.assertEqual(report["citation_count"], 2)
        self.assertEqual(report["citation_exists_rate"], 0.5)
        self.assertIsNone(report["citation_support_rate"])
        self.assertEqual(report["no_source_conclusion_count"], 0)
        self.assertEqual(report["no_source_conclusion_ratio"], 0.0)

    def test_multi_agent_flow_metrics_measure_coverage_failure_abstention_and_lineage(self):
        result = {
            "route": "full_loop",
            "evidence_bundle": [
                {"agent_id": "visual_review", "status": "completed", "output": {"source_ids": ["frame-1"]}, "evidence": [{"agent_id": "visual_review", "source_id": "frame-1"}]},
                {"agent_id": "evidence_quality", "status": "completed", "output": {"source_ids": ["frame-1"]}, "evidence": [{"agent_id": "evidence_quality", "source_id": "frame-1"}]},
                {"agent_id": "knowledge_rag", "status": "completed", "output": {"analysis_status": "abstain", "source_ids": []}, "evidence": []},
                {"agent_id": "case_compare", "status": "failed", "output": {}, "evidence": []},
                {"agent_id": "process_state", "status": "completed", "output": {"source_ids": ["process-window"]}, "evidence": [{"agent_id": "visual_review", "source_id": "process-window"}]},
            ],
        }
        report = multi_agent_flow_metrics([result])
        self.assertEqual(report["status"], "evaluated")
        self.assertFalse(report["accuracy_claimable"])
        self.assertEqual(report["eligible_full_loop_runs"], 1)
        self.assertEqual(report["expected_worker_slots"], 5)
        self.assertEqual(report["worker_slot_coverage_rate"], 1.0)
        self.assertEqual(report["completed_worker_slots"], 4)
        self.assertEqual(report["failed_worker_slots"], 1)
        self.assertEqual(report["missing_worker_slots"], 0)
        self.assertEqual(report["abstained_worker_outputs"], 1)
        self.assertEqual(report["source_lineage_violations"], 1)
        self.assertEqual(report["fully_successful_run_rate"], 0.0)

    def test_multi_agent_metrics_mark_absent_full_loop_as_not_run(self):
        report = multi_agent_flow_metrics([{"route": "simple_loop", "evidence_bundle": []}])
        self.assertEqual(report["status"], "not_run")
        self.assertEqual(report["eligible_full_loop_runs"], 0)
        self.assertIsNone(report["worker_slot_coverage_rate"])
        self.assertIsNone(report["fully_successful_run_rate"])

    def test_multi_agent_lineage_counts_missing_sources_and_excludes_abstentions(self):
        report = multi_agent_flow_metrics([{
            "route": "full_loop",
            "evidence_bundle": [
                {"agent_id": agent_id, "status": "completed", "output": {"source_ids": []}, "evidence": []}
                for agent_id in ("visual_review", "evidence_quality", "case_compare", "process_state")
            ] + [
                {"agent_id": "knowledge_rag", "status": "completed", "output": {"analysis_status": "abstain", "source_ids": []}, "evidence": []}
            ],
        }])
        self.assertEqual(report["source_lineage_checks"], 4)
        self.assertEqual(report["source_lineage_missing"], 4)
        self.assertEqual(report["source_lineage_integrity_rate"], 0.0)

    def test_worker_execution_metrics_report_retry_timeout_and_policy_coverage(self):
        report = worker_execution_metrics([{
            "route": "full_loop",
            "evidence_bundle": [
                {"agent_id": "visual_review", "status": "completed", "attempt": 2, "metadata": {"retry_count": 1, "retry_policy_id": "visual-review-v1"}},
                {"agent_id": "knowledge_rag", "status": "failed", "error_code": "TimeoutError", "metadata": {"retry_count": 0, "retry_policy_id": "knowledge-rag-v1", "timeout_s": 20}},
            ],
        }])
        self.assertEqual(report["worker_count"], 2)
        self.assertEqual(report["retry_total"], 1)
        self.assertEqual(report["workers_retried"], 1)
        self.assertEqual(report["failed_worker_count"], 1)
        self.assertEqual(report["timeout_worker_count"], 1)
        self.assertEqual(report["policy_coverage_rate"], 1.0)
        self.assertEqual(report["policy_counts"], {"knowledge-rag-v1": 1, "visual-review-v1": 1})
        self.assertFalse(report["diagnostic_accuracy_claimable"])

    def test_run_all_scopes_multi_agent_metrics_to_selected_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            runtime = root / "runtime"
            evaluation.mkdir(parents=True)
            runtime.mkdir()
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            full_result = {
                "route": "full_loop",
                "initial_route": "full_loop",
                "route_history": ["full_loop"],
                "upgrade_count": 0,
                "evidence_bundle": [
                    {"agent_id": agent_id, "status": "completed", "output": {"source_ids": []}, "evidence": []}
                    for agent_id in ("visual_review", "evidence_quality", "knowledge_rag", "case_compare", "process_state")
                ],
            }
            (runtime / "runs.jsonl").write_text("\n".join([
                json.dumps({"run_id": "selected", "status": "completed", "result": full_result}),
                json.dumps({"run_id": "unrelated", "status": "completed", "result": full_result}),
            ]) + "\n", encoding="utf-8")
            (runtime / "events.jsonl").write_text("", encoding="utf-8")
            (runtime / "hitl_audit.jsonl").write_text("", encoding="utf-8")
            repeats_path = root / "repeated_process_evaluation.json"
            repeats_path.write_text(json.dumps({"status": "experimental_baseline", "run_count": 5}), encoding="utf-8")
            report = run_all(root, runtime_dir=runtime, runtime_run_ids=["selected"], process_repeats_path=repeats_path)
        self.assertEqual(report["multi_agent"]["eligible_full_loop_runs"], 1)
        self.assertEqual(report["multi_agent"]["fully_successful_runs"], 1)
        self.assertEqual(report["route_flow"]["status"], "evaluated")
        self.assertEqual(report["route_flow"]["run_count"], 1)
        self.assertEqual(report["process"]["repeated_seed_evaluation"]["run_count"], 5)

    def test_runtime_metrics_require_explicit_run_scope_to_be_claimable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            runtime = root / "runtime"
            evaluation.mkdir(parents=True)
            runtime.mkdir()
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            (runtime / "events.jsonl").write_text("\n".join([
                json.dumps({"run_id": "selected", "node": "one", "elapsed_ms": 10, "status": "completed"}),
                json.dumps({"run_id": "unrelated", "node": "two", "elapsed_ms": 900, "status": "completed"}),
            ]) + "\n", encoding="utf-8")
            (runtime / "runs.jsonl").write_text("\n".join([
                json.dumps({"run_id": "selected", "result": {"diagnosis": {}, "evidence_bundle": []}}),
                json.dumps({"run_id": "unrelated", "result": {"diagnosis": {}, "evidence_bundle": []}}),
            ]) + "\n", encoding="utf-8")
            (runtime / "hitl_audit.jsonl").write_text("", encoding="utf-8")

            selected = run_all(root, runtime_dir=runtime, runtime_run_ids=["selected"])
            mixed = run_all(root, runtime_dir=runtime)

            self.assertEqual(selected["system"]["node_latency"]["count"], 1)
            self.assertEqual(selected["system"]["node_latency"]["p50_ms"], 10.0)
            self.assertEqual(selected["system"]["end_to_end_latency"]["count"], 0)
            self.assertFalse(selected["system"]["end_to_end_latency"]["claimable"])
            self.assertTrue(selected["system"]["runtime_scope"]["claimable"])
            self.assertEqual(selected["system"]["runtime_scope"]["scope"], "explicit_run_ids")
            self.assertEqual(mixed["system"]["node_latency"]["count"], 2)
        self.assertFalse(mixed["system"]["runtime_scope"]["claimable"])
        self.assertEqual(mixed["system"]["runtime_scope"]["scope"], "unscoped_history_mixed")

    def test_end_to_end_latency_pairs_by_run_and_excludes_incomplete(self):
        rows = [
            {"run_id": "complete", "started_at": "2026-11-09T00:00:00Z", "status": "running"},
            {"run_id": "complete", "finished_at": "2026-11-09T02:02:00+02:00", "status": "completed"},
            {"run_id": "missing-finish", "started_at": "2026-11-09T00:00:00Z", "status": "running"},
            {"run_id": "missing-start", "finished_at": "2026-11-09T00:03:00Z", "status": "completed"},
            {"run_id": "zero", "started_at": "2026-11-09T09:00:00+09:00", "status": "running"},
            {"run_id": "zero", "finished_at": "2026-11-09T00:00:00Z", "status": "completed"},
            {"run_id": "invalid", "started_at": "not-a-time", "status": "running"},
            {"run_id": "invalid", "finished_at": "2026-11-09T00:01:00Z", "status": "completed"},
        ]
        report = end_to_end_latency_metrics(rows, selected_run_ids=["complete", "missing-finish", "missing-start", "zero", "invalid"])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["p50_ms"], 0.0)
        self.assertEqual(report["p99_ms"], 120000.0)
        self.assertEqual(report["excluded_incomplete_run_count"], 3)
        self.assertEqual(report["excluded_invalid_timestamp_count"], 1)
        self.assertEqual(report["evaluation_kind"], "runtime_end_to_end_latency")

    def test_explicit_runtime_scope_separates_node_and_task_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            runtime = root / "runtime"
            evaluation.mkdir(parents=True)
            runtime.mkdir()
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            (runtime / "events.jsonl").write_text(json.dumps({"run_id": "selected", "elapsed_ms": 20, "status": "completed"}) + "\n", encoding="utf-8")
            (runtime / "runs.jsonl").write_text("\n".join([
                json.dumps({"run_id": "selected", "status": "running", "started_at": "2026-11-09T00:00:00Z"}),
                json.dumps({"run_id": "selected", "status": "completed", "finished_at": "2026-11-09T00:02:00Z", "result": {"diagnosis": {}}}),
                json.dumps({"run_id": "unrelated", "status": "running", "started_at": "2026-11-09T00:00:00Z"}),
                json.dumps({"run_id": "unrelated", "status": "completed", "finished_at": "2026-11-09T01:00:00Z", "result": {"diagnosis": {}}}),
            ]) + "\n", encoding="utf-8")
            (runtime / "hitl_audit.jsonl").write_text("", encoding="utf-8")
            report = run_all(root, runtime_dir=runtime, runtime_run_ids=["selected"])
        node = report["system"]["node_latency"]
        end_to_end = report["system"]["end_to_end_latency"]
        self.assertEqual(node["p50_ms"], 20.0)
        self.assertEqual(end_to_end["p50_ms"], 120000.0)
        self.assertTrue(node["claimable"])
        self.assertTrue(end_to_end["claimable"])

    def test_run_all_is_explicit_about_missing_visual_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "evaluation").mkdir(parents=True)
            (root / "data" / "evaluation" / "route_manifest.jsonl").write_text("", encoding="utf-8")
            report = run_all(root)
        self.assertFalse(report["claims"]["real_superheat_measurement"])
        self.assertEqual(report["visual"]["metric_status"], "structure_validation_only")
        self.assertEqual(report["image_quality_gate"]["evaluation_kind"], "synthetic_quality_gate_regression")
        self.assertEqual(report["visual_image_level_baseline"]["status"], "not_run")
        self.assertEqual(report["process"]["independent_baseline_evaluation"]["status"], "not_run")
        self.assertEqual(report["process"]["feature_attribution"]["status"], "not_run")
        self.assertEqual(report["process"]["repeated_seed_evaluation"]["status"], "not_run")
        self.assertIn("report_path", report["process"]["independent_baseline_evaluation"])

    def test_run_all_surfaces_missing_evaluation_manifests_without_claimability(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_all(Path(directory))
        freeze = report["evaluation_freeze"]
        self.assertEqual(freeze["status"], "missing_manifests")
        self.assertFalse(freeze["frozen"])
        self.assertFalse(freeze["evaluation_data_eligible"])
        self.assertFalse(freeze["claimable"])
        self.assertEqual(set(freeze["missing_manifests"]), set(REQUIRED_MANIFESTS))
        self.assertEqual(freeze["manifest_details"]["knowledge_queries.jsonl"]["record_count"], 0)
        self.assertIn("metric claims", freeze["claimability_reason"])

    def test_run_all_surfaces_invalid_manifest_and_record_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "knowledge_queries.jsonl").write_text(
                "\n".join([
                    json.dumps({"query_id": "reviewed", "query": "query", "expected_source_ids": [], "annotation_status": "reviewed"}),
                    json.dumps({"query_id": "fixture", "query": "query", "expected_source_ids": [], "annotation_status": "demo_fixture", "fixture_type": "demo_fixture"}),
                ]) + "\n",
                encoding="utf-8",
            )
            freeze = run_all(root)["evaluation_freeze"]
        self.assertEqual(freeze["status"], "incomplete")
        self.assertFalse(freeze["frozen"])
        self.assertFalse(freeze["claimable"])
        self.assertIn("knowledge_queries.jsonl", freeze["invalid_manifests"])
        knowledge = freeze["manifest_details"]["knowledge_queries.jsonl"]
        self.assertEqual(knowledge["record_count"], 2)
        self.assertEqual(knowledge["valid_record_count"], 1)
        self.assertEqual(knowledge["minimum_record_count"], MINIMUM_RECORDS["knowledge_queries.jsonl"])
        self.assertIn("insufficient_records:2<10", knowledge["errors"])
        self.assertEqual(knowledge["invalid_error_counts"], {"annotation_not_expert_reviewed": 1})
        self.assertEqual(knowledge["invalid_row_examples"][0]["errors"], ["annotation_not_expert_reviewed"])

    def test_run_all_reports_frozen_data_as_eligible_but_not_metric_claimable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_freezable_manifests(root)
            freeze = run_all(root)["evaluation_freeze"]
        self.assertEqual(freeze["status"], "frozen")
        self.assertTrue(freeze["frozen"])
        self.assertTrue(freeze["evaluation_data_eligible"])
        self.assertFalse(freeze["claimable"])
        self.assertEqual(freeze["missing_manifests"], [])
        self.assertEqual(freeze["invalid_manifests"], [])
        self.assertEqual(freeze["manifest_details"]["visual_manifest.jsonl"]["record_count"], 100)

    def test_run_all_includes_knowledge_pack_structure_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "agentic" / "knowledge"
            knowledge.mkdir(parents=True)
            rule = {
                "rule_id": "alu-rule-1",
                "title": "工艺检查规则",
                "source": "manual",
                "source_version": "v1",
                "effective_date": "2026-01-01",
                "preconditions": ["quality_pass"],
                "applicability": ["visual"],
                "risk_level": "medium",
                "allowed_output": ["inspection_item"],
                "text": "记录可追溯的现场检查证据。",
            }
            (knowledge / "rules_v1.jsonl").write_text(json.dumps(rule, ensure_ascii=False) + "\n", encoding="utf-8")
            report = run_all(root)

        validation = report["knowledge_pack_validation"]
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["record_count"], 1)
        self.assertFalse(validation["claimable"])

    def test_run_all_marks_malformed_optional_report_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            invalid = root / "invalid_process.json"
            invalid.write_text("not-json", encoding="utf-8")
            report = run_all(root, process_evaluation_path=invalid)
        self.assertEqual(report["process"]["independent_baseline_evaluation"]["status"], "invalid_report")
        self.assertEqual(report["process"]["independent_baseline_evaluation"]["reason"], "invalid_json")

    def test_run_all_surfaces_runtime_jsonl_parse_errors_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            runtime = root / "runtime"
            evaluation.mkdir(parents=True)
            runtime.mkdir()
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            (runtime / "events.jsonl").write_text('{"run_id":"ok"}\nnot-json\n', encoding="utf-8")
            (runtime / "runs.jsonl").write_text("", encoding="utf-8")
            (runtime / "hitl_audit.jsonl").write_text("", encoding="utf-8")
            report = run_all(root, runtime_dir=runtime)
        self.assertEqual(report["input_integrity"]["status"], "invalid_runtime_records")
        self.assertFalse(report["input_integrity"]["claimable"])
        self.assertIn("events.jsonl:invalid_json_line:2", report["input_integrity"]["runtime_jsonl_parse_errors"])

    def test_run_all_can_include_explicit_process_baseline_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "data" / "evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "route_manifest.jsonl").write_text("", encoding="utf-8")
            process_report = root / "process_eval.strict.json"
            process_report.write_text(json.dumps({
                "schema_version": "process-evaluation-1",
                "status": "experimental_baseline",
                "alignment_quality": "unknown",
                "models": {"cnn": {"roc_auc": 0.42, "sample_count": 20}},
            }), encoding="utf-8")
            attribution_report = root / "feature_attribution.json"
            attribution_report.write_text(json.dumps({
                "schema_version": "process-feature-attribution-1",
                "status": "experimental_baseline",
                "validation_selected_features": ["potVolt"],
                "claims": {"video_fusion_evaluated": False, "real_superheat_measurement": False},
            }), encoding="utf-8")
            report = run_all(root, process_evaluation_path=process_report, process_attribution_path=attribution_report)
        process = report["process"]["independent_baseline_evaluation"]
        self.assertEqual(process["status"], "experimental_baseline")
        self.assertEqual(process["models"]["cnn"]["sample_count"], 20)
        self.assertFalse(report["process"]["video_classification_improvement_claimed"])
        self.assertEqual(report["process"]["feature_attribution"]["validation_selected_features"], ["potVolt"])


if __name__ == "__main__":
    unittest.main()
