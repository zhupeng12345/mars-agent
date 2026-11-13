import unittest
import time

from agentic.orchestration.diagnosis_graph import run_diagnosis_graph
from agentic.platform.contracts import InspectionFeatures
from agentic.platform.jev.router import JevRouter


def make_features(run_id="run-graph", **kwargs):
    values = {"run_id": run_id, "visual_confidence": 0.9, "risk_score": 0.1, "knowledge_coverage": 1.0, "case_similarity": 0.9, "novelty_score": 0.1, "features": {"quality_status": "pass", "target_present": True, "case_applicable": True}}
    values.update(kwargs)
    return InspectionFeatures(**values)


class DiagnosisGraphTests(unittest.TestCase):
    def test_simple_route_has_aggregation_and_review(self):
        result = run_diagnosis_graph(make_features())
        self.assertEqual(result["result"]["route"], "simple_loop")
        self.assertEqual(result["result"]["initial_route"], "simple_loop")
        self.assertEqual(result["result"]["route_history"], ["simple_loop"])
        self.assertEqual(result["result"]["upgrade_count"], 0)
        self.assertIn("diagnosis_aggregator", [event["node"] for event in result["trace"]])
        decision = result["result"]["route_decision"]
        event = next(event for event in result["trace"] if event["node"] == "jev_router")
        self.assertEqual(decision["router_backend"], "deterministic_rule")
        self.assertEqual(event["router_backend"], decision["router_backend"])
        self.assertEqual(event["gate_version"], decision["gate_version"])
        self.assertEqual(event["policy_version"], decision["policy_version"])
        self.assertEqual(event["policy_hash"], decision["policy_hash"])

    def test_simple_route_skips_aggregator_model_call(self):
        result = run_diagnosis_graph(make_features(run_id="run-simple-skip"), context={"llm_profile": "fake_json"})
        gateway = next(item for item in result["result"]["evidence_bundle"] if item["agent_id"] == "llm_gateway")
        self.assertEqual(gateway["output"]["metadata"]["status"], "skipped_by_jev")
        self.assertFalse(gateway["output"]["content"])

    def test_simple_route_escalates_to_full_loop_after_evidence_gap(self):
        result = run_diagnosis_graph(
            make_features(run_id="run-simple-upgrade"),
            context={
                "llm_profile": "fake_json",
                # Deliberately omit source_id: the deterministic review guard
                # must find the missing visual lineage and escalate the route.
                "visual": {"confidence": 0.9},
            },
        )

        final = result["result"]
        self.assertEqual(final["initial_route"], "simple_loop")
        self.assertEqual(final["route"], "full_loop")
        self.assertEqual(final["route_history"], ["simple_loop", "full_loop"])
        self.assertEqual(final["upgrade_count"], 1)
        self.assertEqual(final["review"]["action"], "request_more_evidence")
        self.assertIn("missing_traceable_source_citation", final["review"]["issues"])
        self.assertTrue({"visual_review", "evidence_quality", "knowledge_rag", "case_compare", "process_state"}.issubset({item["agent_id"] for item in final["evidence_bundle"]}))
        trace_nodes = [event["node"] for event in result["trace"]]
        self.assertIn("simple_loop_upgrade", trace_nodes)
        self.assertIn("full_loop_dispatch", trace_nodes)

    def test_graph_reuses_matching_pre_route_without_second_adapter_call(self):
        class CountingAdapter:
            def __init__(self):
                self.calls = 0

            def predict(self, _features):
                self.calls += 1
                return {"route": "full_loop", "confidence": 0.95}

        features = make_features(run_id="run-route-cache", risk_score=0.1, case_similarity=0.2, features={"quality_status": "pass", "target_present": True, "case_applicable": False})
        adapter = CountingAdapter()
        router = JevRouter(adapter=adapter)
        pre_route = router.route(features)
        self.assertEqual(pre_route.router_backend, "laya_adapter")
        result = run_diagnosis_graph(
            features,
            router=router,
            context={"pre_route_decision": pre_route.to_dict(), "pre_route_features": features.to_dict(), "llm_profile": "fake_json"},
        )
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(result["result"]["route_decision"]["router_backend"], "laya_adapter")

    def test_graph_ignores_pre_route_for_different_feature_snapshot(self):
        class CountingAdapter:
            def __init__(self):
                self.calls = 0

            def predict(self, _features):
                self.calls += 1
                return {"route": "full_loop", "confidence": 0.95}

        original = make_features(run_id="run-stale-route-cache", risk_score=0.1, case_similarity=0.2, features={"quality_status": "pass", "target_present": True, "case_applicable": False})
        changed = make_features(run_id="run-stale-route-cache", risk_score=0.9, case_similarity=0.2, features={"quality_status": "pass", "target_present": True, "case_applicable": False})
        adapter = CountingAdapter()
        router = JevRouter(adapter=adapter)
        cached = JevRouter().route(original)
        result = run_diagnosis_graph(
            changed,
            router=router,
            context={"pre_route_decision": cached.to_dict(), "pre_route_features": original.to_dict(), "llm_profile": "fake_json"},
        )
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(result["result"]["route"], "full_loop")
        self.assertEqual(result["result"]["route_decision"]["router_backend"], "laya_adapter")

    def test_full_route_llm_aggregates_after_parallel_results(self):
        result = run_diagnosis_graph(
            make_features(run_id="run-full-aggregate", risk_score=0.9),
            context={"llm_profile": "fake_json", "workers": {"visual_review": lambda _: {"visual": {"label": "H"}}}},
        )
        gateway = next(item for item in result["result"]["evidence_bundle"] if item["agent_id"] == "llm_gateway")
        self.assertEqual(gateway["output"]["metadata"]["status"], "completed")
        self.assertIn("structured output", result["result"]["diagnosis"]["summary"])
        llm_event = next(event for event in result["trace"] if event["node"] == "diagnosis_aggregator_llm")
        self.assertEqual(llm_event["status"], "completed")

    def test_full_loop_keeps_failed_worker(self):
        def failing(_context):
            raise RuntimeError("synthetic failure")
        result = run_diagnosis_graph(make_features(risk_score=0.9), context={"workers": {"ok": lambda _: {"ok": True}, "failed": failing}})
        self.assertEqual(result["result"]["route"], "full_loop")
        self.assertTrue(any(item["status"] == "failed" for item in result["result"]["evidence_bundle"]))
        nodes = [event["node"] for event in result["trace"]]
        self.assertIn("full_loop_dispatch", nodes)
        self.assertIn("full_loop_fanout", nodes)
        self.assertTrue(any(node.startswith("full_loop_worker:") for node in nodes))
        fanin = next(event for event in result["trace"] if event["node"] == "full_loop_fanout")
        self.assertEqual(fanin["fanin"], "reducer_operator_add")
        self.assertEqual(fanin["child_count"], 2)

    def test_retake_route_is_explicit(self):
        result = run_diagnosis_graph(make_features(features={"quality_status": "retake", "target_present": True}))
        self.assertEqual(result["result"]["route"], "retake")

    def test_worker_retries_and_uses_agent_result_contract(self):
        calls = {"count": 0}

        def flaky(_context):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        result = run_diagnosis_graph(
            make_features(run_id="run-retry", risk_score=0.9),
            context={
                "worker_retries": 0,
                "worker_backoff_seconds": 0,
                "worker_policies": {
                    "flaky": {
                        "retries": 1,
                        "backoff_seconds": 0.02,
                        "retryable_errors": ["RuntimeError"],
                        "policy_id": "flaky-test-policy",
                    }
                },
                "workers": {"flaky": flaky},
            },
        )
        item = result["result"]["evidence_bundle"][0]
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["attempt"], 2)
        self.assertEqual(item["metadata"]["retry_count"], 1)
        self.assertEqual(item["metadata"]["retry_policy_id"], "flaky-test-policy")
        self.assertEqual(item["metadata"]["retry_delays_seconds"], [0.02])
        self.assertEqual(item["schema_version"], "agent-result-1")
        self.assertEqual(calls["count"], 2)

    def test_worker_policy_loads_configured_agent_timeout_and_identity(self):
        result = run_diagnosis_graph(
            make_features(run_id="run-worker-policy", risk_score=0.9),
            context={
                "worker_retries": 0,
                "workers": {"evidence_quality": lambda _: {"quality": "pass"}},
            },
        )
        item = result["result"]["evidence_bundle"][0]
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["metadata"]["timeout_s"], 20.0)
        self.assertEqual(item["metadata"]["retry_policy_id"], "evidence-quality-v1")
        self.assertEqual(item["metadata"]["max_retries"], 0)
        self.assertEqual(item["metadata"]["backoff_seconds"], 0.1)

    def test_worker_policy_does_not_retry_non_retryable_errors(self):
        calls = {"count": 0}

        def fail_fast(_context):
            calls["count"] += 1
            raise ValueError("invalid worker input")

        result = run_diagnosis_graph(
            make_features(run_id="run-non-retryable", risk_score=0.9),
            context={
                "workers": {"fail_fast": fail_fast},
                "worker_policies": {
                    "fail_fast": {
                        "retries": 2,
                        "retryable_errors": ["TimeoutError"],
                    }
                },
            },
        )
        item = result["result"]["evidence_bundle"][0]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_code"], "ValueError")
        self.assertEqual(item["attempt"], 1)
        self.assertEqual(item["metadata"]["retry_count"], 0)
        self.assertEqual(calls["count"], 1)

    def test_worker_timeout_is_retained_as_failed_result(self):
        def slow(context):
            time.sleep(0.2)
            return {"late": True, "cancelled": context["_cancel_event"].is_set()}

        result = run_diagnosis_graph(
            make_features(run_id="run-timeout", risk_score=0.9),
            context={"worker_timeout_s": 0.05, "worker_retries": 0, "workers": {"slow": slow}},
        )
        item = result["result"]["evidence_bundle"][0]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_code"], "timeout")
        self.assertEqual(item["attempt"], 1)
        self.assertEqual(item["metadata"]["timeout_s"], 0.05)
        self.assertEqual(item["metadata"]["execution_mode"], "cooperative_daemon_thread")


if __name__ == "__main__":
    unittest.main()
