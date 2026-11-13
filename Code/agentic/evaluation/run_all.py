"""Regenerate the frozen evaluation summary without model or network calls."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

from agentic.evaluation.metrics import end_to_end_latency_metrics, evidence_metrics, latency_metrics, multi_agent_flow_metrics, route_flow_metrics, route_metrics, rsi_runtime_policy_metrics, system_health_metrics, visual_data_coverage, worker_execution_metrics
from agentic.evaluation.retrieval import evaluate_retrieval
from agentic.evaluation.visual_eval import evaluate_visual_manifest
from agentic.evaluation.ablation import build_ablation_report
from agentic.evaluation.image_quality import evaluate_quality_gate
from agentic.scripts.freeze_evaluation import freeze as freeze_evaluation
from agentic.scripts.route_smoke import replay
from agentic.scripts.validate_knowledge_pack import validate_knowledge_pack


def _jsonl(path: Path, errors: list[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if errors is not None:
                errors.append(f"{path.name}:invalid_json_line:{line_number}")
            continue
        if not isinstance(value, dict):
            if errors is not None:
                errors.append(f"{path.name}:row_not_object:{line_number}")
            continue
        rows.append(value)
    return rows


def _load_optional_report(path: Path, *, default_status: str = "not_run") -> dict[str, Any]:
    """Load an optional report without turning a missing input into an empty result."""
    path = Path(path)
    if not path.exists():
        return {"status": default_status, "report_path": str(path), "reason": "file_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid_report", "report_path": str(path), "reason": "invalid_json"}
    if not isinstance(payload, dict):
        return {"status": "invalid_report", "report_path": str(path), "reason": "report_not_object"}
    payload.setdefault("report_path", str(path))
    payload.setdefault("status", "loaded")
    return payload


def _select_run_records(rows: list[dict[str, Any]], run_ids: set[str] | None) -> list[dict[str, Any]]:
    if run_ids is None:
        return rows
    return [row for row in rows if str(row.get("run_id", "")) in run_ids]


def _runtime_scope(
    run_ids: set[str] | None,
    events: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    if run_ids is None:
        has_history = bool(events or runs or reviews)
        return {
            "scope": "unscoped_history_mixed" if has_history else "empty",
            "selected_run_ids": [],
            "matched_run_ids": [],
            "missing_run_ids": [],
            "claimable": False,
            "claimability_reason": "Pass --run-id to evaluate an explicitly selected run; unscoped runtime files may combine historical runs.",
        }
    run_record_ids = {str(row.get("run_id", "")) for row in runs if row.get("run_id")}
    observed_ids = {
        str(row.get("run_id", ""))
        for row in [*events, *runs, *reviews]
        if row.get("run_id")
    }
    matched = sorted(run_ids & observed_ids)
    missing = sorted(run_ids - run_record_ids)
    claimable = bool(run_ids) and not missing
    return {
        "scope": "explicit_run_ids" if claimable else "explicit_run_ids_incomplete",
        "selected_run_ids": sorted(run_ids),
        "matched_run_ids": matched,
        "missing_run_ids": missing,
        "claimable": claimable,
        "claimability_reason": None if claimable else "Every selected run_id must have a matching run record before runtime metrics are claimable.",
    }


def _evaluation_freeze_summary(root: Path) -> dict[str, Any]:
    report = freeze_evaluation(root)
    files = report.get("files", {})
    missing = sorted(name for name, item in files.items() if item.get("status") == "missing")
    invalid = sorted(name for name, item in files.items() if item.get("status") == "invalid")
    frozen = bool(report.get("frozen"))
    if frozen:
        status = "frozen"
    elif missing and invalid:
        status = "incomplete"
    elif missing:
        status = "missing_manifests"
    else:
        status = "invalid_manifests"
    return {
        "schema_version": report.get("schema_version"),
        "status": status,
        "frozen": frozen,
        "evaluation_data_eligible": frozen,
        "claimable": False,
        "claimability_reason": "Dataset freezing validates evaluation data readiness; metric claims require results evaluated on the frozen manifests and matching run provenance.",
        "missing_manifests": missing,
        "invalid_manifests": invalid,
        "manifest_details": {
            name: {
                "status": item.get("status"),
                "record_count": item.get("records", 0),
                "valid_record_count": item.get("valid_records", 0),
                "minimum_record_count": item.get("minimum_records"),
                "sha256": item.get("sha256"),
                "errors": item.get("errors", []),
                "invalid_error_counts": item.get("invalid_error_counts", {}),
                "invalid_row_examples": item.get("invalid_row_examples", [])[:5],
            }
            for name, item in files.items()
        },
    }


def run_all(
    root: Path,
    *,
    runtime_dir: Path | None = None,
    runtime_run_ids: Iterable[str] | None = None,
    process_evaluation_path: Path | None = None,
    process_attribution_path: Path | None = None,
    process_repeats_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    visual_path = root / "data" / "evaluation" / "visual_manifest.jsonl"
    route_path = root / "data" / "evaluation" / "route_manifest.jsonl"
    process_manifest = root / "data" / "evaluation" / "process_manifest.jsonl"
    if not process_manifest.exists():
        process_manifest = root / "data" / "evaluation" / "process_manifest.json"
    route = replay(route_path, execute=False) if route_path.exists() else {"rows": [], "fixture_count": 0, "passed": False}
    runtime_root = Path(runtime_dir or os.environ.get("MARS_AGENT_RUNTIME_DIR", str(root / "runtime_data"))).expanduser()
    selected_run_ids = {str(item).strip() for item in runtime_run_ids or () if str(item).strip()} or None
    runtime_parse_errors: list[str] = []
    all_events = _jsonl(runtime_root / "events.jsonl", runtime_parse_errors)
    all_run_records = _jsonl(runtime_root / "runs.jsonl", runtime_parse_errors)
    all_review_records = _jsonl(runtime_root / "hitl_audit.jsonl", runtime_parse_errors)
    run_events = _select_run_records(all_events, selected_run_ids)
    run_records = _select_run_records(all_run_records, selected_run_ids)
    review_records = _select_run_records(all_review_records, selected_run_ids)
    runtime_scope = _runtime_scope(selected_run_ids, run_events, run_records, review_records)
    image_training_path = runtime_root / "visual_training" / "image_training_report.json"
    process_evaluation_path = Path(process_evaluation_path) if process_evaluation_path else runtime_root / "process_eval.json"
    process_attribution_path = Path(process_attribution_path) if process_attribution_path else runtime_root / "process_feature_attribution" / "feature_attribution_report.json"
    process_repeats_path = Path(process_repeats_path) if process_repeats_path else runtime_root / "process_repeats" / "repeated_process_evaluation.json"
    completed_records = [row for row in run_records if isinstance(row.get("result"), dict)]
    ablation = build_ablation_report(root)
    visual_rows = _jsonl(visual_path)
    visual_summary = visual_data_coverage(visual_rows)
    visual_summary["label_metrics"] = evaluate_visual_manifest(visual_rows)
    image_training = _load_optional_report(image_training_path)
    process_evaluation = _load_optional_report(process_evaluation_path)
    process_attribution = _load_optional_report(process_attribution_path)
    process_repeats = _load_optional_report(process_repeats_path)
    retrieval = evaluate_retrieval(root)
    evaluation_freeze = _evaluation_freeze_summary(root)
    knowledge_pack = validate_knowledge_pack(root / "agentic" / "knowledge" / "rules_v1.jsonl")
    quality_gate = evaluate_quality_gate()
    node_latency = latency_metrics(run_events)
    end_to_end_latency = end_to_end_latency_metrics(run_records, selected_run_ids=selected_run_ids)
    end_to_end_latency["claimable"] = bool(
        selected_run_ids
        and runtime_scope["claimable"]
        and end_to_end_latency["count"] == len(selected_run_ids)
        and not end_to_end_latency["missing_timing_run_ids"]
    )
    node_latency["claimable"] = bool(runtime_scope["claimable"] and node_latency["count"] > 0)
    system_summary = {
        "node_latency": node_latency,
        "end_to_end_latency": end_to_end_latency,
        "runtime_scope": runtime_scope,
    } | system_health_metrics(run_events, run_records, review_records)
    return {
        "schema_version": "evaluation-report-1",
        "data_version": "local-workspace",
        "visual": visual_summary,
        "visual_image_level_baseline": image_training,
        "image_quality_gate": quality_gate,
        "process": {"manifest": str(process_manifest), "alignment_quality": "unknown", "video_classification_improvement_claimed": False, "independent_baseline_evaluation": process_evaluation, "feature_attribution": process_attribution, "repeated_seed_evaluation": process_repeats},
        "jev": route_metrics(route.get("rows", [])) | {"fixture_replay_passed": route.get("passed", False)},
        "route_flow": route_flow_metrics([row.get("result", {}) for row in completed_records]),
        "rsi_runtime_policy": rsi_runtime_policy_metrics(run_records),
        "evidence": evidence_metrics([row["result"] for row in completed_records]),
        "multi_agent": multi_agent_flow_metrics([row["result"] for row in completed_records]),
        "worker_execution": worker_execution_metrics([row["result"] for row in completed_records]),
        "retrieval": retrieval,
        "knowledge_pack_validation": knowledge_pack,
        "evaluation_freeze": evaluation_freeze,
        "system": system_summary,
        "input_integrity": {
            "runtime_jsonl_parse_errors": runtime_parse_errors,
            "status": "passed" if not runtime_parse_errors else "invalid_runtime_records",
            "claimable": not runtime_parse_errors,
        },
        "ablation_status": {
            "matrix_version": ablation["schema_version"],
            "completed_count": ablation["completed_count"],
            "structure_only_count": ablation["structure_only_count"],
            "not_run_count": ablation["not_run_count"],
            "report": ablation,
        },
        "claims": {"visual_accuracy": "unavailable_without_expert_video_labels", "real_superheat_measurement": False},
    }


def _scalar_rows(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_scalar_rows(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


def write_report_artifacts(
    root: Path,
    report: dict[str, Any],
    output: Path,
    *,
    runtime_dir: Path | None = None,
    runtime_run_ids: Iterable[str] | None = None,
) -> dict[str, str | None]:
    """Write table, failure-sample and graph artifacts beside the JSON report."""
    root = Path(root)
    output = Path(output)
    md_path = output.with_suffix(".md")
    csv_path = output.with_name(output.stem + "_metrics.csv")
    failures_path = output.with_name(output.stem + "_failures.jsonl")
    png_path = output.with_name(output.stem + "_summary.png")
    scalars = _scalar_rows(report)
    md_lines = ["# Evaluation report", "", f"- schema: `{report.get('schema_version', '')}`", f"- data: `{report.get('data_version', '')}`", "", "| Metric | Value |", "|---|---:|"]
    md_lines.extend(f"| `{key}` | `{value}` |" for key, value in scalars if key and not isinstance(value, (dict, list)))
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.writer(csv_buffer, lineterminator="\n")
    csv_writer.writerow(("metric", "value"))
    for key, value in scalars:
        if key and not isinstance(value, (dict, list)):
            csv_writer.writerow((key, json.dumps(value, ensure_ascii=False)))
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")

    route_path = root / "data" / "evaluation" / "route_manifest.jsonl"
    route = replay(route_path, execute=False) if route_path.exists() else {"rows": []}
    failures = [row for row in route.get("rows", []) if not row.get("passed", row.get("route") == row.get("expected_route"))]
    selected_run_ids = {str(item).strip() for item in runtime_run_ids or () if str(item).strip()} or None
    runtime_root = Path(runtime_dir or os.environ.get("MARS_AGENT_RUNTIME_DIR", str(root / "runtime_data"))).expanduser()
    runtime_events = _select_run_records(_jsonl(runtime_root / "events.jsonl"), selected_run_ids)
    failures.extend(row for row in runtime_events if row.get("status") in {"failed", "error", "timeout", "timed_out"})
    failures_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8")

    graph_status = "generated"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["fixture_match", "simple_fixture_recall", "full_fixture_recall"]
        values = [report.get("jev", {}).get("fixture_route_match_rate"), report.get("jev", {}).get("simple_fixture_recall"), report.get("jev", {}).get("complex_fixture_full_recall")]
        pairs = [(label, value) for label, value in zip(labels, values) if isinstance(value, (int, float))]
        fig, axis = plt.subplots(figsize=(7, 4), dpi=150)
        if pairs:
            axis.bar([item[0] for item in pairs], [item[1] for item in pairs], color="#285ea8")
            axis.set_ylim(0, 1)
            axis.set_ylabel("score")
        else:
            axis.text(0.5, 0.5, "No claimable model metrics", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title("JEV policy fixture regression (not model accuracy)")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - dependency/GUI fallback
        graph_status = f"not_generated:{type(exc).__name__}"
        png_path = None
    return {"markdown": str(md_path), "csv": str(csv_path), "failures": str(failures_path), "png": str(png_path) if png_path else None, "graph_status": graph_status}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model-free frozen evaluation summary")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    # Keep the final regression command reproducible without requiring a
    # generated path on every invocation.  An explicit --output still wins.
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-dir", type=Path, default=None, help="runtime directory to summarize; defaults to MARS_AGENT_RUNTIME_DIR/runtime_data")
    parser.add_argument("--run-id", action="append", default=None, help="include only this run_id; repeat to select a paired cohort")
    parser.add_argument("--process-eval", type=Path, default=None, help="include an explicit independent Process State baseline evaluation JSON")
    parser.add_argument("--process-attribution", type=Path, default=None, help="include a process feature attribution report JSON")
    parser.add_argument("--process-repeats", type=Path, default=None, help="include a repeated-seed independent process baseline report JSON")
    args = parser.parse_args(argv)
    report = run_all(args.root, runtime_dir=args.runtime_dir, runtime_run_ids=args.run_id, process_evaluation_path=args.process_eval, process_attribution_path=args.process_attribution, process_repeats_path=args.process_repeats)
    output = args.output or (args.root / "runtime_data" / "evaluation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = write_report_artifacts(args.root, report, output, runtime_dir=args.runtime_dir, runtime_run_ids=args.run_id)
    print(json.dumps({"output": str(output), "route_fixtures_passed": report["jev"]["fixture_replay_passed"], "quality_gate": report["image_quality_gate"]["status"], "visual_status": report["visual"]["metric_status"], "runtime_scope": report["system"]["runtime_scope"], "artifacts": artifacts}, ensure_ascii=False))
    return 0 if report["jev"]["fixture_replay_passed"] and report["image_quality_gate"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
