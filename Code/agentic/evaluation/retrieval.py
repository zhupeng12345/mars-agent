"""Behavioral regression metrics for demo knowledge and case retrieval fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import csv
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic.platform.case_memory import SpecialCaseMemory
from agentic.platform.knowledge import VersionedKnowledgeBase


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: row must be an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def citation_integrity_issues(hit: dict[str, Any], knowledge: VersionedKnowledgeBase) -> list[str]:
    """Check that a retrieved citation resolves and preserves rule provenance."""
    source_id = str(hit.get("source_id", ""))
    if not source_id:
        return ["source_id_missing"]
    rule = knowledge.get_rule(source_id)
    if rule is None:
        return ["source_unresolved"]
    issues: list[str] = []
    expected = {
        "source": rule.source,
        "source_version": rule.source_version,
        "effective_date": rule.effective_date,
        "applicability": list(rule.applicability),
        "locator": f"rule:{rule.rule_id}:paragraph:1",
    }
    for field, value in expected.items():
        if hit.get(field) != value:
            issues.append(f"{field}_mismatch_or_missing")
    try:
        score = float(hit.get("score"))
    except (TypeError, ValueError):
        issues.append("retrieval_score_missing_or_invalid")
    else:
        if not math.isfinite(score) or score < 0:
            issues.append("retrieval_score_missing_or_invalid")
    return issues


def _seed_case(memory: SpecialCaseMemory, item: dict[str, Any], run_id: str) -> str:
    state = str(item.get("state", "candidate"))
    row = memory.add_candidate(
        source_id=str(item["source_id"]),
        features=dict(item.get("features") or {}),
        conclusion={"fixture": True},
        run_id=run_id,
        applicable_when=item.get("applicable_when", []),
    )
    transitions = {
        "candidate": (),
        "eval_only": ("eval_only",),
        "approved_special_case": ("eval_only", "approved_special_case"),
        "active": ("eval_only", "approved_special_case", "active"),
        "deprecated": ("deprecated",),
    }
    if state not in transitions:
        raise ValueError(f"unknown fixture case state: {state}")
    for target in transitions[state]:
        memory.transition(
            row["case_id"], target,
            reviewer_id="demo_fixture_reviewer",
            reason="deterministic retrieval behavior fixture",
            regression_passed=(target == "active"),
        )
    return row["case_id"]


def evaluate_knowledge_queries(query_path: Path, rules_path: Path, *, top_k: int = 3) -> dict[str, Any]:
    queries = _read_jsonl(Path(query_path))
    knowledge = VersionedKnowledgeBase(Path(rules_path))
    rows: list[dict[str, Any]] = []
    relevant_total = 0
    relevant_found = 0
    reciprocal_ranks: list[float] = []
    empty_total = 0
    empty_correct = 0
    citation_audit: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query in queries:
        if query.get("fixture_type") != "demo_fixture":
            raise ValueError("retrieval regression runner only accepts demo_fixture rows")
        query_id = str(query.get("query_id", ""))
        if not query_id or query_id in seen_ids or not str(query.get("query", "")).strip():
            raise ValueError("knowledge fixture query_id must be unique and query text non-empty")
        seen_ids.add(query_id)
        if not isinstance(query.get("expected_source_ids"), list):
            raise ValueError(f"{query_id}: expected_source_ids must be a list")
        expected = [str(item) for item in query.get("expected_source_ids", [])]
        expected_behavior = "hit" if expected else "empty"
        if query.get("expected_behavior") != expected_behavior:
            raise ValueError(f"{query_id}: expected_behavior conflicts with expected_source_ids")
        actual = knowledge.search_bm25(
            str(query.get("query", "")),
            top_k=top_k,
            as_of=query.get("as_of"),
            applicability=query.get("applicability", []),
        )
        actual_ids = [str(item.get("source_id")) for item in actual]
        query_citations = []
        for hit in actual:
            issues = citation_integrity_issues(hit, knowledge)
            audit_row = {
                "query_id": query_id,
                "source_id": hit.get("source_id"),
                "source_version": hit.get("source_version"),
                "locator": hit.get("locator"),
                "valid": not issues,
                "issues": issues,
            }
            query_citations.append(audit_row)
            citation_audit.append(audit_row)
        relevant = [item for item in actual_ids if item in set(expected)]
        relevant_total += len(expected)
        relevant_found += len(relevant)
        if relevant:
            reciprocal_ranks.append(1.0 / (actual_ids.index(relevant[0]) + 1))
            passed = True
        elif expected:
            reciprocal_ranks.append(0.0)
            passed = False
        else:
            empty_total += 1
            passed = not actual_ids
            empty_correct += int(passed)
        rows.append({
            "query_id": query.get("query_id"),
            "expected_source_ids": expected,
            "actual_source_ids": actual_ids,
            "expected_behavior": query.get("expected_behavior"),
            "passed": passed,
            "citation_integrity_passed": all(item["valid"] for item in query_citations) if query_citations else None,
            "citation_integrity_issues": [issue for item in query_citations for issue in item["issues"]],
            "status": "demo_fixture",
        })
    citation_count = len(citation_audit)
    valid_citation_count = sum(item["valid"] for item in citation_audit)
    return {
        "status": "demo_fixture_behavioral_regression",
        "query_count": len(rows),
        "top_k": top_k,
        "fixture_pass_rate": sum(row["passed"] for row in rows) / len(rows) if rows else None,
        "recall_at_k": relevant_found / relevant_total if relevant_total else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "empty_expected_count": empty_total,
        "empty_result_accuracy": empty_correct / empty_total if empty_total else None,
        "citation_integrity": {
            "status": "evaluated" if citation_count else "not_run",
            "citation_count": citation_count,
            "valid_citation_count": valid_citation_count,
            "invalid_citation_count": citation_count - valid_citation_count,
            "pass_rate": valid_citation_count / citation_count if citation_count else None,
            "claimable": False,
            "semantic_support_status": "not_evaluated_without_human_entailment_labels",
            "rows": citation_audit,
        },
        "rows": rows,
    }


def evaluate_case_queries(query_path: Path) -> dict[str, Any]:
    queries = _read_jsonl(Path(query_path))
    rows: list[dict[str, Any]] = []
    expected_match_count = 0
    expected_match_hits = 0
    expected_no_match_count = 0
    expected_no_match_correct = 0
    seen_ids: set[str] = set()
    for query in queries:
        if query.get("fixture_type") != "demo_fixture":
            raise ValueError("retrieval regression runner only accepts demo_fixture rows")
        query_id = str(query.get("query_id", ""))
        if not query_id or query_id in seen_ids or not str(query.get("query", "")).strip():
            raise ValueError("case fixture query_id must be unique and query text non-empty")
        seen_ids.add(query_id)
        with tempfile.TemporaryDirectory(prefix="mars-case-retrieval-eval-") as directory:
            memory = SpecialCaseMemory(Path(directory))
            expected_case_ids: set[str] = set()
            expected_sources = set(query.get("expected_case_sources", []))
            for seed in query.get("seed_cases", []):
                case_id = _seed_case(memory, seed, str(query.get("query_id", "fixture")))
                if seed.get("source_id") in expected_sources:
                    expected_case_ids.add(case_id)
            matches = memory.search(
                dict(query.get("query_features") or {}),
                applicability=query.get("query_applicability", []),
                top_k=10,
                min_feature_coverage=float(query.get("min_feature_coverage", 0.5)),
                min_similarity=float(query.get("min_similarity", 0.5)),
            )
            actual_ids = [str(item["case_id"]) for item in matches]
            expected_outcome = str(query.get("expected_outcome", "no_match"))
            matched_expected = bool(expected_case_ids.intersection(actual_ids))
            if expected_outcome == "match":
                expected_match_count += 1
                expected_match_hits += int(matched_expected)
                passed = matched_expected
            elif expected_outcome == "no_match":
                expected_no_match_count += 1
                passed = not actual_ids
                expected_no_match_correct += int(passed)
            else:
                raise ValueError(f"invalid expected_outcome for {query.get('query_id')}: {expected_outcome}")
            rows.append({
                "query_id": query.get("query_id"),
                "expected_outcome": expected_outcome,
                "expected_case_ids": sorted(expected_case_ids),
                "actual_case_ids": actual_ids,
                "similarities": [round(float(item["similarity"]), 6) for item in matches],
                "feature_coverages": [round(float(item["feature_coverage"]), 6) for item in matches],
                "passed": passed,
                "status": "demo_fixture",
            })
    return {
        "status": "demo_fixture_behavioral_regression",
        "query_count": len(rows),
        "fixture_pass_rate": sum(row["passed"] for row in rows) / len(rows) if rows else None,
        "expected_match_count": expected_match_count,
        "expected_match_hit_rate": expected_match_hits / expected_match_count if expected_match_count else None,
        "expected_no_match_count": expected_no_match_count,
        "expected_no_match_correct_rate": expected_no_match_correct / expected_no_match_count if expected_no_match_count else None,
        "rows": rows,
    }


def evaluate_retrieval(root: Path, *, top_k: int = 3) -> dict[str, Any]:
    root = Path(root)
    evaluation = root / "data" / "evaluation"
    knowledge_queries = evaluation / "knowledge_queries_demo.jsonl"
    case_queries = evaluation / "case_queries_demo.jsonl"
    rules = root / "knowledge" / "rules_v1.jsonl"
    missing_inputs = [str(path) for path in (knowledge_queries, case_queries, rules) if not path.exists()]
    if missing_inputs:
        return {
            "schema_version": "retrieval-evaluation-1",
            "evaluation_kind": "demo_fixture_behavioral_regression",
            "claimable_domain_accuracy": False,
            "complete_fixture_coverage": False,
            "status": "missing_inputs",
            "missing_inputs": missing_inputs,
            "knowledge": {"status": "not_run", "query_count": 0, "fixture_pass_rate": None, "citation_integrity": {"status": "not_run", "citation_count": 0, "valid_citation_count": 0, "invalid_citation_count": 0, "pass_rate": None, "claimable": False, "semantic_support_status": "not_evaluated_without_human_entailment_labels", "rows": []}, "rows": []},
            "cases": {"status": "not_run", "query_count": 0, "fixture_pass_rate": None, "rows": []},
        }
    knowledge_report = evaluate_knowledge_queries(knowledge_queries, rules, top_k=top_k)
    case_report = evaluate_case_queries(case_queries)
    return {
        "schema_version": "retrieval-evaluation-1",
        "status": "complete" if knowledge_report["query_count"] >= 10 and case_report["query_count"] >= 10 else "incomplete_fixture_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "demo_fixture_behavioral_regression",
        "claimable_domain_accuracy": False,
        "required_query_count_per_area": 10,
        "complete_fixture_coverage": knowledge_report["query_count"] >= 10 and case_report["query_count"] >= 10,
        "provenance": {
            "knowledge_queries_sha256": _sha256(knowledge_queries),
            "case_queries_sha256": _sha256(case_queries),
            "knowledge_rules_sha256": _sha256(rules),
        },
        "knowledge": knowledge_report,
        "cases": case_report,
        "claim_policy": "These fixtures validate retrieval filters and lifecycle behavior; they do not measure expert relevance or domain accuracy.",
    }


def write_artifacts(report: dict[str, Any], output: Path) -> dict[str, str]:
    output = Path(output)
    md_path = output.with_suffix(".md")
    csv_path = output.with_name(output.stem + "_metrics.csv")
    failures_path = output.with_name(output.stem + "_failures.jsonl")
    lines = [
        "# Retrieval fixture evaluation", "",
        "Status: `demo_fixture_behavioral_regression`", "",
        "This is a behavior regression suite, not expert relevance or domain accuracy.", "",
        "| Area | Queries | Fixture pass rate | Main metrics |", "|---|---:|---:|---|",
    ]
    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.writer(csv_buffer, lineterminator="\n")
    csv_writer.writerow(("area", "metric", "value"))
    failures = []
    for area in ("knowledge", "cases"):
        item = report[area]
        main_metrics = ", ".join(f"{name}={item.get(name)}" for name in item if name not in {"rows", "status", "query_count", "fixture_pass_rate"})
        lines.append(f"| {area} | {item['query_count']} | {item['fixture_pass_rate']} | {main_metrics} |")
        for name, value in item.items():
            if name not in {"rows", "status"}:
                csv_writer.writerow((area, name, json.dumps(value, ensure_ascii=False)))
        failures.extend({"area": area, **row} for row in item["rows"] if not row["passed"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")
    failures_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    return {"markdown": str(md_path), "csv": str(csv_path), "failures": str(failures_path)}


__all__ = ["evaluate_case_queries", "evaluate_knowledge_queries", "evaluate_retrieval", "write_artifacts"]
