"""Controlled special-case lifecycle and similarity gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from .memory import _exclusive_file_lock


CASE_STATES = ("candidate", "eval_only", "approved_special_case", "active", "deprecated")


class SpecialCaseMemory:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "special_cases.jsonl"

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _snapshot(self) -> list[dict[str, Any]]:
        with _exclusive_file_lock(self.path):
            return self._read()

    def _append(self, row: dict[str, Any]) -> None:
        with _exclusive_file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def case_id(source_id: str, features: dict[str, Any]) -> str:
        payload = json.dumps({"source_id": source_id, "features": features}, ensure_ascii=False, sort_keys=True)
        return "case-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def add_candidate(self, *, source_id: str, features: dict[str, Any], conclusion: dict[str, Any], run_id: str, applicable_when: Iterable[str] = ()) -> dict[str, Any]:
        row = {"case_id": self.case_id(source_id, features), "state": "candidate", "source_id": source_id, "features": features, "conclusion": conclusion, "run_id": run_id, "applicable_when": list(applicable_when), "history": [{"state": "candidate", "run_id": run_id}]}
        self._append(row)
        return row

    def update_candidate_conclusion(self, case_id: str, conclusion: dict[str, Any], *, reviewer_id: str = "", reason: str = "") -> dict[str, Any]:
        """Append an auditable replacement for a staged candidate conclusion.

        Case rows are append-only snapshots. The latest snapshot for a case ID
        is authoritative, so a human amendment cannot leave the searchable
        candidate pointing at the pre-review diagnosis.
        """
        with _exclusive_file_lock(self.path):
            rows = self._read()
            matches = [row for row in rows if row.get("case_id") == case_id]
            if not matches:
                raise KeyError(case_id)
            current = dict(matches[-1])
            if current.get("state") != "candidate":
                raise ValueError("only candidate cases can be amended")
            if current.get("conclusion") == conclusion:
                return current
            current["conclusion"] = dict(conclusion)
            current.setdefault("history", []).append({
                "event": "conclusion_amended",
                "reviewer_id": reviewer_id,
                "reason": reason,
            })
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(current, ensure_ascii=False, sort_keys=True) + "\n")
            return current

    def transition(self, case_id: str, state: str, *, reviewer_id: str, reason: str, regression_passed: bool = False) -> dict[str, Any]:
        if state not in CASE_STATES:
            raise ValueError(f"invalid case state: {state}")
        allowed = {"candidate": {"eval_only", "deprecated"}, "eval_only": {"approved_special_case", "deprecated"}, "approved_special_case": {"active", "deprecated"}, "active": {"deprecated"}, "deprecated": set()}
        with _exclusive_file_lock(self.path):
            rows = self._read()
            matches = [row for row in rows if row.get("case_id") == case_id]
            if not matches:
                raise KeyError(case_id)
            row = dict(matches[-1])
            current = row.get("state")
            if state not in allowed.get(current, set()):
                raise ValueError(f"invalid case transition: {current}->{state}")
            if state == "active" and (not regression_passed or not reviewer_id.strip()):
                raise ValueError("active requires reviewer and frozen regression")
            row["state"] = state
            row.setdefault("history", []).append({"state": state, "reviewer_id": reviewer_id, "reason": reason, "regression_passed": regression_passed})
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row

    @staticmethod
    def _numeric_features(features: Any) -> dict[str, float]:
        if not isinstance(features, dict):
            return {}
        result = {}
        for key, value in features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                result[str(key)] = numeric
        return result

    def search(
        self,
        features: dict[str, float],
        *,
        applicability: Iterable[str] = (),
        top_k: int = 5,
        min_feature_coverage: float = 0.5,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        if not math.isfinite(float(min_feature_coverage)) or not 0.0 < min_feature_coverage <= 1.0:
            raise ValueError("min_feature_coverage must be in (0, 1]")
        if not math.isfinite(float(min_similarity)) or not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        query_features = self._numeric_features(features)
        if not query_features:
            return []
        allowed = set(applicability)
        rows = []
        # The store is append-only for auditability. Collapse duplicate case
        # IDs to their latest snapshot before similarity search.
        latest_by_case: dict[str, dict[str, Any]] = {}
        for row in self._snapshot():
            case_id = str(row.get("case_id", ""))
            if case_id:
                latest_by_case[case_id] = row
        for row in latest_by_case.values():
            if row.get("state") != "active":
                continue
            conditions = set(row.get("applicable_when", []))
            if conditions and not conditions.issubset(allowed):
                continue
            stored = self._numeric_features(row.get("features", {}))
            keys = set(query_features).intersection(stored)
            coverage = len(keys) / max(len(query_features), len(stored), 1)
            if not keys or coverage < min_feature_coverage:
                continue
            distance = sum(abs(query_features[key] - stored[key]) for key in keys) / len(keys)
            similarity = 1.0 / (1.0 + distance)
            if similarity >= min_similarity:
                rows.append((similarity, coverage, row))
        rows.sort(key=lambda item: (-item[0], -item[1], item[2].get("case_id", "")))
        return [
            {
                "case_id": row["case_id"],
                "source_id": row.get("source_id"),
                "similarity": score,
                "feature_coverage": coverage,
                "applicable": True,
                "conclusion": row.get("conclusion", {}),
            }
            for score, coverage, row in rows[:max(1, top_k)]
        ]
