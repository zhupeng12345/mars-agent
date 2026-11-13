"""Reproducible synthetic regression for the pre-diagnosis quality gate."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agentic.agents.image_quality import ImageQualityAgent


def _synthetic_frames() -> list[tuple[str, bool, np.ndarray]]:
    rng = np.random.default_rng(20261109)
    clear = rng.integers(30, 170, size=(512, 512, 3), dtype=np.uint8)
    cv2.rectangle(clear, (48, 42), (462, 468), (70, 100, 145), -1)
    cv2.circle(clear, (256, 245), 118, (40, 130, 245), -1)
    cv2.circle(clear, (256, 245), 82, (40, 220, 255), 7)
    for index in range(18, 500, 19):
        cv2.line(clear, (index, 0), (511 - index, 511), (170, 180, 190), 2)

    blurred = cv2.GaussianBlur(clear, (0, 0), 14)
    underexposed = np.clip(clear.astype(np.float32) * 0.08, 0, 255).astype(np.uint8)
    overexposed = np.full_like(clear, 255)
    low_resolution = cv2.resize(clear, (64, 64), interpolation=cv2.INTER_AREA)
    black_frame = np.zeros_like(clear)
    return [
        ("clear_target", False, clear),
        ("blurred_target", True, blurred),
        ("underexposed_target", True, underexposed),
        ("overexposed_frame", True, overexposed),
        ("low_resolution_target", True, low_resolution),
        ("missing_target", True, black_frame),
    ]


def evaluate_quality_gate(*, min_failure_detection_rate: float = 0.9) -> dict[str, Any]:
    """Run deterministic synthetic fixtures; this is regression, not accuracy."""
    if not 0.0 <= min_failure_detection_rate <= 1.0:
        raise ValueError("min_failure_detection_rate must be within [0, 1]")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mars-image-quality-") as tmp:
        root = Path(tmp)
        agent = ImageQualityAgent()
        for name, expected_failure, image in _synthetic_frames():
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(f"failed to encode fixture: {name}")
            payload = encoded.tobytes()
            path = root / f"{name}.png"
            path.write_bytes(payload)
            result = agent.run(path, run_id=f"quality-fixture-{name}")
            status = str(result["quality_status"])
            caught = status in {"retake", "human"}
            rows.append({
                "fixture_id": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_failure": expected_failure,
                "quality_status": status,
                "failure_caught": caught if expected_failure else None,
                "false_retake": (status == "retake") if not expected_failure else None,
                "issues": list(result.get("issues", [])),
                "threshold_version": result.get("threshold_version"),
            })

    failure_rows = [row for row in rows if row["expected_failure"]]
    valid_rows = [row for row in rows if not row["expected_failure"]]
    caught_count = sum(bool(row["failure_caught"]) for row in failure_rows)
    false_retake_count = sum(bool(row["false_retake"]) for row in valid_rows)
    detection_rate = caught_count / len(failure_rows) if failure_rows else None
    passed = detection_rate is not None and detection_rate >= min_failure_detection_rate and false_retake_count == 0
    return {
        "schema_version": "image-quality-fixture-evaluation-1",
        "status": "passed" if passed else "failed",
        "evaluation_kind": "synthetic_quality_gate_regression",
        "claimable_domain_accuracy": False,
        "min_failure_detection_rate": min_failure_detection_rate,
        "failure_sample_count": len(failure_rows),
        "failure_detected_count": caught_count,
        "failure_detection_rate": detection_rate,
        "valid_sample_count": len(valid_rows),
        "valid_false_retake_count": false_retake_count,
        "fixtures": rows,
    }


__all__ = ["evaluate_quality_gate"]
