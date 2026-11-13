"""Optional Laya adapter; absence or abstention is a safe fallback."""

from __future__ import annotations

from typing import Any

from .calibration import apply_temperature, fit_route_calibration


class LayaAdapter:
    def __init__(self, router: Any = None, *, model: str = "typed-decisions"):
        self.router = router
        self.model = model
        self.calibration: dict[str, Any] = {"temperature": 1.0, "schema_version": "jev-calibration-unfit"}

    def fit(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit a compatible backend and calibrator without hiding failures."""
        calibration = fit_route_calibration(rows)
        if self.router is None:
            try:
                from laya import Router  # type: ignore
                self.router = Router()
            except Exception as exc:
                raise RuntimeError("Laya router is unavailable; no model fit was performed") from exc
        fit_method = getattr(self.router, "fit", None)
        if not callable(fit_method):
            raise NotImplementedError("the configured Laya Router backend does not expose fit(); model training is unsupported")
        try:
            fit_method(rows)
        except Exception as exc:
            raise RuntimeError("Laya router fitting failed; confidence calibration was not applied") from exc
        self.calibration = calibration
        return {**self.calibration, "model_fit_status": "fitted"}

    def calibrate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        self.calibration = fit_route_calibration(rows)
        return dict(self.calibration)

    def predict(self, features: Any) -> dict[str, Any]:
        if self.router is None:
            try:
                from laya import Router  # type: ignore
                self.router = Router()
            except Exception:
                return {"abstain": True, "route": "human", "confidence": 0.0, "reason": "laya_unavailable"}
        try:
            state = features.to_dict() if hasattr(features, "to_dict") else dict(features)
            questions = {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the safest diagnostic loop for this inspection task.",
                    "criteria": {
                        "simple_loop": "low-risk, stable evidence, approved case or rule match",
                        "full_loop": "complex, conflicting, high-risk, or evidence-intensive task",
                        "retake": "media quality is insufficient and a new capture is required",
                        "human": "direct operator review is required",
                    },
                }
            }
            result = self.router.predict(state, questions=questions, model=self.model)
            if hasattr(result, "to_dict"):
                result = result.to_dict()
            if isinstance(result, dict):
                if result.get("abstain") or result.get("unbound"):
                    return {"abstain": True, "route": "human", "confidence": 0.0, "reason": "laya_abstain"}
                answer = result.get("answers", {}).get("route", {}) if isinstance(result.get("answers"), dict) else {}
                route = answer.get("choice") or answer.get("value") or result.get("route") or result.get("choice")
                raw_score = float(
                    answer.get("answer_confidence", answer.get("confidence", result.get("confidence", result.get("score", 0.0))))
                    or 0.0
                )
                return {"route": route, "confidence": apply_temperature(raw_score, float(self.calibration.get("temperature", 1.0))), "raw": result, "calibration": self.calibration}
        except Exception as exc:
            return {"abstain": True, "route": "human", "confidence": 0.0, "reason": f"laya_error:{exc}"}
        return {"abstain": True, "route": "human", "confidence": 0.0, "reason": "laya_invalid_response"}
