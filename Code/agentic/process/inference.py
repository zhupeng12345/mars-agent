"""Checkpoint inference for independent process-event probabilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .baselines import build_process_baseline
from .data import ProcessNormalizer


class ProcessCheckpointPredictor:
    """Load the model and the exact train-fitted preprocessing together."""

    def __init__(self, checkpoint: Path | str) -> None:
        self.path = Path(checkpoint).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != "process-checkpoint-2":
            raise ValueError("unsupported process checkpoint; train_process v2 metadata required")
        self.feature_columns = tuple(payload["feature_columns"])
        self.history_steps = int(payload["temporal_contract"]["history_steps"])
        self.future_minutes = int(payload["temporal_contract"]["future_steps"]) * int(payload["temporal_contract"]["sample_minutes"])
        self.normalizer = ProcessNormalizer.from_dict(payload["normalization"])
        config = payload["model_config"]
        if len(self.feature_columns) != int(config["input_dim"]) or len(self.normalizer.mean) != len(self.feature_columns):
            raise ValueError("process checkpoint feature metadata mismatch")
        self.model = build_process_baseline(payload["model"], input_dim=int(config["input_dim"]), hidden_dim=int(config["hidden_dim"]))
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.profile = str(payload["model_profile"])
        self.threshold = float(payload.get("threshold_from_val", 0.5))
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @torch.no_grad()
    def predict(self, window: np.ndarray, *, feature_columns: Sequence[str]) -> dict[str, Any]:
        columns = list(feature_columns)
        if len(columns) != len(set(columns)) or set(columns) != set(self.feature_columns):
            raise ValueError("process checkpoint input needs the recorded feature columns")
        values = np.asarray(window, dtype=np.float32)
        if values.shape != (self.history_steps, len(columns)):
            raise ValueError(f"process checkpoint expects [{self.history_steps},{len(columns)}] window")
        order = [columns.index(name) for name in self.feature_columns]
        values = values[:, order]
        observed_fraction = float(np.isfinite(values).mean())
        if observed_fraction < 0.8:
            return {"status": "skipped_missing_features", "learned_event_probability": None, "observed_fraction": observed_fraction, "model_profile": self.profile}
        normalized = self.normalizer.transform(values)
        probability = float(torch.sigmoid(self.model(torch.from_numpy(normalized).unsqueeze(0)))[0])
        return {
            "status": "completed", "learned_event_probability": probability,
            "threshold_from_val": self.threshold, "predicted_event": probability >= self.threshold,
            "model_profile": self.profile, "checkpoint": str(self.path), "checkpoint_sha256": self.checkpoint_sha256,
            "normalization_fit_split": "train", "observed_fraction": observed_fraction,
            "probability_semantics": "sigmoid probability of source process-event label in future horizon",
            "calibration_status": "raw_sigmoid_evaluated_with_brier_ece",
            "future_horizon_minutes": self.future_minutes,
            "alignment_quality": "unknown", "scope": "independent_process_event; no_video_fusion; no_superheat_measurement",
        }


__all__ = ["ProcessCheckpointPredictor"]
