"""Small deterministic novelty scorers used as JEV features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EmbeddingNoveltyScorer:
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "EmbeddingNoveltyScorer":
        values = np.asarray(embeddings, dtype=float)
        if values.ndim != 2 or not len(values):
            raise ValueError("embeddings must have shape [N,D] and contain rows")
        self.mean = np.nanmean(values, axis=0)
        self.scale = np.nanstd(values, axis=0).clip(min=1e-6)
        return self

    def score(self, embedding: np.ndarray) -> float:
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit must be called before score")
        value = np.asarray(embedding, dtype=float).reshape(-1)
        if value.shape != self.mean.shape:
            raise ValueError("embedding dimension differs from fitted reference")
        distance = float(np.sqrt(np.mean(((value - self.mean) / self.scale) ** 2)))
        return float(1.0 - np.exp(-distance / 3.0))


def energy_novelty(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or temperature <= 0:
        raise ValueError("logits must have shape [N,C] and temperature must be positive")
    shifted = values / temperature
    energy = np.log(np.exp(shifted - shifted.max(axis=1, keepdims=True)).sum(axis=1)) + shifted.max(axis=1)
    return 1.0 / (1.0 + np.exp(np.clip(energy, -30.0, 30.0)))


__all__ = ["EmbeddingNoveltyScorer", "energy_novelty"]
