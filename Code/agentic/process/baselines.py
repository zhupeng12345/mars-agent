"""Small trainable baselines for the independent 14-variable process branch."""

from __future__ import annotations

import torch
from torch import nn


class ProcessCNN(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("process input must have shape [B,T,F]")
        return self.net(value.transpose(1, 2)).squeeze(-1)


class ProcessLSTM(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("process input must have shape [B,T,F]")
        _, (hidden, _) = self.lstm(value)
        return self.head(hidden[-1]).squeeze(-1)


class ProcessTCN(ProcessCNN):
    """Dilated convolution baseline with the same tensor contract."""

    def __init__(self, input_dim: int = 14, hidden_dim: int = 32):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, 1),
        )


def build_process_baseline(name: str, *, input_dim: int = 14, hidden_dim: int = 32) -> nn.Module:
    normalized = str(name).lower()
    if normalized in {"cnn", "1dcnn"}:
        return ProcessCNN(input_dim, hidden_dim)
    if normalized in {"lstm", "rnn"}:
        return ProcessLSTM(input_dim, hidden_dim)
    if normalized in {"tcn", "temporal_cnn"}:
        return ProcessTCN(input_dim, hidden_dim)
    raise ValueError(f"unknown process baseline: {name}")


__all__ = ["ProcessCNN", "ProcessLSTM", "ProcessTCN", "build_process_baseline"]
