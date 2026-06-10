"""LSTM regressor for AQI time-series forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Attention pooling over LSTM outputs using the last timestep as query."""

    def __init__(self, hidden_size: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_size)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        q = self.query(x[:, -1:, :])
        k = self.key(x)
        v = self.value(x)
        weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(self.dropout(weights), v).squeeze(1)


class TimeSeriesLSTMRegressor(nn.Module):
    """Pure time-series LSTM regressor with shape-compatible Mamba interface."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.horizon = int(horizon)

        self.input_proj = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.horizon),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_seq)
        x, _ = self.lstm(x)
        x = self.norm(self.attention(x))
        return self.head(x)
