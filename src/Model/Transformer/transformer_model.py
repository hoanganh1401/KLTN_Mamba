"""Transformer regressor for AQI time-series forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TimeSeriesTransformerRegressor(nn.Module):
    """Encoder-only Transformer regressor with the same interface as Mamba."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.1,
        n_heads: int = 4,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.horizon = int(horizon)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for Transformer.")

        dim_feedforward = dim_feedforward or d_model * 4
        self.input_proj = nn.Linear(self.num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_seq)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.head(self.norm(x[:, -1, :]))
