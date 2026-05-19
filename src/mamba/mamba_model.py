"""
mamba/mamba_model.py
--------------------
Mamba model definitions for AQI forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TimeSeriesMambaRegressor(nn.Module):
    """Mamba-based time-series regressor with location embedding."""

    def __init__(
        self,
        num_features: int,
        num_locations: int,
        d_model: int = 64,
        n_layers: int = 2,
        loc_embed_dim: int = 8,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.location_emb = nn.Embedding(num_locations, loc_embed_dim)
        self.input_proj = nn.Linear(num_features + loc_embed_dim, d_model)
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        self.num_features = num_features
        self.horizon = horizon

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x_seq   : (B, T, F)
        loc_ids : (B,)
        Returns : (B, H)
        """
        loc_vec = self.location_emb(loc_ids)
        loc_seq = loc_vec.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x = torch.cat([x_seq, loc_seq], dim=-1)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, -1, :])


class TimeSeriesMambaRegressorNoLoc(nn.Module):
    """Variant without location embedding."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        self.horizon = horizon

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x_seq)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])
