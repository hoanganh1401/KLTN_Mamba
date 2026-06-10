"""
Mamba model for AQI time-series forecasting.

The province/location name is used only by the data pipeline to group a
continuous time series per province. The model itself receives only numeric
time-series features with shape (batch, seq_len, num_features).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

try:
    from mamba_ssm import Mamba
except ModuleNotFoundError:
    from .mamba_ssm import Mamba


def _fused_fast_path_available() -> bool:
    try:
        import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    except Exception:
        return False
    return (
        selective_scan_interface.causal_conv1d_fwd_function is not None
        and selective_scan_interface.causal_conv1d_bwd_function is not None
        and selective_scan_interface.selective_scan_cuda is not None
    )


class TimeSeriesMambaRegressor(nn.Module):
    """Pure time-series Mamba regressor without location embedding."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool | None = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.horizon = int(horizon)
        self.use_location_embedding = False
        self.dropout = nn.Dropout(float(dropout))

        if use_fast_path is None:
            use_fast_path_requested = os.environ.get("MAMBA_USE_FASTPATH", "0").lower() in {"1", "true", "yes"}
            use_fast_path = use_fast_path_requested and _fused_fast_path_available()
        else:
            use_fast_path = bool(use_fast_path) and _fused_fast_path_available()

        self.input_proj = nn.Linear(self.num_features, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=use_fast_path,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, self.horizon),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Return forecast with shape (batch, horizon)."""
        x = self.input_proj(x_seq)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])


class TimeSeriesMambaRegressorNoLoc(TimeSeriesMambaRegressor):
    """Backward-compatible alias for the no-location Mamba regressor."""
