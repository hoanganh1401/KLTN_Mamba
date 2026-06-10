"""Load trained Mamba AQI checkpoints for inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_MODEL_ROOT = _SRC_ROOT / "Model"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT), str(_MODEL_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.Model.Mamba.mamba_model import TimeSeriesMambaRegressor


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve an inference device."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_metadata(path: str | Path | None) -> dict[str, Any]:
    """Load optional model metadata JSON."""
    if path is None:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def build_mamba_model(
    *,
    num_features: int,
    d_model: int = 64,
    n_layers: int = 2,
    horizon: int = 1,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Create the pure time-series Mamba regressor used by training."""
    return TimeSeriesMambaRegressor(
        num_features=num_features,
        d_model=d_model,
        n_layers=n_layers,
        horizon=horizon,
        dropout=dropout,
    )


def load_mamba_model(
    checkpoint_path: str | Path,
    *,
    num_features: int | None = None,
    d_model: int | None = None,
    n_layers: int | None = None,
    horizon: int | None = None,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained Mamba checkpoint and return `(model, metadata)`."""
    metadata = load_metadata(metadata_path)
    model_cfg = metadata.get("model", {}) if isinstance(metadata.get("model", {}), dict) else {}

    nf = num_features or metadata.get("num_features") or metadata.get("feature_count_after_encode") or model_cfg.get("num_features")
    dm = d_model or metadata.get("d_model") or model_cfg.get("d_model") or 64
    layers = n_layers or metadata.get("n_layers") or model_cfg.get("n_layers") or 2
    hz = horizon or metadata.get("horizon") or model_cfg.get("horizon") or 1
    dropout = float(model_cfg.get("dropout", metadata.get("dropout", 0.0)) or 0.0)

    if nf is None:
        raise ValueError("num_features is required when metadata does not provide it.")

    target_device = resolve_device(device)
    model = build_mamba_model(
        num_features=int(nf),
        d_model=int(dm),
        n_layers=int(layers),
        horizon=int(hz),
        dropout=dropout,
    )

    state = torch.load(Path(checkpoint_path), map_location=target_device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state)
    model.to(target_device)
    model.eval()
    return model, metadata
