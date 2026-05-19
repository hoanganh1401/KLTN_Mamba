"""
load_mamba.py
-------------
Model loader for Mamba.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_ROOT)]:
	if path not in sys.path:
		sys.path.insert(0, path)

from mamba.mamba_model import TimeSeriesMambaRegressor


def load_mamba_model(
	checkpoint_path: str | Path,
	num_features: int,
	num_locations: int,
	d_model: int,
	n_layers: int,
	horizon: int,
	device: str = "auto",
) -> TimeSeriesMambaRegressor:
	ckpt = Path(checkpoint_path).expanduser().resolve()
	if not ckpt.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model = TimeSeriesMambaRegressor(
		num_features=num_features,
		num_locations=num_locations,
		d_model=d_model,
		n_layers=n_layers,
		horizon=horizon,
	).to(device)

	state = torch.load(str(ckpt), map_location=device)
	model.load_state_dict(state)
	model.eval()
	return model
