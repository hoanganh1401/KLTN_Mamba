"""
predict_aqi_next.py
-------------------
Run inference for next AQI values using a trained Mamba model.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_ROOT)]:
	if path not in sys.path:
		sys.path.insert(0, path)

from common.config import MINIO_GOLD_BUCKET
from common.minio_io import download_bytes, get_client
from Inference.load_mamba import load_mamba_model


def _load_scaler_payload(dataset_dir: Path | None, minio_prefix: str | None) -> dict:
	if dataset_dir:
		payload = (dataset_dir / "scaler.pkl").read_bytes()
	elif minio_prefix:
		client = get_client()
		payload = download_bytes(client, MINIO_GOLD_BUCKET, f"{minio_prefix.rstrip('/')}/scaler.pkl")
	else:
		raise ValueError("Provide --dataset-dir or --minio-prefix to load scaler.pkl")
	return pickle.loads(payload)


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict AQI next steps with Mamba")
	parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt)")
	parser.add_argument("--inference-dir", required=True, help="Folder with X_inference.npy and loc_id.npy")
	parser.add_argument("--dataset-dir", default=None, help="Folder with scaler.pkl")
	parser.add_argument("--minio-prefix", default=None, help="MinIO prefix for scaler.pkl")
	parser.add_argument("--d-model", type=int, default=64)
	parser.add_argument("--n-layers", type=int, default=2)
	parser.add_argument("--horizon", type=int, default=12)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
	parser.add_argument("--output", default=None, help="Output CSV path")
	args = parser.parse_args()

	inference_dir = Path(args.inference_dir).resolve()
	x_path = inference_dir / "X_inference.npy"
	loc_path = inference_dir / "loc_id.npy"

	if not x_path.exists() or not loc_path.exists():
		raise FileNotFoundError("X_inference.npy or loc_id.npy not found in inference dir.")

	x_infer = np.load(x_path)
	loc_id = np.load(loc_path)

	payload = _load_scaler_payload(Path(args.dataset_dir).resolve() if args.dataset_dir else None, args.minio_prefix)
	y_scaler = payload["y_scaler"]
	num_locations = payload["num_locations"]

	model = load_mamba_model(
		checkpoint_path=args.model_path,
		num_features=x_infer.shape[-1],
		num_locations=num_locations,
		d_model=args.d_model,
		n_layers=args.n_layers,
		horizon=args.horizon,
		device=args.device,
	)

	device = next(model.parameters()).device
	with torch.inference_mode():
		x_tensor = torch.from_numpy(x_infer).to(device)
		loc_tensor = torch.from_numpy(loc_id.astype(np.int64)).to(device)
		preds_norm = model(x_tensor, loc_tensor).detach().cpu().numpy()

	preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).reshape(preds_norm.shape)
	steps = np.arange(preds.shape[1])

	out = pd.DataFrame({"step": steps, "predicted_aqi": preds.flatten()})
	output_path = Path(args.output).resolve() if args.output else (inference_dir / "predictions.csv")
	out.to_csv(output_path, index=False)

	summary = {"output": str(output_path), "n_steps": int(len(steps))}
	print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
	main()
