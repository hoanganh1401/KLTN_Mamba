"""
prepare_inference_input.py
==========================
Prepare the latest input window for Mamba inference.

Reads Gold features, loads scaler.pkl from training, and outputs:
X_inference.npy, loc_id.npy
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from minio import Minio

from common.config import MINIO_GOLD_BUCKET
from common.minio_io import download_bytes, get_client, load_gold_features, upload_bytes


def _resolve_location_key(loc: dict) -> str:
	return loc.get("location_key") or loc.get("location") or f"{loc['latitude']}_{loc['longitude']}"


def _load_scaler_payload(dataset_dir: Path | None, minio_prefix: str | None) -> dict:
	if dataset_dir:
		payload = (dataset_dir / "scaler.pkl").read_bytes()
	elif minio_prefix:
		client = get_client()
		payload = download_bytes(client, MINIO_GOLD_BUCKET, f"{minio_prefix.rstrip('/')}/scaler.pkl")
	else:
		raise ValueError("Provide --dataset-dir or --minio-prefix to load scaler.pkl")
	return pickle.loads(payload)


def _load_latest_window(
	client: Minio,
	location_key: str,
	end_date: datetime,
	seq_len: int,
	location_col: str,
	time_col: str,
	max_lookback_days: int,
):
	frames = []
	cur = end_date.date()
	for _ in range(max_lookback_days):
		df = load_gold_features(client, location_key, cur.year, cur.month, cur.day)
		if df is not None and not df.empty:
			if location_col not in df.columns:
				df[location_col] = location_key
			frames.append(df)
		cur -= timedelta(days=1)

		if frames:
			merged = pd.concat(frames, ignore_index=True)
			merged[time_col] = pd.to_datetime(merged[time_col], utc=True, errors="coerce")
			merged = merged.dropna(subset=[time_col])
			merged = merged.sort_values(time_col)
			if len(merged) >= seq_len:
				return merged.tail(seq_len).reset_index(drop=True)

	raise ValueError("Not enough data to build inference window. Increase lookback days or reduce seq_len.")


def run_prepare_inference_input(
	locations_path: str,
	location_key: str | None,
	seq_len: int,
	pred_len: int,
	location_col: str | None,
	time_col: str | None,
	dataset_dir: Path | None,
	minio_prefix: str | None,
	output_dir: Path,
	output_minio_prefix: str | None,
	max_lookback_days: int,
):
	if location_key is None:
		with open(locations_path, encoding="utf-8") as f:
			locations = [json.loads(line) for line in f if line.strip()]
		if not locations:
			raise ValueError("Locations file is empty.")
		location_key = _resolve_location_key(locations[0])

	payload = _load_scaler_payload(dataset_dir, minio_prefix)
	feature_cols = payload["feature_cols"]
	location_to_id = payload["location_to_id"]
	x_scaler = payload["x_scaler"]
	location_col = location_col or payload.get("location_col", "location")
	time_col = time_col or payload.get("time_col", "time")

	if location_key not in location_to_id:
		raise ValueError(f"Location '{location_key}' not found in training metadata.")

	client = get_client()
	end_dt = datetime.utcnow()
	window_df = _load_latest_window(
		client,
		location_key=location_key,
		end_date=end_dt,
		seq_len=seq_len,
		location_col=location_col,
		time_col=time_col,
		max_lookback_days=max_lookback_days,
	)

	for col in feature_cols:
		if col not in window_df.columns:
			raise ValueError(f"Missing feature column in Gold data: {col}")

	x_raw = window_df[feature_cols].to_numpy(dtype=np.float32)
	x_scaled = x_scaler.transform(x_raw).astype(np.float32)
	x_scaled = x_scaled.reshape(1, seq_len, -1)
	loc_id = np.array([location_to_id[location_key]], dtype=np.int64)

	output_dir.mkdir(parents=True, exist_ok=True)
	# Save arrays
	with open(output_dir / "X_inference.npy", "wb") as f:
		np.save(f, x_scaled)
	with open(output_dir / "loc_id.npy", "wb") as f:
		np.save(f, loc_id)

	meta = {
		"created_at": datetime.utcnow().isoformat(),
		"location_key": location_key,
		"seq_len": seq_len,
		"pred_len": pred_len,
		"feature_cols": feature_cols,
	}
	(output_dir / "inference_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

	if output_minio_prefix:
		prefix = output_minio_prefix.rstrip("/")
		upload_bytes(client, MINIO_GOLD_BUCKET, f"{prefix}/X_inference.npy", (output_dir / "X_inference.npy").read_bytes())
		upload_bytes(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_id.npy", (output_dir / "loc_id.npy").read_bytes())
		upload_bytes(client, MINIO_GOLD_BUCKET, f"{prefix}/inference_metadata.json", (output_dir / "inference_metadata.json").read_bytes(), content_type="application/json")

	print(f"OK: Inference input prepared at: {output_dir}")
	if output_minio_prefix:
		print(f"OK: Uploaded to MinIO: s3://{MINIO_GOLD_BUCKET}/{output_minio_prefix}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Prepare inference input for Mamba")
	parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
	parser.add_argument("--location", default=None, help="Location key (optional)")
	parser.add_argument("--seq-len", type=int, default=96)
	parser.add_argument("--pred-len", type=int, default=12)
	parser.add_argument("--location-col", type=str, default=None)
	parser.add_argument("--time-col", type=str, default=None)
	parser.add_argument("--dataset-dir", type=str, default=None)
	parser.add_argument("--minio-prefix", type=str, default=None)
	parser.add_argument("--output-dir", type=str, default=None)
	parser.add_argument("--output-minio-prefix", type=str, default=None)
	parser.add_argument("--max-lookback-days", type=int, default=30)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[2]
	dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None
	output_dir = Path(args.output_dir).resolve() if args.output_dir else (project_root / "artifacts" / "inference_input")

	run_prepare_inference_input(
		locations_path=args.locations,
		location_key=args.location,
		seq_len=args.seq_len,
		pred_len=args.pred_len,
		location_col=args.location_col,
		time_col=args.time_col,
		dataset_dir=dataset_dir,
		minio_prefix=args.minio_prefix,
		output_dir=output_dir,
		output_minio_prefix=args.output_minio_prefix,
		max_lookback_days=args.max_lookback_days,
	)


if __name__ == "__main__":
	main()
