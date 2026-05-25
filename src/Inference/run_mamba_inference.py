"""Run trained Mamba AQI model on Gold inference input."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_MODEL_ROOT = _SRC_ROOT / "Model"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT), str(_MODEL_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.Inference.load_mamba import load_mamba_model
from src.Inference.predict_aqi_next import predict_aqi_next
from src.common.config import MINIO_ARTIFACTS_BUCKET, MINIO_GOLD_BUCKET
from src.common.minio_io import get_client, load_json_object, load_npy, upload_csv, upload_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next AQI horizon using trained Mamba")
    parser.add_argument("--inference-prefix", default=None, help="Gold prefix, e.g. inference_input/run_id=<id>")
    parser.add_argument("--inference-run-id", default=None, help="Inference input run id")
    parser.add_argument("--checkpoint", required=True, help="Path to best_mamba_aqi.pt")
    parser.add_argument("--metadata", required=True, help="Path to training_metadata.json")
    parser.add_argument("--output-path", default=None, help="Local CSV output path")
    parser.add_argument("--artifact-run-id", default=None, help="MinIO artifact run id")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload predictions to MinIO artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inference_prefix:
        if not args.inference_run_id:
            raise ValueError("Provide --inference-prefix or --inference-run-id.")
        args.inference_prefix = f"inference_input/run_id={args.inference_run_id}"
    inference_prefix = args.inference_prefix.rstrip("/")

    client = get_client()
    x_infer = load_npy(client, MINIO_GOLD_BUCKET, f"{inference_prefix}/X_inference.npy")
    if x_infer is None:
        raise FileNotFoundError(f"Missing s3://{MINIO_GOLD_BUCKET}/{inference_prefix}/X_inference.npy")

    infer_meta = load_json_object(client, MINIO_GOLD_BUCKET, f"{inference_prefix}/inference_metadata.json")
    if infer_meta is None:
        raise FileNotFoundError(f"Missing s3://{MINIO_GOLD_BUCKET}/{inference_prefix}/inference_metadata.json")

    model, train_meta = load_mamba_model(
        args.checkpoint,
        metadata_path=args.metadata,
        device=args.device,
    )
    target_norm = train_meta.get("target_normalization") or {}
    y_mean = target_norm.get("mean")
    y_std = target_norm.get("std")

    preds = predict_aqi_next(
        model,
        x_infer,
        device=args.device,
        y_mean=y_mean,
        y_std=y_std,
        use_amp=args.amp,
    )

    locations = infer_meta.get("locations") or [str(i) for i in range(preds.shape[0])]
    ranges = infer_meta.get("ranges") or {}

    rows: list[dict] = []
    for i, province in enumerate(locations):
        input_end = ranges.get(str(province), {}).get("end")
        base_time = pd.Timestamp(input_end) if input_end else pd.Timestamp.utcnow()
        for step in range(preds.shape[1]):
            rows.append(
                {
                    "province": province,
                    "horizon_step": step + 1,
                    "forecast_time": (base_time + pd.Timedelta(hours=step + 1)).isoformat(),
                    "y_pred": float(preds[i, step]),
                }
            )

    out_df = pd.DataFrame(rows)
    if args.output_path:
        out_path = Path(args.output_path)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
    else:
        run_id = args.artifact_run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = _REPO_ROOT / "runs" / "inference" / run_id / "future_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    artifact_run_id = args.artifact_run_id or out_path.parent.name
    artifact_prefix = f"mamba_inference/run_id={artifact_run_id}"
    if not args.no_upload:
        upload_csv(client, MINIO_ARTIFACTS_BUCKET, f"{artifact_prefix}/future_predictions.csv", out_df)
        upload_json(
            client,
            MINIO_ARTIFACTS_BUCKET,
            f"{artifact_prefix}/prediction_metadata.json",
            {
                "artifact_run_id": artifact_run_id,
                "inference_prefix": inference_prefix,
                "checkpoint": str(args.checkpoint),
                "metadata": str(args.metadata),
                "rows": len(out_df),
                "locations": locations,
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        print(f"Uploaded: s3://{MINIO_ARTIFACTS_BUCKET}/{artifact_prefix}/future_predictions.csv")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
