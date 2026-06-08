"""Run trained Mamba AQI model on Gold inference input."""

from __future__ import annotations

import argparse
import sys
import tempfile
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
from src.common.minio_io import get_client, load_bytes, load_json_object, load_npy, upload_csv, upload_json
from src.common.time_utils import now_local, parse_time_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next AQI horizon using trained Mamba")
    parser.add_argument("--inference-prefix", default=None, help="Gold prefix, e.g. inference_input/run_id=<id>")
    parser.add_argument("--inference-run-id", default=None, help="Inference input run id")
    parser.add_argument("--province", default=None, help="Province key for per-province model, e.g. an_giang")
    parser.add_argument("--model-run-id", default=None, help="Model run id on MinIO, e.g. mamba_an_giang_20260525")
    parser.add_argument("--checkpoint", default=None, help="Optional local path to best_mamba_aqi.pt")
    parser.add_argument("--metadata", default=None, help="Optional local path to training_metadata.json")
    parser.add_argument("--output-path", default=None, help="Local CSV output path")
    parser.add_argument("--artifact-run-id", default=None, help="MinIO artifact run id")
    parser.add_argument("--keep-local", action="store_true", help="Keep local prediction CSV after uploading to MinIO")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload predictions to MinIO artifacts")
    return parser.parse_args()


def resolve_model_files_from_minio(client, args: argparse.Namespace) -> tuple[str, str, tempfile.TemporaryDirectory | None]:
    if args.checkpoint and args.metadata:
        return args.checkpoint, args.metadata, None
    if not args.province or not args.model_run_id:
        raise ValueError("Provide either --checkpoint/--metadata or --province/--model-run-id.")

    prefix = f"mamba/province={args.province}/run_id={args.model_run_id}"
    checkpoint_bytes = load_bytes(client, MINIO_ARTIFACTS_BUCKET, f"{prefix}/best_mamba_aqi.pt")
    if checkpoint_bytes is None:
        checkpoint_bytes = load_bytes(client, MINIO_ARTIFACTS_BUCKET, f"{prefix}/best_model.pt")
    metadata_bytes = load_bytes(client, MINIO_ARTIFACTS_BUCKET, f"{prefix}/training_metadata.json")
    if checkpoint_bytes is None:
        raise FileNotFoundError(f"Missing model checkpoint on s3://{MINIO_ARTIFACTS_BUCKET}/{prefix}/")
    if metadata_bytes is None:
        raise FileNotFoundError(f"Missing training_metadata.json on s3://{MINIO_ARTIFACTS_BUCKET}/{prefix}/")

    tmp = tempfile.TemporaryDirectory(prefix=f"{args.model_run_id}_infer_")
    tmp_path = Path(tmp.name)
    checkpoint_path = tmp_path / "best_mamba_aqi.pt"
    metadata_path = tmp_path / "training_metadata.json"
    checkpoint_path.write_bytes(checkpoint_bytes)
    metadata_path.write_bytes(metadata_bytes)
    return str(checkpoint_path), str(metadata_path), tmp


def resolve_artifact_prefix(out_df: pd.DataFrame, artifact_run_id: str, args: argparse.Namespace) -> tuple[str, dict]:
    if out_df.empty or "forecast_time" not in out_df.columns:
        forecast_start = ""
        forecast_end = ""
        forecast_date = now_local().date().isoformat()
    else:
        forecast_times = parse_time_local(out_df["forecast_time"]).dropna()
        if forecast_times.empty:
            forecast_start = ""
            forecast_end = ""
            forecast_date = now_local().date().isoformat()
        else:
            forecast_start_ts = forecast_times.min()
            forecast_end_ts = forecast_times.max()
            forecast_start = forecast_start_ts.isoformat()
            forecast_end = forecast_end_ts.isoformat()
            forecast_date = forecast_start_ts.date().isoformat()

    province = args.province
    if not province:
        provinces = sorted(out_df["province"].dropna().astype(str).unique().tolist()) if "province" in out_df.columns else []
        province = provinces[0] if len(provinces) == 1 else "all_locations"

    prefix = f"mamba_inference/province={province}/forecast_date={forecast_date}/run_id={artifact_run_id}"
    return prefix, {
        "province": province,
        "forecast_date": forecast_date,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
    }


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

    checkpoint_path, metadata_path, tmp_model_dir = resolve_model_files_from_minio(client, args)

    model, train_meta = load_mamba_model(
        checkpoint_path,
        metadata_path=metadata_path,
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
        base_time = parse_time_local(pd.Series([input_end])).iloc[0] if input_end else pd.Timestamp(now_local())
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
    cleanup_output_dir = False
    if args.output_path:
        out_path = Path(args.output_path)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
    else:
        run_id = args.artifact_run_id or now_local().strftime("%Y%m%d_%H%M%S")
        tmp_out_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}_prediction_")).resolve()
        out_path = tmp_out_dir / "future_predictions.csv"
        cleanup_output_dir = not args.keep_local
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    artifact_run_id = args.artifact_run_id or out_path.parent.name
    artifact_prefix, forecast_meta = resolve_artifact_prefix(out_df, artifact_run_id, args)
    if not args.no_upload:
        upload_csv(client, MINIO_ARTIFACTS_BUCKET, f"{artifact_prefix}/future_predictions.csv", out_df)
        upload_json(
            client,
            MINIO_ARTIFACTS_BUCKET,
            f"{artifact_prefix}/prediction_metadata.json",
            {
                "artifact_run_id": artifact_run_id,
                "artifact_prefix": artifact_prefix,
                "inference_prefix": inference_prefix,
                **forecast_meta,
                "model_run_id": args.model_run_id,
                "checkpoint": str(args.checkpoint or f"s3://{MINIO_ARTIFACTS_BUCKET}/mamba/province={args.province}/run_id={args.model_run_id}/best_mamba_aqi.pt"),
                "metadata": str(args.metadata or f"s3://{MINIO_ARTIFACTS_BUCKET}/mamba/province={args.province}/run_id={args.model_run_id}/training_metadata.json"),
                "rows": len(out_df),
                "locations": locations,
                "created_at": now_local().isoformat(),
            },
        )
        print(f"Uploaded: s3://{MINIO_ARTIFACTS_BUCKET}/{artifact_prefix}/future_predictions.csv")

    print(f"Saved: {out_path}")
    if cleanup_output_dir:
        tmp_dir = out_path.parent
        tmp_file = str(out_path)
        try:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Temporary local prediction removed: {tmp_file}")
        except Exception as exc:
            print(f"[WARN] Could not remove temporary local prediction dir {tmp_dir}: {exc}")
    if tmp_model_dir is not None:
        tmp_model_dir.cleanup()


if __name__ == "__main__":
    main()
