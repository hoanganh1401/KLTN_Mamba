"""
prepare_training_dataset.py
===========================
Build training/validation/test datasets for Mamba from Gold features.

Flow:
Gold features -> train/val/test split -> fit scalers on train -> sliding windows
Outputs:
X_train.npy, y_train.npy, loc_ids_train.npy
X_val.npy, y_val.npy, loc_ids_val.npy
X_test.npy, y_test.npy, loc_ids_test.npy
scaler.pkl, dataset_metadata.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from minio import Minio
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from common.config import MINIO_GOLD_BUCKET, load_processing_config
from common.minio_io import (
    get_client,
    load_gold_features,
    upload_bytes,
    upload_json,
)


def _date_range(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _resolve_location_key(loc: dict) -> str:
    return loc.get("location_key") or loc.get("location") or f"{loc['latitude']}_{loc['longitude']}"


def _load_gold_range(client: Minio, location_key: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    for d in _date_range(start, end):
        df = load_gold_features(client, location_key, d.year, d.month, d.day)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged


def _build_samples(
    df: pd.DataFrame,
    time_col: str,
    location_col: str,
    target_col: str,
    feature_cols: list[str],
    seq_len: int,
    pred_len: int,
    sample_stride: int,
):
    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' column in Gold features.")
    if location_col not in df.columns:
        raise ValueError(f"Missing '{location_col}' column in Gold features.")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = work.dropna(subset=[time_col, location_col, target_col])

    feature_cols = [c for c in feature_cols if c in work.columns]
    if not feature_cols:
        raise ValueError("No feature columns found after filtering.")

    # Ensure numeric
    for col in feature_cols + [target_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=feature_cols + [target_col]).copy()
    work["_loc_id"] = work[location_col].astype("category").cat.codes.astype(np.int64)
    loc_map = (
        work.assign(_loc_key_str=work[location_col].astype(str))
        .drop_duplicates(subset=["_loc_key_str"])
        .set_index("_loc_key_str")["_loc_id"]
        .to_dict()
    )
    num_locations = int(work["_loc_id"].max()) + 1

    work = work.sort_values(["_loc_id", time_col]).reset_index(drop=True)

    x_list, loc_list, y_list, y_ts_list = [], [], [], []
    for loc_id, group in work.groupby("_loc_id", sort=False):
        x_vals = group[feature_cols].to_numpy(dtype=np.float32)
        y_vals = group[target_col].to_numpy(dtype=np.float32)
        ts_vals = group[time_col].to_numpy(dtype="datetime64[ns]")
        n = len(group)
        max_start = n - seq_len - pred_len + 1
        if max_start <= 0:
            continue

        starts = np.arange(0, max_start, sample_stride, dtype=np.int64)
        ends = starts + seq_len
        target_ends = ends + pred_len

        row_s, feat_s = x_vals.strides
        x_all = np.lib.stride_tricks.as_strided(
            x_vals,
            shape=(len(starts), seq_len, x_vals.shape[1]),
            strides=(sample_stride * row_s, row_s, feat_s),
        )
        y_idx = ends[:, None] + np.arange(pred_len, dtype=np.int64)[None, :]

        x_list.append(x_all.copy())
        loc_list.append(np.full(len(starts), loc_id, dtype=np.int64))
        y_list.append(y_vals[y_idx].astype(np.float32, copy=False))
        y_ts_list.append(ts_vals[target_ends - 1])

    if not x_list:
        raise ValueError("No samples were generated. Increase data size or adjust seq_len/pred_len.")

    return (
        np.concatenate(x_list, axis=0).astype(np.float32),
        np.concatenate(loc_list, axis=0).astype(np.int64),
        np.concatenate(y_list, axis=0).astype(np.float32),
        np.concatenate(y_ts_list, axis=0).astype("datetime64[ns]"),
        loc_map,
        num_locations,
        feature_cols,
    )


def _split_by_timeline(
    x: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
):
    if len(y) < 3:
        raise ValueError("Need at least 3 samples to split.")
    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split sizes.")
    idx_train = order[:train_end]
    idx_val = order[train_end:val_end]
    idx_test = order[val_end:]
    return (
        (x[idx_train], loc_ids[idx_train], y[idx_train]),
        (x[idx_val], loc_ids[idx_val], y[idx_val]),
        (x[idx_test], loc_ids[idx_test], y[idx_test]),
    )


def _scaler_factory(name: str):
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _save_numpy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, arr)


def run_prepare_training_dataset(
    locations_path: str,
    start_date: str,
    end_date: str,
    seq_len: int | None,
    pred_len: int | None,
    train_ratio: float | None,
    val_ratio: float | None,
    target_col: str | None,
    location_col: str | None,
    time_col: str | None,
    sample_stride: int,
    scaler_name: str | None,
    include_target_history: bool | None,
    output_dir: Path,
    minio_prefix: str | None,
):
    client = get_client()
    cfg = load_processing_config(client)

    with open(locations_path, encoding="utf-8") as f:
        locations = [json.loads(line) for line in f if line.strip()]

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    time_col = time_col or cfg["time_col"]
    location_col = location_col or cfg["location_col"]
    target_col = target_col or cfg["target_col"]

    seq_len = int(seq_len or cfg["dataset"]["seq_len"])
    pred_len = int(pred_len or cfg["dataset"]["pred_len"])
    train_ratio = float(train_ratio or cfg["dataset"]["train_ratio"])
    val_ratio = float(val_ratio or cfg["dataset"]["val_ratio"])
    include_target_history = (
        cfg["dataset"].get("include_target_history", True)
        if include_target_history is None
        else include_target_history
    )
    scaler_name = scaler_name or cfg["normalization_method"]

    feature_cols = [c for c in (cfg["metric_cols"] + cfg["time_features"])]
    if not include_target_history:
        feature_cols = [c for c in feature_cols if c != target_col]

    all_frames = []
    for loc in locations:
        loc_key = _resolve_location_key(loc)
        df_loc = _load_gold_range(client, loc_key, start, end)
        if df_loc.empty:
            continue
        if location_col not in df_loc.columns:
            df_loc[location_col] = loc_key
        all_frames.append(df_loc)

    if not all_frames:
        raise ValueError("No Gold features found for the selected date range.")

    merged = pd.concat(all_frames, ignore_index=True)
    x_all, loc_ids, y_all, y_ts, loc_map, num_locations, feature_cols = _build_samples(
        merged,
        time_col=time_col,
        location_col=location_col,
        target_col=target_col,
        feature_cols=feature_cols,
        seq_len=seq_len,
        pred_len=pred_len,
        sample_stride=sample_stride,
    )

    (x_train, loc_train, y_train), (x_val, loc_val, y_val), (x_test, loc_test, y_test) = _split_by_timeline(
        x_all,
        loc_ids,
        y_all,
        y_ts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    x_scaler = _scaler_factory(scaler_name)
    y_scaler = _scaler_factory(scaler_name)

    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    x_scaler.fit(x_train_2d)
    y_scaler.fit(y_train.reshape(-1, 1))

    def _scale_x(x):
        return x_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    def _scale_y(y):
        return y_scaler.transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)

    x_train = _scale_x(x_train)
    x_val = _scale_x(x_val)
    x_test = _scale_x(x_test)
    y_train = _scale_y(y_train)
    y_val = _scale_y(y_val)
    y_test = _scale_y(y_test)

    _save_numpy(output_dir / "X_train.npy", x_train)
    _save_numpy(output_dir / "y_train.npy", y_train)
    _save_numpy(output_dir / "loc_ids_train.npy", loc_train)
    _save_numpy(output_dir / "X_val.npy", x_val)
    _save_numpy(output_dir / "y_val.npy", y_val)
    _save_numpy(output_dir / "loc_ids_val.npy", loc_val)
    _save_numpy(output_dir / "X_test.npy", x_test)
    _save_numpy(output_dir / "y_test.npy", y_test)
    _save_numpy(output_dir / "loc_ids_test.npy", loc_test)

    scaler_payload = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "location_col": location_col,
        "time_col": time_col,
        "location_to_id": loc_map,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "num_locations": num_locations,
        "include_target_history": include_target_history,
    }
    scaler_path = output_dir / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_payload, f)

    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "source_range": {"start": start_date, "end": end_date},
        "seq_len": seq_len,
        "pred_len": pred_len,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "target_col": target_col,
        "location_col": location_col,
        "time_col": time_col,
        "include_target_history": include_target_history,
        "feature_cols": feature_cols,
        "num_locations": num_locations,
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
    }
    meta_path = output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if minio_prefix:
        prefix = minio_prefix.rstrip("/")
        client = get_client()
        for name in [
            "X_train.npy",
            "y_train.npy",
            "loc_ids_train.npy",
            "X_val.npy",
            "y_val.npy",
            "loc_ids_val.npy",
            "X_test.npy",
            "y_test.npy",
            "loc_ids_test.npy",
        ]:
            payload = (output_dir / name).read_bytes()
            upload_bytes(client, MINIO_GOLD_BUCKET, f"{prefix}/{name}", payload)

        upload_bytes(client, MINIO_GOLD_BUCKET, f"{prefix}/scaler.pkl", scaler_path.read_bytes())
        upload_json(client, MINIO_GOLD_BUCKET, f"{prefix}/dataset_metadata.json", metadata)

    print(f"OK: Training dataset prepared at: {output_dir}")
    if minio_prefix:
        print(f"OK: Uploaded to MinIO: s3://{MINIO_GOLD_BUCKET}/{minio_prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training dataset for Mamba")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--location-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--scaler", type=str, default=None, choices=["standard", "robust", "minmax"])
    parser.add_argument("--exclude-target-history", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--minio-prefix", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()
    else:
        dataset_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = (project_root / "artifacts" / "datasets" / dataset_id).resolve()

    run_prepare_training_dataset(
        locations_path=args.locations,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_col=args.target_col,
        location_col=args.location_col,
        time_col=args.time_col,
        sample_stride=args.sample_stride,
        scaler_name=args.scaler,
        include_target_history=None if not args.exclude_target_history else False,
        output_dir=output_dir,
        minio_prefix=args.minio_prefix,
    )


if __name__ == "__main__":
    main()
