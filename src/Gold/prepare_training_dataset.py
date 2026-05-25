"""
prepare_training_dataset.py — Gold/features → Gold/training_dataset
==================================================================
Build train/val/test arrays for Mamba from Gold feature data in MinIO.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common.config import MINIO_GOLD_BUCKET, load_processing_config, load_project_config
from common.minio_io import (
    get_client,
    load_gold_features,
    upload_json,
    upload_npy,
    upload_pickle,
)

DEFAULT_HARD_FAIL_COLS = [
    "_invalid_segment",
    "_flag_duplicate",
    "_flag_physical_bound",
    "_flag_aqi_range",
]
DEFAULT_IMPUTED_COL = "_imputed"
DEFAULT_LOW_COVERAGE_COL = "_flag_low_coverage"


def iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parse_date(value: str | None, default: date | None = None) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    if default is None:
        raise ValueError("Missing date value")
    return default


def load_locations(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        for col in df.columns:
            if col.lower() == "time":
                df = df.rename(columns={col: "time"})
                break
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def collect_gold_features(
    locations: list[dict],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    client = get_client()
    frames: list[pd.DataFrame] = []

    for loc in locations:
        loc_key = loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
        for day in iter_dates(start_date, end_date):
            df = load_gold_features(client, loc_key, day.year, day.month, day.day)
            if df is None or df.empty:
                continue

            df = df.copy()
            if "location_key" not in df.columns:
                df["location_key"] = df.get("location", loc_key)

            df = normalize_time_column(df)

            frames.append(df)

    if not frames:
        raise ValueError("No Gold feature data found for the selected range.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["time"]).copy()
    combined = combined.sort_values(["location_key", "time"]).reset_index(drop=True)

    return combined


def as_bool_array(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.zeros(0, dtype=bool)
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).to_numpy(dtype=float) != 0
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"}).to_numpy(dtype=bool)


def window_quality_status(
    window: pd.DataFrame,
    hard_fail_cols: list[str],
    imputed_col: str,
    max_imputed_ratio: float,
    low_coverage_col: str,
    max_low_coverage_ratio: float,
) -> tuple[bool, str]:
    for col in hard_fail_cols:
        if col in window.columns and as_bool_array(window[col]).any():
            return False, col

    if imputed_col in window.columns:
        imputed_ratio = float(as_bool_array(window[imputed_col]).mean())
        if imputed_ratio > max_imputed_ratio:
            return False, f"{imputed_col}_ratio"

    if low_coverage_col in window.columns:
        low_coverage_ratio = float(as_bool_array(window[low_coverage_col]).mean())
        if low_coverage_ratio > max_low_coverage_ratio:
            return False, f"{low_coverage_col}_ratio"

    return True, "ok"


def build_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    pred_len: int,
    sample_stride: int,
    hard_fail_cols: list[str],
    imputed_col: str,
    max_imputed_ratio: float,
    low_coverage_col: str,
    max_low_coverage_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    missing_cols = [c for c in ["time", target_col] + feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=feature_cols + [target_col, "time"]).copy()

    location_keys = sorted(df["location_key"].astype(str).unique().tolist())
    loc_map = {loc: idx for idx, loc in enumerate(location_keys)}

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    loc_ids: list[int] = []
    y_ts: list[np.datetime64] = []
    quality_stats: dict[str, int] = {"accepted": 0}

    for loc in location_keys:
        g = df.loc[df["location_key"].astype(str) == loc].copy()
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < seq_len + pred_len:
            continue

        values = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        times = pd.to_datetime(g["time"], utc=True, errors="coerce").dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

        max_start = len(g) - seq_len - pred_len
        for start in range(0, max_start + 1, sample_stride):
            full_window = g.iloc[start : start + seq_len + pred_len]
            ok, reason = window_quality_status(
                full_window,
                hard_fail_cols=hard_fail_cols,
                imputed_col=imputed_col,
                max_imputed_ratio=max_imputed_ratio,
                low_coverage_col=low_coverage_col,
                max_low_coverage_ratio=max_low_coverage_ratio,
            )
            if not ok:
                quality_stats[reason] = quality_stats.get(reason, 0) + 1
                continue

            x = values[start : start + seq_len]
            y = targets[start + seq_len : start + seq_len + pred_len]
            if np.isnan(x).any() or np.isnan(y).any():
                quality_stats["nan_feature_or_target"] = quality_stats.get("nan_feature_or_target", 0) + 1
                continue
            x_list.append(x)
            y_list.append(y)
            loc_ids.append(loc_map[loc])
            y_ts.append(times[start + seq_len])
            quality_stats["accepted"] += 1

    if not x_list:
        raise ValueError("No training samples generated. Check seq_len/pred_len or data range.")

    x_arr = np.stack(x_list, axis=0)
    y_arr = np.stack(y_list, axis=0)
    loc_arr = np.asarray(loc_ids, dtype=np.int64)
    ts_arr = np.asarray(y_ts, dtype="datetime64[ns]")
    return x_arr, y_arr, loc_arr, ts_arr, loc_map, quality_stats


def get_quality_config(project_cfg: dict) -> dict:
    quality_cfg = project_cfg.get("quality", {}) or {}
    return {
        "hard_fail_cols": quality_cfg.get("hard_fail_cols") or DEFAULT_HARD_FAIL_COLS,
        "imputed_col": quality_cfg.get("imputed_col", DEFAULT_IMPUTED_COL),
        "max_imputed_ratio": float(quality_cfg.get("max_imputed_ratio", 0.3)),
        "low_coverage_col": quality_cfg.get("low_coverage_col", DEFAULT_LOW_COVERAGE_COL),
        "max_low_coverage_ratio": float(quality_cfg.get("max_low_coverage_ratio", 0.5)),
    }


def split_by_time(
    x: np.ndarray,
    y: np.ndarray,
    loc_ids: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, dict[str, np.ndarray]]:
    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid train/val split ratios for dataset size.")

    def take(idx: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "x": x[idx],
            "y": y[idx],
            "loc_ids": loc_ids[idx],
            "y_ts": y_ts[idx],
        }

    return {
        "train": take(order[:train_end]),
        "val": take(order[train_end:val_end]),
        "test": take(order[val_end:]),
    }


def make_scaler(method: str):
    if method == "robust":
        return RobustScaler()
    if method == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def apply_scaler(x: np.ndarray, scaler, metric_idx: list[int]) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    scaled = scaler.transform(flat[:, metric_idx])
    flat[:, metric_idx] = scaled
    return flat.reshape(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mamba training dataset from Gold features")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--location-keys", default=None, help="Comma-separated province keys, e.g. an_giang")
    parser.add_argument("--config", default=None, help="Path to project YAML config")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--days", type=int, default=180, help="Lookback days if start-date not set")
    parser.add_argument("--seq-len", type=int, default=None, help="Input window length")
    parser.add_argument("--pred-len", type=int, default=None, help="Prediction horizon")
    parser.add_argument("--sample-stride", type=int, default=None, help="Sliding window stride")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--run-id", default=None, help="Optional run id for output prefix")
    args = parser.parse_args()

    project_cfg = load_project_config(args.config)
    data_cfg = project_cfg.get("data", {})
    dataset_cfg = project_cfg.get("dataset", {})
    scaling_cfg = project_cfg.get("scaling", {})
    features_cfg = project_cfg.get("features", {})

    end_default = (datetime.utcnow() - timedelta(days=1)).date()
    end_date = parse_date(args.end_date, end_default)
    if args.start_date:
        start_date = parse_date(args.start_date)
    else:
        start_date = end_date - timedelta(days=max(1, args.days) - 1)

    seq_len = args.seq_len or int(dataset_cfg.get("seq_len", 96))
    pred_len = args.pred_len or int(dataset_cfg.get("pred_len", 12))
    sample_stride = args.sample_stride or int(dataset_cfg.get("sample_stride", 1))

    train_ratio = args.train_ratio if args.train_ratio is not None else float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(dataset_cfg.get("val_ratio", 0.1))
    test_ratio = dataset_cfg.get("test_ratio", None)
    if test_ratio is not None and args.val_ratio is None:
        try:
            test_ratio = float(test_ratio)
            remain = 1.0 - train_ratio - test_ratio
            if remain > 0:
                val_ratio = remain
        except Exception:
            test_ratio = None

    target_col = args.target_col or data_cfg.get("target_col", "aqi")

    if args.location_keys:
        locations = [{"location_key": key.strip()} for key in args.location_keys.split(",") if key.strip()]
    else:
        locations = load_locations(args.locations)
    data = collect_gold_features(locations, start_date, end_date)

    cfg = load_processing_config(get_client())
    metric_cols = data_cfg.get("metric_cols") or cfg["metric_cols"]
    time_features = features_cfg.get("time_features") or cfg["time_features"]
    scaling_method = str(scaling_cfg.get("method") or cfg["normalization_method"]).lower()

    fit_on = str(scaling_cfg.get("fit_on") or "train").lower()
    if fit_on in {"train", "train_only"}:
        fit_on = "train"
    elif fit_on in {"all", "full", "all_data"}:
        fit_on = "all"
    else:
        fit_on = "train"

    if target_col not in metric_cols:
        metric_cols = metric_cols + [target_col]

    feature_cols = metric_cols + [c for c in time_features if c not in metric_cols]
    feature_cols = [c for c in feature_cols if not c.startswith("_")]
    quality_cfg = get_quality_config(project_cfg)

    x_arr, y_arr, loc_ids, y_ts, loc_map, quality_stats = build_samples(
        data,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
        sample_stride=sample_stride,
        hard_fail_cols=quality_cfg["hard_fail_cols"],
        imputed_col=quality_cfg["imputed_col"],
        max_imputed_ratio=quality_cfg["max_imputed_ratio"],
        low_coverage_col=quality_cfg["low_coverage_col"],
        max_low_coverage_ratio=quality_cfg["max_low_coverage_ratio"],
    )

    splits = split_by_time(
        x_arr,
        y_arr,
        loc_ids,
        y_ts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    metric_idx = [feature_cols.index(c) for c in metric_cols if c in feature_cols]
    scaler = make_scaler(scaling_method)
    fit_data = x_arr if fit_on == "all" else splits["train"]["x"]
    scaler.fit(fit_data.reshape(-1, x_arr.shape[-1])[:, metric_idx])

    for split in splits.values():
        split["x"] = apply_scaler(split["x"], scaler, metric_idx)

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = f"training_dataset/run_id={run_id}"

    client = get_client()
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_train.npy", splits["train"]["x"].astype(np.float32))
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_train.npy", splits["train"]["y"].astype(np.float32))
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_val.npy", splits["val"]["x"].astype(np.float32))
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_val.npy", splits["val"]["y"].astype(np.float32))
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_test.npy", splits["test"]["x"].astype(np.float32))
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_test.npy", splits["test"]["y"].astype(np.float32))

    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_train.npy", splits["train"]["loc_ids"])
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_val.npy", splits["val"]["loc_ids"])
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_test.npy", splits["test"]["loc_ids"])
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_train.npy", splits["train"]["y_ts"])
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_val.npy", splits["val"]["y_ts"])
    upload_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_test.npy", splits["test"]["y_ts"])

    upload_pickle(client, MINIO_GOLD_BUCKET, f"{prefix}/scaler.pkl", scaler)

    metadata = {
        "run_id": run_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "metric_cols": metric_cols,
        "time_features": time_features,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "sample_stride": sample_stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scaling_method": scaling_method,
        "scaling_fit_on": fit_on,
        "quality_gate": quality_cfg,
        "quality_stats": quality_stats,
        "location_to_id": loc_map,
        "locations": sorted(loc_map.keys()),
        "scope": "all_locations" if not args.location_keys else "selected_locations",
        "shapes": {
            "X_train": list(splits["train"]["x"].shape),
            "y_train": list(splits["train"]["y"].shape),
            "X_val": list(splits["val"]["x"].shape),
            "y_val": list(splits["val"]["y"].shape),
            "X_test": list(splits["test"]["x"].shape),
            "y_test": list(splits["test"]["y"].shape),
        },
    }
    upload_json(client, MINIO_GOLD_BUCKET, f"{prefix}/dataset_metadata.json", metadata)

    print(f"✅ Training dataset saved to s3://{MINIO_GOLD_BUCKET}/{prefix}")


if __name__ == "__main__":
    main()
