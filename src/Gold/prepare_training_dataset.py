"""
prepare_training_dataset.py — Gold/feature_engineering → Gold/train_preprocessed
==============================================================================
File này chứa tiến trình normalize/scale đã có sẵn trong Data_processing.py.

Lưu ý quan trọng:
- Logic hiện tại fit scaler trên phần dữ liệu đang được xử lý và bỏ qua _invalid_segment.
- Khi bạn triển khai train/val/test split, nên sửa bước này để:
  fit scaler chỉ trên train set, sau đó transform val/test.
- File này chưa tạo sliding window vì Data_processing.py gốc chưa có bước đó.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import pandas as pd
from minio import Minio

from common.config import MINIO_GOLD_BUCKET, load_processing_config
from common.minio_io import (
    get_client,
    gold_train_preprocessed_path,
    load_gold_features,
    upload_csv,
    upload_json,
)


def step_normalize(
    df: pd.DataFrame,
    metric_cols: list[str],
    method: str = "robust",
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize per-location per-column.
      "robust"   → RobustScaler:  (x - median) / IQR
      "standard" → StandardScaler: (x - mean) / std
      "minmax"   → MinMaxScaler:  (x - min) / (max - min)

    Fit chỉ trên các row KHÔNG bị flag invalid.
    """
    df = df.copy()
    scaler_params: dict[str, dict] = {}
    valid_mask = ~df.get("_invalid_segment", pd.Series(False, index=df.index))

    for col in metric_cols:
        if col not in df.columns:
            continue
        valid_vals = df.loc[valid_mask, col].dropna()
        if len(valid_vals) < 4:
            continue

        if method == "robust":
            center = valid_vals.median()
            q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
            scale = (q3 - q1) if (q3 - q1) > 0 else 1.0
            df[f"{col}_scaled"] = (df[col] - center) / scale
            scaler_params[col] = {
                "method": "robust",
                "median": round(center, 4),
                "iqr": round(scale, 4),
            }

        elif method == "standard":
            center = valid_vals.mean()
            scale = valid_vals.std() if valid_vals.std() > 0 else 1.0
            df[f"{col}_scaled"] = (df[col] - center) / scale
            scaler_params[col] = {
                "method": "standard",
                "mean": round(center, 4),
                "std": round(scale, 4),
            }

        elif method == "minmax":
            vmin, vmax = valid_vals.min(), valid_vals.max()
            scale = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
            df[f"{col}_scaled"] = (df[col] - vmin) / scale
            scaler_params[col] = {
                "method": "minmax",
                "min": round(vmin, 4),
                "max": round(vmax, 4),
            }

        else:
            raise ValueError(
                f"Unknown normalization method: {method!r}. "
                "Choose 'robust', 'standard', or 'minmax'."
            )

    return df, scaler_params


def step_reorder_training(df: pd.DataFrame, metric_cols: list[str], time_features: list[str]) -> pd.DataFrame:
    """Sắp xếp cột sau normalize: identity → metrics → scaled → time features → flags → other."""
    identity = ["time", "location", "latitude", "longitude"]
    metrics = [c for c in metric_cols if c in df.columns]
    scaled = [f"{c}_scaled" for c in metric_cols if f"{c}_scaled" in df.columns]
    time_feat = [c for c in time_features if c in df.columns]
    flags = sorted([c for c in df.columns if c.startswith("_")])
    other = [c for c in df.columns if c not in identity + metrics + scaled + time_feat + flags]

    ordered = identity + metrics + scaled + time_feat + flags + other
    return df[[c for c in ordered if c in df.columns]]


def process_day(client: Minio, location_key: str, target_date: date, cfg: dict) -> dict:
    """Process 1 location × 1 ngày cho training preprocessing."""
    year, month, day = target_date.year, target_date.month, target_date.day
    print(f"\n── TRAIN PREPROCESSING | {location_key} | {target_date} ──")

    log: dict = {"layer": "training_preprocessing", "location": location_key, "date": str(target_date)}

    df = load_gold_features(client, location_key, year, month, day)
    if df is None:
        log["status"] = "SKIP_NO_DATA"
        return log

    log["rows_input"] = len(df)

    df, scaler_params = step_normalize(
        df,
        metric_cols=cfg["metric_cols"],
        method=cfg["normalization_method"],
    )
    log["scaler_params"] = scaler_params
    log["normalization_method"] = cfg["normalization_method"]

    df = step_reorder_training(df, cfg["metric_cols"], cfg["time_features"])

    log["rows_output"] = len(df)
    log["status"] = "OK"

    path = gold_train_preprocessed_path(location_key, year, month, day)
    upload_csv(client, MINIO_GOLD_BUCKET, path, df)
    print(f"  ✅ Train preprocessed saved: s3://{MINIO_GOLD_BUCKET}/{path} ({len(df)} rows)")

    return log


def run_training_preprocessing(locations_path: str, target_date_str: str) -> None:
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    client = get_client()
    cfg = load_processing_config(client)

    with open(locations_path, encoding="utf-8") as f:
        locations = [json.loads(line) for line in f if line.strip()]

    all_logs = []
    for loc in locations:
        loc_key = loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
        all_logs.append(process_day(client, loc_key, target_date, cfg))

    n_ok = sum(1 for l in all_logs if l.get("status") == "OK")
    n_skip = sum(1 for l in all_logs if l.get("status") != "OK")

    year, month, day = target_date.year, target_date.month, target_date.day
    log_path = f"train_preprocessing_logs/year={year}/month={month:02d}/day={day:02d}/train_preprocessing_log.json"
    summary = {
        "target_date": target_date_str,
        "processed_at": datetime.utcnow().isoformat(),
        "normalization_method": cfg["normalization_method"],
        "total": len(all_logs),
        "ok": n_ok,
        "skipped": n_skip,
        "details": all_logs,
    }
    upload_json(client, MINIO_GOLD_BUCKET, log_path, summary)

    print(f"\n✅ TRAIN PREPROCESSING SUMMARY — {target_date_str}: processed={n_ok}, skipped={n_skip}")
    print(f"📄 Train preprocessing log: s3://{MINIO_GOLD_BUCKET}/{log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training preprocessing — normalize existing features")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    target = args.date or (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    run_training_preprocessing(args.locations, target)


if __name__ == "__main__":
    main()
