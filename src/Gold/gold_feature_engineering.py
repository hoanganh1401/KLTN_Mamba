"""
gold_feature_engineering.py — Silver/processed → Gold/feature_engineering
=========================================================================
File này chỉ chứa tiến trình đã có sẵn trong Data_processing.py:
- Tạo cyclic time features: hour_sin, hour_cos, month_sin, month_cos
- Tạo calendar features: day_of_week, is_weekend

Chưa thêm lag/rolling/sliding-window vì file gốc chưa có các bước đó.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from minio import Minio

from config import MINIO_GOLD_BUCKET, load_processing_config
from minio_io import (
    get_client,
    gold_feature_path,
    load_silver_processed,
    upload_csv,
    upload_json,
)


def step_time_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Thêm cyclic time features theo danh sách được chỉ định từ EDA config.
    Supported: hour_sin, hour_cos, month_sin, month_cos, day_of_week, is_weekend
    """
    if "time" not in df.columns:
        raise ValueError("Missing required column: time")

    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    hour = t.dt.hour
    month = t.dt.month

    feature_map = {
        "hour_sin": lambda: np.sin(2 * np.pi * hour / 24),
        "hour_cos": lambda: np.cos(2 * np.pi * hour / 24),
        "month_sin": lambda: np.sin(2 * np.pi * month / 12),
        "month_cos": lambda: np.cos(2 * np.pi * month / 12),
        "day_of_week": lambda: t.dt.dayofweek.astype("int8"),
        "is_weekend": lambda: (t.dt.dayofweek >= 5).astype("int8"),
    }

    df = df.copy()
    df["time"] = t
    for feat in features:
        if feat in feature_map:
            df[feat] = feature_map[feat]()
        else:
            print(f"  [WARN] Unknown time feature '{feat}' — skipped")

    return df


def step_reorder_gold_features(df: pd.DataFrame, metric_cols: list[str], time_features: list[str]) -> pd.DataFrame:
    """Sắp xếp cột cho Gold features: identity → metrics → time features → flags → other."""
    identity = ["time", "location", "latitude", "longitude"]
    metrics = [c for c in metric_cols if c in df.columns]
    time_feat = [c for c in time_features if c in df.columns]
    flags = sorted([c for c in df.columns if c.startswith("_")])
    other = [c for c in df.columns if c not in identity + metrics + time_feat + flags]

    ordered = identity + metrics + time_feat + flags + other
    return df[[c for c in ordered if c in df.columns]]


def process_day(client: Minio, location_key: str, target_date: date, cfg: dict) -> dict:
    """Process 1 location × 1 ngày ở Gold feature engineering."""
    year, month, day = target_date.year, target_date.month, target_date.day
    print(f"\n── GOLD FEATURES | {location_key} | {target_date} ──")

    log: dict = {"layer": "gold_feature_engineering", "location": location_key, "date": str(target_date)}

    df = load_silver_processed(client, location_key, year, month, day)
    if df is None:
        log["status"] = "SKIP_NO_DATA"
        return log

    log["rows_input"] = len(df)

    df = step_time_features(df, cfg["time_features"])
    log["step_time_features"] = cfg["time_features"]

    df = step_reorder_gold_features(df, cfg["metric_cols"], cfg["time_features"])

    log["rows_output"] = len(df)
    log["status"] = "OK"

    path = gold_feature_path(location_key, year, month, day)
    upload_csv(client, MINIO_GOLD_BUCKET, path, df)
    print(f"  ✅ Gold features saved: s3://{MINIO_GOLD_BUCKET}/{path} ({len(df)} rows)")

    return log


def run_feature_engineering(locations_path: str, target_date_str: str) -> None:
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
    log_path = f"feature_engineering_logs/year={year}/month={month:02d}/day={day:02d}/feature_log.json"
    summary = {
        "target_date": target_date_str,
        "processed_at": datetime.utcnow().isoformat(),
        "total": len(all_logs),
        "ok": n_ok,
        "skipped": n_skip,
        "details": all_logs,
    }
    upload_json(client, MINIO_GOLD_BUCKET, log_path, summary)

    print(f"\n✅ GOLD FEATURE SUMMARY — {target_date_str}: processed={n_ok}, skipped={n_skip}")
    print(f"📄 Feature log: s3://{MINIO_GOLD_BUCKET}/{log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold feature engineering — time features only")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    target = args.date or (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    run_feature_engineering(args.locations, target)


if __name__ == "__main__":
    main()
