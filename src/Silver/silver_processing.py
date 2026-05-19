"""
silver_processing.py — Silver/validated → Silver/processed
===========================================================
Chạy sau data_validation.py trong Airflow pipeline.

Nhiệm vụ của file này chỉ thuộc lớp Silver:
- Deduplicate theo (time, location)
- Clip theo physical bounds
- Reindex hourly
- Impute missing values
- Gắn cờ _imputed và _invalid_segment
- Ghi ra air-quality-silver/processed/...

Các bước đã được tách ra ngoài:
- Normalize/scale → training_preprocessing.py
- Time features  → gold_feature_engineering.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import pandas as pd
from minio import Minio

from common.config import MINIO_SILVER_BUCKET, load_processing_config
from common.minio_io import (
    get_client,
    load_silver_validated,
    silver_processed_path,
    upload_csv,
    upload_json,
)


# =============================
# Silver Processing Steps
# =============================
def step_dedup(df: pd.DataFrame, time_col: str, location_col: str) -> tuple[pd.DataFrame, int]:
    """Remove duplicate (time, location), keep last."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[time_col, location_col], keep="last")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df, n_before - len(df)


def step_clip(df: pd.DataFrame, physical_bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip values về physical bounds. Không remove row, chỉ clip."""
    for col, (lo, hi) in physical_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def step_impute(
    df: pd.DataFrame,
    time_col: str,
    location_col: str,
    metric_cols: list[str],
    max_interpolate_gap_h: int,
    max_ffill_gap_h: int,
    target_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    - Reindex thành chuỗi hourly liên tục trong khoảng [min_time, max_time]
    - Interpolate linear cho gap <= max_interpolate_gap_h
    - Forward-fill cho gap <= max_ffill_gap_h
    - Đánh dấu _imputed=True cho các row được tạo thêm bởi reindex
    - Đánh dấu _invalid_segment=True cho các row nằm trong gap > max_ffill_gap_h
    """
    if df.empty or df[time_col].isna().all():
        return df, {}

    df = df.set_index(time_col)

    full_idx = pd.date_range(
        start=df.index.min().floor("h"),
        end=df.index.max().ceil("h"),
        freq="1h",
        tz="UTC",
    )
    df = df.reindex(full_idx)
    df["_imputed"] = df[location_col].isna()

    for col in [location_col, "latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    df["_invalid_segment"] = False
    missing_mask = df[target_col].isna() if target_col in df.columns else pd.Series(False, index=df.index)

    if missing_mask.any():
        run_id = (missing_mask != missing_mask.shift()).cumsum()
        for _, group in df[missing_mask].groupby(run_id[missing_mask]):
            if len(group) > max_ffill_gap_h:
                df.loc[group.index, "_invalid_segment"] = True

    for col in metric_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].interpolate(
            method="time",
            limit=max_interpolate_gap_h,
            limit_direction="both",
        )

    for col in metric_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].ffill(limit=max_ffill_gap_h)

    flag_cols = [c for c in df.columns if c.startswith("_flag_")]
    for c in flag_cols:
        df[c] = df[c].fillna(False)

    stats = {
        "rows_added_by_reindex": int(df["_imputed"].sum()),
        "invalid_segment_rows": int(df["_invalid_segment"].sum()),
        "remaining_null_target": int(df[target_col].isna().sum()) if target_col in df.columns else -1,
    }

    df = df.reset_index().rename(columns={"index": time_col})
    return df, stats


def step_reorder_silver(df: pd.DataFrame, metric_cols: list[str], time_col: str, location_col: str) -> pd.DataFrame:
    """Sắp xếp lại cột cho Silver: identity → metrics → flags → other."""
    identity = [time_col, location_col, "latitude", "longitude"]
    metrics = [c for c in metric_cols if c in df.columns]
    flags = sorted([c for c in df.columns if c.startswith("_")])
    other = [c for c in df.columns if c not in identity + metrics + flags]

    ordered = identity + metrics + flags + other
    return df[[c for c in ordered if c in df.columns]]


# =============================
# Main Silver Logic
# =============================
def process_day(
    client: Minio,
    location_key: str,
    target_date: date,
    cfg: dict,
) -> dict:
    """Process 1 location × 1 ngày ở lớp Silver."""
    year, month, day = target_date.year, target_date.month, target_date.day
    print(f"\n── SILVER | {location_key} | {target_date} ──")

    log: dict = {"layer": "silver", "location": location_key, "date": str(target_date)}

    df = load_silver_validated(client, location_key, year, month, day)
    if df is None:
        log["status"] = "SKIP_NO_DATA"
        return log

    log["rows_input"] = len(df)

    df, n_dup = step_dedup(df, cfg["time_col"], cfg["location_col"])
    log["step1_dedup_removed"] = n_dup

    df = step_clip(df, cfg["physical_bounds"])
    log["step2_clip"] = "done"

    df, impute_stats = step_impute(
        df,
        time_col=cfg["time_col"],
        location_col=cfg["location_col"],
        metric_cols=cfg["metric_cols"],
        max_interpolate_gap_h=cfg["max_interpolate_gap_h"],
        max_ffill_gap_h=cfg["max_ffill_gap_h"],
        target_col=cfg["target_col"],
    )
    log["step3_impute"] = impute_stats

    df = step_reorder_silver(df, cfg["metric_cols"], cfg["time_col"], cfg["location_col"])

    log["rows_output"] = len(df)
    log["status"] = "OK"

    path = silver_processed_path(location_key, year, month, day)
    upload_csv(client, MINIO_SILVER_BUCKET, path, df)
    print(f"  ✅ Silver processed saved: s3://{MINIO_SILVER_BUCKET}/{path} ({len(df)} rows)")

    return log


def run_processing(locations_path: str, target_date_str: str) -> None:
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
    log_path = f"processing_logs/silver/year={year}/month={month:02d}/day={day:02d}/processing_log.json"

    cfg_serializable = {
        **cfg,
        "physical_bounds": {col: list(v) for col, v in cfg["physical_bounds"].items()},
    }
    summary = {
        "target_date": target_date_str,
        "processed_at": datetime.utcnow().isoformat(),
        "config_used": cfg_serializable,
        "total": len(all_logs),
        "ok": n_ok,
        "skipped": n_skip,
        "details": all_logs,
    }
    upload_json(client, MINIO_SILVER_BUCKET, log_path, summary)

    print(f"\n✅ SILVER SUMMARY — {target_date_str}: processed={n_ok}, skipped={n_skip}")
    print(f"📄 Processing log: s3://{MINIO_SILVER_BUCKET}/{log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver processing — clean, clip, impute")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    target = args.date or (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    run_processing(args.locations, target)


if __name__ == "__main__":
    main()
