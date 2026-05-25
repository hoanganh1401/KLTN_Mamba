"""
prepare_inference_input.py — Gold/features → Gold/inference_input
=================================================================
Build the latest inference input for Mamba using Gold feature data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common.config import MINIO_GOLD_BUCKET, load_project_config
from common.minio_io import (
    get_client,
    load_gold_features,
    load_json_object,
    load_pickle,
    upload_json,
    upload_npy,
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


def collect_recent_features(
    locations: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    client = get_client()
    frames: list[pd.DataFrame] = []

    for loc_key in locations:
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


def get_quality_config(project_cfg: dict, metadata: dict) -> dict:
    quality_cfg = metadata.get("quality_gate") or project_cfg.get("quality", {}) or {}
    return {
        "hard_fail_cols": quality_cfg.get("hard_fail_cols") or DEFAULT_HARD_FAIL_COLS,
        "imputed_col": quality_cfg.get("imputed_col", DEFAULT_IMPUTED_COL),
        "max_imputed_ratio": float(quality_cfg.get("max_imputed_ratio", 0.3)),
        "low_coverage_col": quality_cfg.get("low_coverage_col", DEFAULT_LOW_COVERAGE_COL),
        "max_low_coverage_ratio": float(quality_cfg.get("max_low_coverage_ratio", 0.5)),
    }


def apply_scaler(x: np.ndarray, scaler, metric_idx: list[int]) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    flat[:, metric_idx] = scaler.transform(flat[:, metric_idx])
    return flat.reshape(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare inference input for Mamba")
    parser.add_argument("--locations", required=True, help="Path to JSONL locations file")
    parser.add_argument("--config", default=None, help="Path to project YAML config")
    parser.add_argument("--location-keys", default=None, help="Comma-separated location keys override")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--lookback-days", type=int, default=14, help="Days to scan for latest data")
    parser.add_argument("--dataset-prefix", default=None, help="MinIO prefix to training dataset artifacts")
    parser.add_argument("--run-id", default=None, help="Training run id to locate scaler/metadata")
    parser.add_argument("--output-run-id", default=None, help="Optional run id for inference output")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len from metadata")
    args = parser.parse_args()

    if not args.dataset_prefix:
        if not args.run_id:
            raise ValueError("Provide --dataset-prefix or --run-id for training artifacts.")
        dataset_prefix = f"training_dataset/run_id={args.run_id}"
    else:
        dataset_prefix = args.dataset_prefix.rstrip("/")

    client = get_client()
    meta = load_json_object(client, MINIO_GOLD_BUCKET, f"{dataset_prefix}/dataset_metadata.json")
    if meta is None:
        raise ValueError("Training dataset metadata not found.")

    scaler = load_pickle(client, MINIO_GOLD_BUCKET, f"{dataset_prefix}/scaler.pkl")
    if scaler is None:
        raise ValueError("Scaler not found in training dataset prefix.")

    project_cfg = load_project_config(args.config)
    dataset_cfg = project_cfg.get("dataset", {})
    inference_cfg = project_cfg.get("inference", {})

    default_seq = inference_cfg.get("use_latest_hours") or dataset_cfg.get("seq_len")
    seq_len = args.seq_len or int(meta.get("seq_len", default_seq or 96))
    feature_cols = meta.get("feature_cols", [])
    metric_cols = meta.get("metric_cols", [])
    quality_cfg = get_quality_config(project_cfg, meta)

    if not feature_cols or not metric_cols:
        raise ValueError("Metadata missing feature_cols or metric_cols.")

    if args.location_keys:
        location_keys = [x.strip() for x in args.location_keys.split(",") if x.strip()]
    else:
        location_keys = [
            loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
            for loc in load_locations(args.locations)
        ]

    end_default = datetime.utcnow().date()
    end_date = parse_date(args.end_date, end_default)
    start_date = end_date - timedelta(days=max(1, args.lookback_days) - 1)

    data = collect_recent_features(location_keys, start_date, end_date)

    metric_idx = [feature_cols.index(c) for c in metric_cols if c in feature_cols]

    x_batches: list[np.ndarray] = []
    used_locations: list[str] = []
    ranges: dict[str, dict[str, str]] = {}
    skipped_locations: dict[str, str] = {}

    for loc in location_keys:
        g = data.loc[data["location_key"].astype(str) == str(loc)].copy()
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < seq_len:
            skipped_locations[str(loc)] = "not_enough_history"
            continue

        latest = g.tail(seq_len).copy()
        missing_features = [c for c in feature_cols if c not in latest.columns]
        if missing_features:
            skipped_locations[str(loc)] = f"missing_features:{','.join(missing_features)}"
            continue

        ok, reason = window_quality_status(
            latest,
            hard_fail_cols=quality_cfg["hard_fail_cols"],
            imputed_col=quality_cfg["imputed_col"],
            max_imputed_ratio=quality_cfg["max_imputed_ratio"],
            low_coverage_col=quality_cfg["low_coverage_col"],
            max_low_coverage_ratio=quality_cfg["max_low_coverage_ratio"],
        )
        if not ok:
            skipped_locations[str(loc)] = reason
            continue

        x = latest[feature_cols].to_numpy(dtype=np.float32)
        if np.isnan(x).any():
            skipped_locations[str(loc)] = "nan_feature"
            continue

        x = apply_scaler(x.reshape(1, seq_len, -1), scaler, metric_idx)

        x_batches.append(x[0])
        used_locations.append(str(loc))
        ranges[str(loc)] = {
            "start": latest["time"].iloc[0].isoformat(),
            "end": latest["time"].iloc[-1].isoformat(),
        }

    if not x_batches:
        reason_counts = Counter(skipped_locations.values())
        print("Inference skipped locations summary:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
        examples = list(skipped_locations.items())[:10]
        if examples:
            print("Examples:")
            for loc, reason in examples:
                print(f"  {loc}: {reason}")
        raise ValueError(
            f"No valid locations with enough history for inference. "
            f"Need seq_len={seq_len} rows per location after quality gate."
        )

    x_infer = np.stack(x_batches, axis=0)

    run_id = args.output_run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_prefix = f"inference_input/run_id={run_id}"

    upload_npy(client, MINIO_GOLD_BUCKET, f"{out_prefix}/X_inference.npy", x_infer.astype(np.float32))
    upload_json(
        client,
        MINIO_GOLD_BUCKET,
        f"{out_prefix}/inference_metadata.json",
        {
            "run_id": run_id,
            "dataset_prefix": dataset_prefix,
            "seq_len": seq_len,
            "feature_cols": feature_cols,
            "metric_cols": metric_cols,
            "quality_gate": quality_cfg,
            "locations": used_locations,
            "skipped_locations": skipped_locations,
            "ranges": ranges,
            "shape": list(x_infer.shape),
        },
    )

    print(f"✅ Inference input saved to s3://{MINIO_GOLD_BUCKET}/{out_prefix}")


if __name__ == "__main__":
    main()
