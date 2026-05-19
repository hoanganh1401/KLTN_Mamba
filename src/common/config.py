"""
config.py
=========
Cấu hình chung cho pipeline Air Quality Data Lake.

File này được tách từ Data_processing.py.
Nhiệm vụ:
- Khai báo biến môi trường MinIO/bucket.
- Khai báo fallback defaults.
- Load config từ EDA outputs:
  + eda_outputs/preprocessing_strategy.json
  + eda_outputs/validation_rules.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from minio import Minio


# =============================
# 0) Config & Constants
# =============================
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost:9004")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "admin123")
MINIO_SILVER_BUCKET = os.environ.get("MINIO_SILVER_BUCKET", "air-quality-silver")
MINIO_GOLD_BUCKET = os.environ.get("MINIO_GOLD_BUCKET", "air-quality-gold")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

# EDA config objects
EDA_STRATEGY_OBJECT = "eda_outputs/preprocessing_strategy.json"
EDA_RULES_OBJECT = "eda_outputs/validation_rules.json"

# Fallback defaults — chỉ dùng khi không load được JSON từ EDA
_DEFAULT_METRIC_COLS: list[str] = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2", "aqi",
]

_DEFAULT_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": (0, 1000),
    "pm10": (0, 2000),
    "no2": (0, 1000),
    "o3": (0, 500),
    "so2": (0, 1000),
    "co": (0, 50000),
    "aod": (0, 5),
    "dust": (0, 2000),
    "uv_index": (0, 20),
    "co2": (300, 5000),
    "aqi": (0, 500),
}

_DEFAULT_MAX_INTERPOLATE_GAP_H = 3
_DEFAULT_MAX_FFILL_GAP_H = 6
_DEFAULT_TIME_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend"
]

_DEFAULT_DATA_CONFIG = {
    "freq": "h",
    "time_col": "time",
    "location_col": "location",
    "target_col": "aqi",
    "metric_cols": _DEFAULT_METRIC_COLS,
}

_DEFAULT_FEATURES_CONFIG = {
    "time_features": _DEFAULT_TIME_FEATURES,
}

_DEFAULT_DATASET_CONFIG = {
    "seq_len": 96,
    "pred_len": 12,
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
    "include_target_history": True,
}

_DEFAULT_SCALING_CONFIG = {
    "method": "standard",
    "fit_on": "train_only",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_config(path: str | None = None) -> dict:
    """Load project config from Conf/air_quality.yaml.

    Returns a dict with normalized keys for data/features/dataset/scaling.
    """
    if path is None:
        path = os.environ.get("AIR_QUALITY_CONFIG", "Conf/air_quality.yaml")
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = _project_root() / cfg_path

    if not cfg_path.exists():
        return {
            "data": _DEFAULT_DATA_CONFIG,
            "features": _DEFAULT_FEATURES_CONFIG,
            "dataset": _DEFAULT_DATASET_CONFIG,
            "scaling": _DEFAULT_SCALING_CONFIG,
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = {**_DEFAULT_DATA_CONFIG, **(raw.get("data") or {})}
    features_cfg = {**_DEFAULT_FEATURES_CONFIG, **(raw.get("features") or {})}
    dataset_cfg = {**_DEFAULT_DATASET_CONFIG, **(raw.get("dataset") or {})}
    scaling_cfg = {**_DEFAULT_SCALING_CONFIG, **(raw.get("scaling") or {})}

    return {
        "project": raw.get("project", {}),
        "data": data_cfg,
        "features": features_cfg,
        "dataset": dataset_cfg,
        "scaling": scaling_cfg,
        "minio": raw.get("minio", {}),
        "model": raw.get("model", {}),
        "training": raw.get("training", {}),
        "inference": raw.get("inference", {}),
    }


def load_processing_config(client: Minio) -> dict:
    """
    Load preprocessing params từ hai file JSON do EDA notebook upload lên MinIO.

    Trả về dict chuẩn hoá với các key:
      metric_cols             : list[str]
      physical_bounds         : dict[str, tuple[float, float]]
      max_interpolate_gap_h   : int
      max_ffill_gap_h         : int
      normalization_method    : str   ("robust" | "standard" | "minmax")
      time_features           : list[str]

    Lưu ý:
    - Silver dùng: metric_cols, physical_bounds, max_interpolate_gap_h, max_ffill_gap_h.
    - Gold dùng: time_features.
    - Training preprocessing dùng: normalization_method.
    """
    config: dict = {}
    proj_cfg = load_project_config()

    data_cfg = proj_cfg.get("data", {})
    features_cfg = proj_cfg.get("features", {})
    dataset_cfg = proj_cfg.get("dataset", {})
    scaling_cfg = proj_cfg.get("scaling", {})

    config["time_col"] = data_cfg.get("time_col", "time")
    config["location_col"] = data_cfg.get("location_col", "location")
    config["target_col"] = data_cfg.get("target_col", "aqi")
    config["metric_cols"] = data_cfg.get("metric_cols", _DEFAULT_METRIC_COLS)
    config["time_features"] = features_cfg.get("time_features", _DEFAULT_TIME_FEATURES)
    config["normalization_method"] = scaling_cfg.get("method", "standard")
    config["dataset"] = dataset_cfg

    # Load validation_rules.json từ MinIO
    try:
        resp = client.get_object(MINIO_SILVER_BUCKET, EDA_RULES_OBJECT)
        try:
            vrules = json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
            resp.release_conn()

        raw_bounds = vrules.get("physical_bounds", {})
        config["physical_bounds"] = {
            col: (float(v[0]), float(v[1])) for col, v in raw_bounds.items()
        }

        identity = {"time", "location", "latitude", "longitude"}
        req_cols = vrules.get("required_columns", [])
        metric_cols = [c for c in req_cols if c not in identity]
        if metric_cols:
            config["metric_cols"] = metric_cols

        if "max_interpolation_gap_hours" in vrules:
            config["max_interpolate_gap_h"] = int(vrules["max_interpolation_gap_hours"])
        if "max_forward_fill_gap_hours" in vrules:
            config["max_ffill_gap_h"] = int(vrules["max_forward_fill_gap_hours"])

        print(f"OK: Loaded validation_rules from MinIO: {EDA_RULES_OBJECT}")
    except Exception as exc:
        print(f"WARN: validation_rules.json not found on MinIO: {exc} — using defaults")

    # Load preprocessing_strategy.json từ MinIO
    try:
        resp = client.get_object(MINIO_SILVER_BUCKET, EDA_STRATEGY_OBJECT)
        try:
            strategy = json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
            resp.release_conn()

        norm_cfg = strategy.get("3_normalization", {})
        rule_text = norm_cfg.get("rule", "").lower()
        if "robust" in rule_text:
            config["normalization_method"] = "robust"
        elif "standard" in rule_text:
            config["normalization_method"] = "standard"
        elif "minmax" in rule_text or "min-max" in rule_text:
            config["normalization_method"] = "minmax"
        else:
            config.setdefault("normalization_method", "robust")

        tf_cfg = strategy.get("4_time_features", {})
        tf_list = tf_cfg.get("features", [])
        if tf_list:
            config["time_features"] = tf_list

        print(f"OK: Loaded preprocessing_strategy from MinIO: {EDA_STRATEGY_OBJECT}")
    except Exception as exc:
        print(f"WARN: preprocessing_strategy.json not found on MinIO: {exc} — using defaults")

    config.setdefault("metric_cols", _DEFAULT_METRIC_COLS)
    config.setdefault("physical_bounds", _DEFAULT_PHYSICAL_BOUNDS)
    config.setdefault("max_interpolate_gap_h", _DEFAULT_MAX_INTERPOLATE_GAP_H)
    config.setdefault("max_ffill_gap_h", _DEFAULT_MAX_FFILL_GAP_H)
    config.setdefault("normalization_method", "standard")
    config.setdefault("time_features", _DEFAULT_TIME_FEATURES)

    print(
        "\nActive config:"
        f"\n  time_col             : {config['time_col']}"
        f"\n  location_col         : {config['location_col']}"
        f"\n  target_col           : {config['target_col']}"
        f"\n  metric_cols          : {config['metric_cols']}"
        f"\n  max_interpolate_gap_h: {config['max_interpolate_gap_h']}h"
        f"\n  max_ffill_gap_h      : {config['max_ffill_gap_h']}h"
        f"\n  normalization_method : {config['normalization_method']}"
        f"\n  time_features        : {config['time_features']}\n"
    )

    return config


def _apply_yaml_minio_overrides() -> None:
    cfg = load_project_config()
    minio_cfg = cfg.get("minio", {})
    global MINIO_SILVER_BUCKET, MINIO_GOLD_BUCKET
    if minio_cfg.get("silver_bucket"):
        MINIO_SILVER_BUCKET = str(minio_cfg["silver_bucket"])
    if minio_cfg.get("gold_bucket"):
        MINIO_GOLD_BUCKET = str(minio_cfg["gold_bucket"])


_apply_yaml_minio_overrides()
