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

from minio import Minio

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# =============================
# 0) Config & Constants
# =============================
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost:9004")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "admin123")
MINIO_SILVER_BUCKET = os.environ.get("MINIO_SILVER_BUCKET", "air-quality-silver")
MINIO_GOLD_BUCKET = os.environ.get("MINIO_GOLD_BUCKET", "air-quality-gold")
MINIO_ARTIFACTS_BUCKET = os.environ.get("MINIO_ARTIFACTS_BUCKET", "air-quality-artifacts")
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

_PROJECT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "Conf" / "air_quality.yaml"


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

        print(f"✅ Loaded validation_rules from MinIO: {EDA_RULES_OBJECT}")
    except Exception as exc:
        print(f"⚠️  validation_rules.json not found on MinIO: {exc} — using defaults")

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

        print(f"✅ Loaded preprocessing_strategy from MinIO: {EDA_STRATEGY_OBJECT}")
    except Exception as exc:
        print(f"⚠️  preprocessing_strategy.json not found on MinIO: {exc} — using defaults")

    config.setdefault("metric_cols", _DEFAULT_METRIC_COLS)
    config.setdefault("physical_bounds", _DEFAULT_PHYSICAL_BOUNDS)
    config.setdefault("max_interpolate_gap_h", _DEFAULT_MAX_INTERPOLATE_GAP_H)
    config.setdefault("max_ffill_gap_h", _DEFAULT_MAX_FFILL_GAP_H)
    config.setdefault("normalization_method", "robust")
    config.setdefault("time_features", _DEFAULT_TIME_FEATURES)

    print(
        f"\n📋 Active config:"
        f"\n   metric_cols          : {config['metric_cols']}"
        f"\n   max_interpolate_gap_h: {config['max_interpolate_gap_h']}h"
        f"\n   max_ffill_gap_h      : {config['max_ffill_gap_h']}h"
        f"\n   normalization_method : {config['normalization_method']}"
        f"\n   time_features        : {config['time_features']}\n"
    )

    return config


def load_project_config(path: str | None = None) -> dict:
    """Load project YAML config from Conf/air_quality.yaml (or custom path)."""
    config_path = Path(path) if path else _PROJECT_CONFIG_PATH
    if not config_path.exists():
        print(f"⚠️  Project config not found: {config_path}")
        return {}
    if yaml is None:
        print("⚠️  PyYAML not installed; cannot read project config.")
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Failed to load project config: {exc}")
        return {}
