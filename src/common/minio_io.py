"""
minio_io.py
===========
Các helper đọc/ghi MinIO và chuẩn hoá path object.

File này được tách từ Data_processing.py.
"""

from __future__ import annotations

import io
import json
import pickle
from typing import Any

import pandas as pd
import numpy as np
from minio import Minio

try:
    from .config import (
        MINIO_ACCESS,
        MINIO_ARTIFACTS_BUCKET,
        MINIO_GOLD_BUCKET,
        MINIO_HOST,
        MINIO_SECRET,
        MINIO_SECURE,
        MINIO_SILVER_BUCKET,
    )
except ImportError:
    from config import (
        MINIO_ACCESS,
        MINIO_ARTIFACTS_BUCKET,
        MINIO_GOLD_BUCKET,
        MINIO_HOST,
        MINIO_SECRET,
        MINIO_SECURE,
        MINIO_SILVER_BUCKET,
    )

try:
    from .time_utils import parse_time_local
except ImportError:
    from time_utils import parse_time_local


def get_client() -> Minio:
    return Minio(
        MINIO_HOST,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"  [INFO] Created MinIO bucket: {bucket}")


# =============================
# Object paths
# =============================
def silver_validated_path(location_key: str, year: int, month: int, day: int) -> str:
    """Silver input: dữ liệu đã qua validation."""
    return (
        f"validated/province={location_key}"
        f"/year={year}/month={month:02d}/day={day:02d}/data.csv"
    )


def silver_processed_path(location_key: str, year: int, month: int, day: int) -> str:
    """Silver output: dữ liệu sạch/processed, chưa tạo feature model."""
    return (
        f"processed/province={location_key}"
        f"/year={year}/month={month:02d}/day={day:02d}/data.csv"
    )


def gold_feature_path(location_key: str, year: int, month: int, day: int) -> str:
    """Gold feature engineering output."""
    return (
        f"feature_engineering/province={location_key}"
        f"/year={year}/month={month:02d}/day={day:02d}/data.csv"
    )


def gold_train_preprocessed_path(location_key: str, year: int, month: int, day: int) -> str:
    """Gold/training preprocessing output: dữ liệu đã scale theo logic hiện có."""
    return (
        f"train_preprocessed/province={location_key}"
        f"/year={year}/month={month:02d}/day={day:02d}/data.csv"
    )


# =============================
# Read/write helpers
# =============================
def load_csv_object(client: Minio, bucket: str, path: str) -> pd.DataFrame | None:
    try:
        resp = client.get_object(bucket, path)
        try:
            raw = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        df = pd.read_csv(io.BytesIO(raw))
        if "time" not in df.columns:
            for col in df.columns:
                if col.lower() == "time":
                    df = df.rename(columns={col: "time"})
                    break
        if "time" in df.columns:
            df["time"] = parse_time_local(df["time"])
        return df
    except Exception as exc:
        print(f"  [WARN] Object not found: s3://{bucket}/{path} — {exc}")
        return None


def upload_csv(client: Minio, bucket: str, path: str, df: pd.DataFrame) -> None:
    ensure_bucket(client, bucket)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(
        bucket,
        path,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def upload_json(client: Minio, bucket: str, path: str, obj: dict[str, Any]) -> None:
    ensure_bucket(client, bucket)
    j_bytes = json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    client.put_object(
        bucket,
        path,
        data=io.BytesIO(j_bytes),
        length=len(j_bytes),
        content_type="application/json",
    )


def load_silver_validated(client: Minio, location_key: str, year: int, month: int, day: int) -> pd.DataFrame | None:
    return load_csv_object(
        client,
        MINIO_SILVER_BUCKET,
        silver_validated_path(location_key, year, month, day),
    )


def load_silver_processed(client: Minio, location_key: str, year: int, month: int, day: int) -> pd.DataFrame | None:
    return load_csv_object(
        client,
        MINIO_SILVER_BUCKET,
        silver_processed_path(location_key, year, month, day),
    )


def load_gold_features(client: Minio, location_key: str, year: int, month: int, day: int) -> pd.DataFrame | None:
    return load_csv_object(
        client,
        MINIO_GOLD_BUCKET,
        gold_feature_path(location_key, year, month, day),
    )


def load_json_object(client: Minio, bucket: str, path: str) -> dict | None:
    try:
        resp = client.get_object(bucket, path)
        try:
            raw = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(f"  [WARN] Object not found: s3://{bucket}/{path} — {exc}")
        return None


def load_bytes(client: Minio, bucket: str, path: str) -> bytes | None:
    try:
        resp = client.get_object(bucket, path)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except Exception as exc:
        print(f"  [WARN] Object not found: s3://{bucket}/{path} — {exc}")
        return None


def upload_bytes(
    client: Minio,
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    ensure_bucket(client, bucket)
    client.put_object(
        bucket,
        path,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def upload_npy(client: Minio, bucket: str, path: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    upload_bytes(client, bucket, path, buf.getvalue())


def load_npy(client: Minio, bucket: str, path: str) -> np.ndarray | None:
    raw = load_bytes(client, bucket, path)
    if raw is None:
        return None
    return np.load(io.BytesIO(raw), allow_pickle=False)


def upload_pickle(client: Minio, bucket: str, path: str, obj: object) -> None:
    data = pickle.dumps(obj)
    upload_bytes(client, bucket, path, data, content_type="application/octet-stream")


def load_pickle(client: Minio, bucket: str, path: str) -> object | None:
    raw = load_bytes(client, bucket, path)
    if raw is None:
        return None
    return pickle.loads(raw)
