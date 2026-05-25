"""
src/Model/train_mamba_aqi.py
------------------------
Script huấn luyện Mamba cho bài toán dự đoán AQI.

Import từ core:
    core.data_structs  → SplitData, AQIDataset
    core.metrics       → compute_metrics, denormalize
    core.utils         → setup_logger, resolve_device, set_seed

Import từ mamba:
    mamba.mamba_model  → TimeSeriesMambaRegressor

Chạy:
    python src/Model/train_mamba_aqi.py --data-path path/to/debug.csv --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# --- Đảm bảo Python tìm thấy thư mục gốc của project khi chạy trực tiếp ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_MODEL_ROOT = _SRC_ROOT / "Model"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT), str(_MODEL_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.core.data_structs import AQIDataset, SplitData
from src.core.metrics import compute_metrics, denormalize
from src.core.utils import resolve_device, set_seed, setup_logger
from src.Inference.predict_aqi_next import build_future_24h_frame, format_time_utc_strings
from src.Model.mamba_model import TimeSeriesMambaRegressor
from src.common.config import MINIO_ARTIFACTS_BUCKET, MINIO_GOLD_BUCKET, load_project_config
from src.common.minio_io import get_client, load_bytes, load_json_object, load_npy, upload_bytes, upload_json


# ---------------------------------------------------------------------------
# Xây dựng samples time-series
# ---------------------------------------------------------------------------

def build_time_series_samples(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    horizon: int,
    feature_cols: list[str] | None = None,
    include_target_history: bool = True,
    sample_stride: int = 1,
):
    """Tạo sliding-window samples từ DataFrame time-series nhiều location.

    Parameters
    ----------
    df          : DataFrame goc, can co cot time, location/location_key, target_col
    target_col  : tên cột target cần dự đoán
    window_size : số timestep đầu vào (T)
    horizon     : dự đoán y(t + horizon)

    Returns
    -------
    x_seq        : (N, T, F) float32
    loc_ids      : (N,)      int64
    y            : (N, H)    float32
    y_ts         : (N,)      datetime64[ns]
    num_locations: int
    feature_cols : list[str] — tên các cột feature được dùng
    """
    # Validate — auto-detect timestamp column (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    if "ts_utc" in col_map:
        ts_col = col_map["ts_utc"]
    elif "time" in col_map:
        ts_col = col_map["time"]
    elif "timestamp" in col_map:
        ts_col = col_map["timestamp"]
    else:
        raise ValueError(
            "Cot timestamp khong tim thay trong dataset. "
            "Can co cot 'time' (uu tien), 'ts_utc', hoac 'timestamp'."
        )

    work = df.copy()
    if "location_key" not in work.columns:
        if "location" in work.columns:
            work["location_key"] = work["location"].astype(str)
        else:
            work["location_key"] = "default"

    for col, label in [(target_col, "target"), (ts_col, "timestamp"), ("location_key", "location")]:
        if col not in work.columns:
            raise ValueError(f"Cot {label} '{col}' khong tim thay trong dataset.")
    if window_size < 1:
        raise ValueError("window_size phải >= 1.")
    if horizon < 1:
        raise ValueError("horizon phải >= 1.")

    if sample_stride < 1:
        raise ValueError("sample_stride phải >= 1.")

    work["_ts"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    missing_required = work[["_ts", "location_key", target_col]].isna().any(axis=1)
    if missing_required.any():
        raise ValueError(f"Du lieu chua NaN o {ts_col}/location_key/target. Vui long lam sach truoc.")

    work["_loc_id"] = work["location_key"].astype("category").cat.codes.astype(np.int64)
    num_locations   = int(work["_loc_id"].max()) + 1

    # Chọn feature: ưu tiên feature_cols nếu được truyền vào
    if feature_cols is None:
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        for col in [target_col, "_loc_id"]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        if include_target_history and target_col in work.columns:
            numeric_cols.append(target_col)
    else:
        numeric_cols = [c for c in feature_cols if c in work.columns and c != "_loc_id"]
        if include_target_history and target_col in work.columns and target_col not in numeric_cols:
            numeric_cols.append(target_col)
        if not include_target_history and target_col in numeric_cols:
            numeric_cols.remove(target_col)

    if not numeric_cols:
        raise ValueError("Không tìm thấy cột feature numeric nào sau khi lọc.")

    # Ép numeric, forward-fill theo location; đánh dấu row impute để bỏ khỏi train
    cols_to_fill = list(dict.fromkeys(numeric_cols + [target_col]))
    for col in cols_to_fill:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    all_nan_cols = [c for c in cols_to_fill if work[c].isna().all()]
    if target_col in all_nan_cols:
        raise ValueError("Target cột toàn NaN, không thể tạo sample.")
    if all_nan_cols:
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]
        cols_to_fill = [c for c in cols_to_fill if c not in all_nan_cols]

    missing_mask = work[cols_to_fill].isna().any(axis=1)
    work[cols_to_fill] = (
        work.groupby("_loc_id", sort=False)[cols_to_fill]
        .ffill()
    )
    still_missing = work[cols_to_fill].isna().any(axis=1)
    work["_invalid"] = missing_mask | still_missing

    work = work.sort_values(["_loc_id", "_ts"]).reset_index(drop=True)

    x_seq_list, loc_id_list, y_list, y_ts_list = [], [], [], []

    for loc_id, group in work.groupby("_loc_id", sort=False):
        x_vals = group[numeric_cols].to_numpy(dtype=np.float32)
        y_vals = group[target_col].to_numpy(dtype=np.float32)
        ts_vals = group["_ts"].to_numpy(dtype="datetime64[ns]")
        invalid = group["_invalid"].to_numpy(dtype=np.int64)
        n = len(group)

        max_start = n - window_size - horizon + 1
        if max_start <= 0:
            continue

        invalid_prefix = np.concatenate([[0], np.cumsum(invalid)])
        starts = np.arange(0, max_start, sample_stride, dtype=np.int64)
        ends = starts + window_size
        target_ends = ends + horizon

        valid_x = (invalid_prefix[ends] - invalid_prefix[starts]) == 0
        valid_y = (invalid_prefix[target_ends] - invalid_prefix[ends]) == 0
        valid = valid_x & valid_y
        if not np.any(valid):
            continue

        row_s, feat_s = x_vals.strides
        x_all = np.lib.stride_tricks.as_strided(
            x_vals,
            shape=(len(starts), window_size, x_vals.shape[1]),
            strides=(sample_stride * row_s, row_s, feat_s),
        )

        valid_starts = starts[valid]
        valid_ends = ends[valid]
        valid_target_ends = target_ends[valid]
        y_idx = valid_ends[:, None] + np.arange(horizon, dtype=np.int64)[None, :]

        x_seq_list.append(x_all[valid].copy())
        loc_id_list.append(np.full(len(valid_starts), loc_id, dtype=np.int64))
        y_list.append(y_vals[y_idx].astype(np.float32, copy=False))
        y_ts_list.append(ts_vals[valid_target_ends - 1])

    if not x_seq_list:
        raise ValueError(
            "Không tạo được sample nào. "
            "Thử giảm --window-size / --horizon hoặc cung cấp nhiều dữ liệu hơn."
        )

    return (
        np.concatenate(x_seq_list, axis=0).astype(np.float32),
        np.concatenate(loc_id_list, axis=0).astype(np.int64),
        np.concatenate(y_list, axis=0).astype(np.float32),
        np.concatenate(y_ts_list, axis=0).astype("datetime64[ns]"),
        num_locations,
        numeric_cols,
    )


# ---------------------------------------------------------------------------
# Split theo timeline
# ---------------------------------------------------------------------------

def split_data_by_timeline(
    x_seq:   np.ndarray,
    loc_ids: np.ndarray,
    y:       np.ndarray,
    y_ts:    np.ndarray,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.1,
) -> tuple[SplitData, SplitData, SplitData]:
    """Chia dữ liệu theo thứ tự thời gian (không shuffle).

    Train 70% → Val 10% → Test 20% (mặc định).
    """
    if len(y) < 3:
        raise ValueError("Cần ít nhất 3 sample để chia train/val/test.")

    order     = np.argsort(y_ts)
    n         = len(order)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            "Kích thước split không hợp lệ. Cần nhiều sample hơn hoặc điều chỉnh ratio."
        )

    def _take(idx):
        return SplitData(
            x_seq=x_seq[idx],
            loc_ids=loc_ids[idx],
            y=y[idx],
        )

    return (
        _take(order[:train_end]),
        _take(order[train_end:val_end]),
        _take(order[val_end:]),
    )


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def standardize(
    train: SplitData,
    val:   SplitData,
    test:  SplitData,
) -> tuple[SplitData, SplitData, SplitData, np.ndarray, np.ndarray, float, float]:
    """Chuẩn hoá x_seq và y dựa trên thống kê của tập train.

    Returns
    -------
    train, val, test (đã normalize), x_mean, x_std, y_mean, y_std
    """
    # X: normalize per-feature theo toàn bộ timestep của train
    x_mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std  = train.x_seq.std(axis=(0, 1), keepdims=True)
    x_std  = np.where(x_std < 1e-6, 1.0, x_std)

    for split in [train, val, test]:
        split.x_seq = (split.x_seq - x_mean) / x_std

    # Y: normalize scalar
    y_mean = float(train.y.mean())
    y_std  = float(train.y.std())
    if y_std < 1e-6:
        y_std = 1.0

    for split in [train, val, test]:
        split.y = (split.y - y_mean) / y_std

    return train, val, test, x_mean, x_std, y_mean, y_std


def standardize_targets(
    train: SplitData,
    val: SplitData,
    test: SplitData,
) -> tuple[SplitData, SplitData, SplitData, float, float]:
    """Normalize prepared target arrays using train split statistics."""
    y_mean = float(train.y.mean())
    y_std = float(train.y.std())
    if y_std < 1e-6:
        y_std = 1.0

    for split in (train, val, test):
        split.y = ((split.y - y_mean) / y_std).astype(np.float32)

    return train, val, test, y_mean, y_std


def _require_npy(client, bucket: str, path: str) -> np.ndarray:
    arr = load_npy(client, bucket, path)
    if arr is None:
        raise FileNotFoundError(f"Missing MinIO object: s3://{bucket}/{path}")
    return arr


def _optional_npy(client, bucket: str, path: str) -> np.ndarray | None:
    try:
        return load_npy(client, bucket, path)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        raw = load_bytes(client, bucket, path)
        if raw is None:
            return None
        arr = np.load(io.BytesIO(raw), allow_pickle=True)
        if arr.dtype == object:
            arr = pd.to_datetime(arr, utc=True, errors="coerce").tz_convert(None).to_numpy(dtype="datetime64[ns]")
        return arr


def load_prepared_dataset_from_minio(dataset_prefix: str) -> tuple[SplitData, SplitData, SplitData, dict]:
    """Load train/val/test arrays created by src/Gold/prepare_training_dataset.py."""
    prefix = dataset_prefix.rstrip("/")
    client = get_client()

    metadata = load_json_object(client, MINIO_GOLD_BUCKET, f"{prefix}/dataset_metadata.json")
    if metadata is None:
        raise FileNotFoundError(f"Missing dataset metadata: s3://{MINIO_GOLD_BUCKET}/{prefix}/dataset_metadata.json")

    train = SplitData(
        x_seq=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_train.npy").astype(np.float32),
        loc_ids=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_train.npy").astype(np.int64),
        y=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_train.npy").astype(np.float32),
        y_ts=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_train.npy"),
    )
    val = SplitData(
        x_seq=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_val.npy").astype(np.float32),
        loc_ids=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_val.npy").astype(np.int64),
        y=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_val.npy").astype(np.float32),
        y_ts=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_val.npy"),
    )
    test = SplitData(
        x_seq=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_test.npy").astype(np.float32),
        loc_ids=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_test.npy").astype(np.int64),
        y=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_test.npy").astype(np.float32),
        y_ts=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_test.npy"),
    )

    return train, val, test, metadata


def upload_training_outputs_to_minio(out_dir: str, run_id: str, dataset_prefix: str, logger) -> None:
    client = get_client()
    prefix = f"mamba/run_id={run_id}"
    files = {
        "best_mamba_aqi.pt": "application/octet-stream",
        "metrics_history.csv": "text/csv",
        "test_predictions.csv": "text/csv",
        "training_metadata.json": "application/json",
    }
    for filename, content_type in files.items():
        path = Path(out_dir) / filename
        if not path.exists():
            continue
        upload_bytes(
            client,
            MINIO_ARTIFACTS_BUCKET,
            f"{prefix}/{filename}",
            path.read_bytes(),
            content_type=content_type,
        )

    upload_json(
        client,
        MINIO_ARTIFACTS_BUCKET,
        f"{prefix}/artifact_manifest.json",
        {
            "run_id": run_id,
            "dataset_prefix": dataset_prefix,
            "artifact_prefix": prefix,
            "bucket": MINIO_ARTIFACTS_BUCKET,
            "uploaded_at": datetime.utcnow().isoformat(),
        },
    )
    logger.info("Training artifacts uploaded: s3://%s/%s", MINIO_ARTIFACTS_BUCKET, prefix)


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    use_amp: bool,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    amp_enabled = use_amp and device.type == "cuda"
    for x_seq, y in loader:
        x_seq, y = x_seq.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq)
        preds.append(pred.detach().float().cpu().numpy())
        targets.append(y.detach().float().cpu().numpy())

    pred_arr = denormalize(np.concatenate(preds), y_mean, y_std)
    target_arr = denormalize(np.concatenate(targets), y_mean, y_std)
    return pred_arr, target_arr


def save_prediction_results(
    out_dir: str,
    split: SplitData,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: dict,
    horizon: int,
    filename: str = "test_predictions.csv",
) -> Path:
    id_to_location = {
        int(v): str(k)
        for k, v in (metadata.get("location_to_id") or {}).items()
    }
    pred_arr = np.asarray(preds, dtype=np.float32)
    target_arr = np.asarray(targets, dtype=np.float32)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(-1, 1)

    base_times = split.y_ts
    rows: list[dict] = []
    for i in range(pred_arr.shape[0]):
        if split.loc_ids is not None:
            loc_id = int(split.loc_ids[i])
            province = id_to_location.get(loc_id, str(loc_id))
        else:
            province = ""
        base_time = None
        if base_times is not None:
            base_time = pd.Timestamp(base_times[i]).to_pydatetime()
        for step in range(min(horizon, pred_arr.shape[1], target_arr.shape[1])):
            pred_time = None
            if base_time is not None:
                pred_time = base_time + timedelta(hours=step)
            y_true = float(target_arr[i, step])
            y_pred = float(pred_arr[i, step])
            rows.append(
                {
                    "split": "test",
                    "sample_index": i,
                    "province": province,
                    "horizon_step": step + 1,
                    "time": pred_time.isoformat() if pred_time is not None else "",
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": y_pred - y_true,
                    "abs_error": abs(y_pred - y_true),
                }
            )

    out_path = Path(out_dir) / filename
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Training engine (Mamba-specific: tqdm, amp, grad_accum)
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    logger,
    epoch_idx:       int,
    total_epochs:    int,
    log_interval:    int,
    use_amp:         bool,
    grad_accum_steps: int,
    max_grad_norm:   float,
) -> tuple[float, float]:
    """Chạy 1 epoch train, trả về (train_loss, elapsed_seconds)."""
    model.train()
    running_loss = 0.0
    start_t      = time.time()
    amp_enabled  = use_amp and device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)

    for step, (x_seq, y) in enumerate(pbar, start=1):
        x_seq, y = x_seq.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq)
            loss = criterion(pred, y)
            loss_for_backward = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss tại epoch %d step %d, bỏ qua batch này.", epoch_idx, step)
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if amp_enabled:
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * y.size(0)
        avg_loss      = running_loss / (step * y.size(0))
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_loss:.5f}")

        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            logger.info(
                "Epoch %d/%d | step %d/%d | batch_loss=%.6f | running_avg=%.6f",
                epoch_idx, total_epochs, step, len(loader), loss.item(), avg_loss,
            )

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, time.time() - start_t


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp:  bool,
    y_mean:   float,
    y_std:    float,
) -> dict[str, float]:
    """Evaluate model trên một DataLoader, trả về dict metrics."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    amp_enabled = use_amp and device.type == "cuda"

    for x_seq, y in loader:
        x_seq, y = x_seq.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq)
            loss = criterion(pred, y)
        total_loss += loss.item() * y.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds_arr   = denormalize(np.concatenate(preds),   y_mean, y_std)
    targets_arr = denormalize(np.concatenate(targets), y_mean, y_std)

    preds_norm_arr = np.concatenate(preds).astype(np.float32).flatten()
    targets_norm_arr = np.concatenate(targets).astype(np.float32).flatten()

    metrics = compute_metrics(targets_arr, preds_arr)
    try:
        mse_norm = mean_squared_error(targets_norm_arr, preds_norm_arr)
        metrics["mae_norm"] = float(mean_absolute_error(targets_norm_arr, preds_norm_arr))
        metrics["rmse_norm"] = float(np.sqrt(mse_norm))
    except Exception:
        metrics["mae_norm"] = float("nan")
        metrics["rmse_norm"] = float("nan")
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mamba AQI forecasting model")
    parser.add_argument("--data-path",        type=str,   default=None, help="Optional local CSV for debugging only")
    parser.add_argument("--dataset-prefix",   type=str,   default=None, help="MinIO Gold prefix from prepare_training_dataset.py")
    parser.add_argument("--run-id",           type=str,   default=None, help="Training dataset run id, maps to training_dataset/run_id=<run-id>")
    parser.add_argument("--config",           type=str,   default=None, help="Path to project YAML config")
    parser.add_argument("--target-col",       type=str,   default="aqi")
    parser.add_argument("--window-size",      type=int,   default=24)
    parser.add_argument("--horizon",          type=int,   default=1)
    parser.add_argument("--sample-stride",    type=int,   default=4)
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--batch-size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--d-model",          type=int,   default=64)
    parser.add_argument("--n-layers",         type=int,   default=2)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--num-workers",      type=int,   default=0)
    parser.add_argument("--out-dir",          type=str,   default=None)
    parser.add_argument("--device",           type=str,   default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--location",         type=str,   default=None,   help="[Deprecated] Dùng --locations thay thế")
    parser.add_argument("--locations",        type=str,   default=None,   help="Comma-separated location_key list")
    parser.add_argument("--log-interval",     type=int,   default=50)
    parser.add_argument("--loss",             type=str,   default="huber", choices=["mse", "huber"])
    parser.add_argument("--amp",              action="store_true")
    parser.add_argument("--grad-accum-steps", type=int,   default=1)
    parser.add_argument("--max-grad-norm",    type=float, default=1.0)
    parser.add_argument("--patience",         type=int,   default=5)
    parser.add_argument("--min-delta",        type=float, default=0.0)
    parser.add_argument("--forecast-24h",     action="store_true", help="Generate next-24h forecast CSV")
    parser.add_argument("--forecast-base",    type=str,   default=None, help="CSV path for forecast base (optional)")
    parser.add_argument("--forecast-out",     type=str,   default=None)
    args = parser.parse_args()

    project_root = _REPO_ROOT
    project_cfg = load_project_config(args.config)
    data_cfg = project_cfg.get("data", {})
    dataset_cfg = project_cfg.get("dataset", {})
    model_cfg = project_cfg.get("model", {})
    training_cfg = project_cfg.get("training", {})

    args.target_col = args.target_col or data_cfg.get("target_col", "aqi")
    args.window_size = int(dataset_cfg.get("seq_len", args.window_size))
    args.horizon = int(dataset_cfg.get("pred_len", args.horizon))
    args.sample_stride = int(dataset_cfg.get("sample_stride", args.sample_stride))
    args.epochs = int(training_cfg.get("epochs", args.epochs))
    args.batch_size = int(training_cfg.get("batch_size", args.batch_size))
    args.lr = float(training_cfg.get("learning_rate", args.lr))
    args.patience = int(training_cfg.get("patience", args.patience))
    args.device = str(training_cfg.get("device", args.device))
    args.d_model = int(model_cfg.get("d_model", args.d_model))

    if args.run_id and not args.dataset_prefix:
        args.dataset_prefix = f"training_dataset/run_id={args.run_id}"

    if not args.data_path and not args.dataset_prefix:
        raise ValueError(
            "train_mamba_aqi.py khong tu lay CSV local nua. "
            "Luong chinh la chay src/Gold/prepare_training_dataset.py de tao dataset tu MinIO Gold, "
            "sau do truyen --dataset-prefix training_dataset/run_id=<run_id> hoac --run-id <run_id>. "
            "Chi truyen --data-path khi can debug local."
        )

    if args.out_dir is None:
        train_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (project_root / "runs" / train_run_id).resolve()
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
        train_run_id = out_dir.name
    args.out_dir = str(out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, name="train_mamba_aqi")
    set_seed(args.seed)

    df = None
    feature_cols = []
    if args.dataset_prefix:
        logger.info("Loading prepared training dataset: s3://%s/%s", MINIO_GOLD_BUCKET, args.dataset_prefix)
        train, val, test, dataset_meta = load_prepared_dataset_from_minio(args.dataset_prefix)
        feature_cols = dataset_meta.get("feature_cols", [])
        args.window_size = int(dataset_meta.get("seq_len", train.x_seq.shape[1]))
        args.horizon = int(dataset_meta.get("pred_len", train.y.shape[1] if train.y.ndim > 1 else 1))
        loc_count = len(dataset_meta.get("location_to_id", {}))
        max_loc_id = int(max(train.loc_ids.max(), val.loc_ids.max(), test.loc_ids.max()) + 1) if train.loc_ids is not None else 0
        num_locations = loc_count or max_loc_id
        train, val, test, y_mean, y_std = standardize_targets(train, val, test)
        logger.info("Prepared dataset shapes: %s", json.dumps(dataset_meta.get("shapes", {}), ensure_ascii=False))
        logger.info("Target normalized with train split stats: mean=%.6f | std=%.6f", y_mean, y_std)
    else:
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = (project_root / data_path).resolve()
        args.data_path = str(data_path)

        logger.info("Loading local debug dataset: %s", args.data_path)
        df = pd.read_csv(args.data_path)
        logger.info("Total rows: %d", len(df))

        selected = []
        if args.locations:
            selected = [x.strip() for x in args.locations.split(',') if x.strip()]
        elif args.location:
            selected = [args.location.strip()]

        if selected:
            if "location_key" not in df.columns:
                raise ValueError("Dataset khong co cot 'location_key'.")
            df = df[df["location_key"].astype(str).isin(selected)].copy()
            if df.empty:
                raise ValueError(f"Khong tim thay du lieu cho locations: {selected}")
            logger.info("Filtered %d locations: %s | rows=%d", len(selected), selected, len(df))

        x_seq, loc_ids, y, y_ts, num_locations, feature_cols = build_time_series_samples(
            df, args.target_col, args.window_size, args.horizon,
            sample_stride=args.sample_stride,
        )
        logger.info("Features (%d): %s", len(feature_cols), feature_cols)
        logger.info("Samples: %d | Locations: %d | Sample stride: %d", len(y), num_locations, args.sample_stride)

        train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
        train, val, test, x_mean, x_std, y_mean, y_std = standardize(train, val, test)
    logger.info(
        "Split — train: %d | val: %d | test: %d",
        len(train.y), len(val.y), len(test.y),
    )

    # DataLoaders
    device      = resolve_device(args.device)
    pin_memory  = device.type == "cuda"
    use_amp     = args.amp and device.type == "cuda"
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(AQIDataset(train), shuffle=False, **loader_kwargs)
    val_loader   = DataLoader(AQIDataset(val),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(AQIDataset(test),  shuffle=False, **loader_kwargs)

    # Model
    model = TimeSeriesMambaRegressor(
        num_features=train.x_seq.shape[-1],
        d_model=args.d_model,
        n_layers=args.n_layers,
        horizon=args.horizon,
    ).to(device)
    logger.info("Model input: pure time-series features only; location embedding disabled.")

    criterion = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info("Device: %s | AMP: %s | grad_accum: %d", device, use_amp, args.grad_accum_steps)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Training loop
    best_val_loss  = float("inf")
    best_path      = os.path.join(args.out_dir, "best_mamba_aqi.pt")
    history_path   = os.path.join(args.out_dir, "metrics_history.csv")
    epochs_without_improvement = 0

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "mae", "rmse", "mae_norm", "rmse_norm", "val_r2", "train_sec"]
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_sec = run_epoch(
            model, train_loader, criterion, optimizer, device, logger,
            epoch, args.epochs, args.log_interval,
            use_amp, args.grad_accum_steps, args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std)

        logger.info(
            "Epoch %02d/%02d | train=%.6f | val_loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | %.1fs",
            epoch, args.epochs,
            train_loss, val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["r2"],
            train_sec,
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.8f}",
                f"{val_metrics['loss']:.8f}",
                f"{val_metrics['mae']:.8f}",
                f"{val_metrics['rmse']:.8f}",
                f"{val_metrics.get('mae_norm', float('nan')):.8f}",
                f"{val_metrics.get('rmse_norm', float('nan')):.8f}",
                f"{val_metrics['r2']:.8f}",
                f"{train_sec:.2f}",
            ])

        if val_metrics["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
            logger.info("→ Checkpoint mới: %s", best_path)

        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping at epoch %d/%d | best_val_loss=%.6f",
                epoch, args.epochs, best_val_loss,
            )
            break

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp, y_mean, y_std)
    test_preds, test_targets = collect_predictions(model, test_loader, device, use_amp, y_mean, y_std)
    prediction_path = save_prediction_results(
        args.out_dir,
        test,
        test_preds,
        test_targets,
        dataset_meta if args.dataset_prefix else {},
        horizon=args.horizon,
    )

    logger.info(
        "TEST | loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | mae_norm=%.4f | rmse_norm=%.4f",
        test_metrics["loss"],
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        test_metrics.get("mae_norm", float("nan")),
        test_metrics.get("rmse_norm", float("nan")),
    )
    logger.info("Test predictions saved: %s", prediction_path)

    metadata_path = Path(args.out_dir) / "training_metadata.json"
    training_metadata = {
        "train_run_id": train_run_id,
        "dataset_prefix": args.dataset_prefix,
        "target_col": args.target_col,
        "feature_cols": feature_cols,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "num_locations": num_locations,
        "num_features": int(train.x_seq.shape[-1]),
        "use_location": False,
        "use_location_embedding": False,
        "best_val_loss": best_val_loss,
        "target_normalization": {
            "method": "standard",
            "fit_on": "train",
            "mean": y_mean,
            "std": y_std,
        },
        "test_metrics": test_metrics,
        "prediction_file": str(prediction_path),
        "model": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": str(device),
        },
        "created_at": datetime.utcnow().isoformat(),
    }
    metadata_path.write_text(json.dumps(training_metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.dataset_prefix:
        upload_training_outputs_to_minio(args.out_dir, train_run_id, args.dataset_prefix, logger)
        if args.forecast_24h and df is None:
            raise ValueError("Forecast 24h is handled by src/Inference after training from prepared MinIO arrays.")

    if args.forecast_24h:
        forecast_hours = int(args.horizon)
        logger.info("Generating %dh forecast...", forecast_hours)
        base_df = df.copy() if df is not None else None
        if args.forecast_base:
            forecast_path = Path(args.forecast_base)
            if not forecast_path.is_absolute():
                forecast_path = (project_root / forecast_path).resolve()
            base_df = pd.read_csv(forecast_path)
        if base_df is None:
            raise ValueError("Forecast base data is required.")

        if "location_key" not in base_df.columns:
            raise ValueError("Dataset forecast cần có cột 'location_key'.")

        col_map = {c.lower(): c for c in base_df.columns}
        base_ts_col = col_map.get("ts_utc") or col_map.get("time") or col_map.get("timestamp")
        if base_ts_col is None:
            raise ValueError("Cần có cột timestamp ('ts_utc', 'Time', hoặc 'timestamp') để dự báo tiếp theo.")
        if base_ts_col != "ts_utc":
            base_df["ts_utc"] = base_df[base_ts_col]

        cleaned = df.copy()
        clean_col_map = {c.lower(): c for c in cleaned.columns}
        clean_ts_col = clean_col_map.get("ts_utc") or clean_col_map.get("time") or clean_col_map.get("timestamp")
        if clean_ts_col is None:
            raise ValueError("Dataset train không có cột timestamp hợp lệ để forecast.")
        cleaned["_ts"] = pd.to_datetime(cleaned[clean_ts_col], utc=True, errors="coerce")
        cleaned = cleaned.dropna(subset=["_ts", "location_key", args.target_col]).copy()
        cleaned["_loc_id"] = cleaned["location_key"].astype("category").cat.codes.astype(np.int64)
        loc_to_id = (
            cleaned.assign(_loc_key_str=cleaned["location_key"].astype(str))
            .drop_duplicates(subset=["_loc_key_str"])
            .set_index("_loc_key_str")["_loc_id"]
            .to_dict()
        )

        base_df = base_df.loc[
            base_df["location_key"].astype(str).isin(list(loc_to_id.keys()))
        ].copy()
        if base_df.empty:
            raise ValueError("Forecast base không có location trùng với dữ liệu train đã chọn.")

        for col in feature_cols:
            if col not in base_df.columns:
                base_df[col] = np.nan
            base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
            fill_val = base_df[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            base_df[col] = base_df[col].fillna(fill_val)

        future_df = build_future_24h_frame(
            base_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            hours=forecast_hours,
        )
        for col in feature_cols:
            future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
            fill_val = base_df[col].median() if col in base_df.columns else 0.0
            if pd.isna(fill_val):
                fill_val = 0.0
            future_df[col] = future_df[col].fillna(fill_val)

        preds_rows: list[dict] = []
        model.eval()
        infer_x, infer_meta = [], []
        x_mean_2d = x_mean.squeeze(0)
        x_std_2d = x_std.squeeze(0)

        with torch.inference_mode():
            for loc in sorted(future_df["location_key"].astype(str).unique().tolist()):
                if loc not in loc_to_id:
                    continue
                loc_hist = (
                    base_df.loc[base_df["location_key"].astype(str) == loc]
                    .copy()
                    .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                    .dropna(subset=["ts_utc"])
                    .sort_values("ts_utc")
                )
                if len(loc_hist) < args.window_size:
                    continue

                rolling_window = loc_hist[feature_cols].tail(args.window_size).to_numpy(dtype=np.float32)
                loc_future = (
                    future_df.loc[future_df["location_key"].astype(str) == loc]
                    .copy()
                    .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                    .sort_values("ts_utc")
                )

                for _, row in loc_future.iterrows():
                    x_norm = (rolling_window - x_mean_2d) / x_std_2d
                    infer_x.append(x_norm.astype(np.float32, copy=False))
                    infer_meta.append((row["ts_utc"], loc))
                    next_feats = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                    rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

            if infer_x:
                x_all = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred_norm_all = model(x_all).detach().float().cpu().numpy()

                pred_all = pred_norm_all * y_std + y_mean
                for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                    pred_scalar = float(np.asarray(pred_val, dtype=np.float32).reshape(-1)[0])
                    preds_rows.append({"time": ts_val, "location": loc_val, "predicted": pred_scalar})

        if not preds_rows:
            raise RuntimeError("Không tạo được dự báo cho Mamba.")

        future_out = (
            pd.DataFrame(preds_rows)
            .assign(time=lambda d: format_time_utc_strings(d["time"]))
            [["time", "location", "predicted"]]
            .sort_values(["location", "time"])
            .reset_index(drop=True)
        )

        forecast_name = args.forecast_out or f"future_{forecast_hours}h_predictions.csv"
        forecast_out = Path(args.out_dir) / forecast_name
        future_out.to_csv(forecast_out, index=False)
        logger.info("Forecast saved: %s", forecast_out)
    logger.info("Best model: %s", best_path)
    logger.info("History   : %s", history_path)


if __name__ == "__main__":
    main()
