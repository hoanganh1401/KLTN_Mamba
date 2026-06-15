"""Shared training helpers for Mamba, LSTM, and Transformer AQI models."""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
_MODEL_ROOT = _SRC_ROOT / "Model"
_MAMBA_ROOT = _MODEL_ROOT / "Mamba"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT), str(_MODEL_ROOT), str(_MAMBA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cfg_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _mamba_fastpath_ready() -> bool:
    try:
        import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    except Exception:
        return False
    return (
        selective_scan_interface.causal_conv1d_fwd_function is not None
        and selective_scan_interface.causal_conv1d_bwd_function is not None
        and selective_scan_interface.selective_scan_cuda is not None
    )


def _mamba_cuda_path_ready() -> bool:
    try:
        import mamba_ssm.modules.mamba_simple as mamba_simple
        import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    except Exception:
        return False
    return (
        mamba_simple.causal_conv1d_fn is not None
        and selective_scan_interface.selective_scan_cuda is not None
    )

from src.core.data_structs import SplitData
from src.core.metrics import compute_metrics, denormalize
from src.common.config import MINIO_ARTIFACTS_BUCKET, MINIO_GOLD_BUCKET
from src.common.minio_io import get_client, load_bytes, load_json_object, load_npy, upload_bytes, upload_json
from src.common.time_utils import now_local, parse_time_local


# ---------------------------------------------------------------------------
# Xây dựng samples time-series
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


def prepare_training_targets(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    target_mode: str,
) -> tuple[SplitData, SplitData, SplitData, float, float, str]:
    mode = str(target_mode or "absolute").lower()
    if mode not in {"absolute", "residual"}:
        mode = "absolute"
    if mode == "residual":
        if train.y_base is None or val.y_base is None or test.y_base is None:
            mode = "absolute"
        else:
            for split in (train, val, test):
                split.y = (split.y - split.y_base[:, None]).astype(np.float32)
    train, val, test, y_mean, y_std = standardize_targets(train, val, test)
    return train, val, test, y_mean, y_std, mode


def restore_target_scale(
    arr: np.ndarray,
    y_mean: float,
    y_std: float,
    target_mode: str = "absolute",
    y_base: np.ndarray | None = None,
) -> np.ndarray:
    out = denormalize(arr, y_mean, y_std)
    if target_mode == "residual" and y_base is not None:
        out = out + np.asarray(y_base, dtype=np.float32).reshape(-1, 1)
    return out


def _find_target_feature_index(metadata: dict | None) -> int | None:
    if not metadata:
        return None
    feature_cols = metadata.get("feature_cols")
    target_col = metadata.get("target_col", "aqi")
    if isinstance(feature_cols, list) and target_col in feature_cols:
        return int(feature_cols.index(target_col))
    return None


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_f = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    denom = float(np.sum((y_true_f - y_true_f.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true_f - y_pred_f) ** 2) / denom)


def persistence_baseline_metrics(
    split: SplitData,
    metadata: dict | None,
    scaler,
    y_mean: float,
    y_std: float,
    target_mode: str = "absolute",
) -> dict[str, float] | None:
    target_idx = _find_target_feature_index(metadata)
    if target_idx is None:
        return None

    feature_cols = metadata.get("feature_cols", []) if metadata else []
    scale_cols = metadata.get("scale_cols") or metadata.get("metric_cols") or []
    if not feature_cols or not scale_cols:
        return None
    target_col = metadata.get("target_col", "aqi")
    if target_col not in scale_cols:
        return None

    try:
        scaler_target_pos = list(scale_cols).index(target_col)
    except ValueError:
        return None

    last_scaled = split.x_seq[:, -1, target_idx].reshape(-1, 1)
    filler = np.zeros((last_scaled.shape[0], len(scale_cols)), dtype=np.float32)
    filler[:, scaler_target_pos] = last_scaled[:, 0]
    try:
        last_aqi = scaler.inverse_transform(filler)[:, scaler_target_pos].astype(np.float32)
    except Exception:
        return None
    pred = np.repeat(last_aqi[:, None], split.y.shape[1], axis=1)
    target = restore_target_scale(split.y, y_mean, y_std, target_mode, split.y_base)

    return {
        "mae": float(mean_absolute_error(target.reshape(-1), pred.reshape(-1))),
        "rmse": float(np.sqrt(mean_squared_error(target.reshape(-1), pred.reshape(-1)))),
        "r2": _safe_r2(target, pred),
    }


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
            arr = parse_time_local(arr).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
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
        y_base=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_base_train.npy"),
    )
    val = SplitData(
        x_seq=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_val.npy").astype(np.float32),
        loc_ids=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_val.npy").astype(np.int64),
        y=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_val.npy").astype(np.float32),
        y_ts=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_val.npy"),
        y_base=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_base_val.npy"),
    )
    test = SplitData(
        x_seq=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/X_test.npy").astype(np.float32),
        loc_ids=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/loc_ids_test.npy").astype(np.int64),
        y=_require_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_test.npy").astype(np.float32),
        y_ts=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_ts_test.npy"),
        y_base=_optional_npy(client, MINIO_GOLD_BUCKET, f"{prefix}/y_base_test.npy"),
    )

    return train, val, test, metadata


def _resolve_single_province(metadata: dict | None) -> str | None:
    if not metadata:
        return None
    locations = metadata.get("locations")
    if isinstance(locations, list) and len(locations) == 1:
        return str(locations[0])
    location_to_id = metadata.get("location_to_id")
    if isinstance(location_to_id, dict) and len(location_to_id) == 1:
        return str(next(iter(location_to_id.keys())))
    return None


def upload_training_outputs_to_minio(
    out_dir: str,
    run_id: str,
    dataset_prefix: str,
    logger,
    dataset_metadata: dict | None = None,
    model_name: str = "mamba",
    best_model_filename: str = "best_mamba_aqi.pt",
) -> None:
    client = get_client()
    province = _resolve_single_province(dataset_metadata)
    if province:
        prefix = f"{model_name}/province={province}/run_id={run_id}"
    else:
        prefix = f"{model_name}/run_id={run_id}"
    files = {
        best_model_filename: "application/octet-stream",
        "metrics.json": "application/json",
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

    best_model_path = Path(out_dir) / best_model_filename
    if best_model_path.exists():
        upload_bytes(
            client,
            MINIO_ARTIFACTS_BUCKET,
            f"{prefix}/best_model.pt",
            best_model_path.read_bytes(),
            content_type="application/octet-stream",
        )

    upload_json(
        client,
        MINIO_ARTIFACTS_BUCKET,
        f"{prefix}/artifact_manifest.json",
        {
            "run_id": run_id,
            "province": province,
            "scope": "single_province" if province else "multi_province",
            "dataset_prefix": dataset_prefix,
            "artifact_prefix": prefix,
            "bucket": MINIO_ARTIFACTS_BUCKET,
            "uploaded_at": now_local().isoformat(),
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
    target_mode: str = "absolute",
    y_base: np.ndarray | None = None,
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

    pred_arr = restore_target_scale(np.concatenate(preds), y_mean, y_std, target_mode, y_base)
    target_arr = restore_target_scale(np.concatenate(targets), y_mean, y_std, target_mode, y_base)
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
    running_loss = torch.zeros((), device=device)
    seen_samples = 0
    start_t      = time.time()
    amp_enabled  = use_amp and device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False, mininterval=2.0)

    for step, (x_seq, y) in enumerate(pbar, start=1):
        x_seq = x_seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq)
            loss = criterion(pred, y)
            loss_for_backward = loss / grad_accum_steps

        if not torch.isfinite(loss.detach()):
            logger.warning("Non-finite loss tại epoch %d step %d, bỏ qua batch này.", epoch_idx, step)
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if amp_enabled:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = y.size(0)
        running_loss += loss.detach() * batch_size
        seen_samples += batch_size

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        if should_log:
            loss_value = float(loss.detach().cpu())
            avg_loss = float((running_loss / max(1, seen_samples)).detach().cpu())
            pbar.set_postfix(loss=f"{loss_value:.5f}", avg=f"{avg_loss:.5f}")
            logger.info(
                "Epoch %d/%d | step %d/%d | batch_loss=%.6f | running_avg=%.6f",
                epoch_idx, total_epochs, step, len(loader), loss_value, avg_loss,
            )

    epoch_loss = float((running_loss / len(loader.dataset)).detach().cpu())
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
    target_mode: str = "absolute",
    y_base: np.ndarray | None = None,
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

    preds_arr = restore_target_scale(np.concatenate(preds), y_mean, y_std, target_mode, y_base)
    targets_arr = restore_target_scale(np.concatenate(targets), y_mean, y_std, target_mode, y_base)

    preds_norm_arr = np.concatenate(preds).astype(np.float32).flatten()
    targets_norm_arr = np.concatenate(targets).astype(np.float32).flatten()

    metrics = compute_metrics(targets_arr, preds_arr)
    finite_norm_mask = np.isfinite(targets_norm_arr) & np.isfinite(preds_norm_arr)
    metrics["nonfinite_norm_count"] = int(len(targets_norm_arr) - finite_norm_mask.sum())
    try:
        valid_targets_norm = targets_norm_arr[finite_norm_mask]
        valid_preds_norm = preds_norm_arr[finite_norm_mask]
        if len(valid_targets_norm) == 0:
            raise ValueError("No finite normalized prediction pairs.")
        mse_norm = mean_squared_error(valid_targets_norm, valid_preds_norm)
        metrics["mae_norm"] = float(mean_absolute_error(valid_targets_norm, valid_preds_norm))
        metrics["mse_norm"] = float(mse_norm)
        metrics["rmse_norm"] = float(np.sqrt(mse_norm))
        metrics["r2_norm"] = _safe_r2(valid_targets_norm, valid_preds_norm)
    except Exception:
        metrics["mae_norm"] = float("nan")
        metrics["mse_norm"] = float("nan")
        metrics["rmse_norm"] = float("nan")
        metrics["r2_norm"] = float("nan")
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main