"""
train_mamba_aqi.py
------------------
Train Mamba for AQI forecasting on a CSV dataset.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core.data_structs import AQIDataset, SplitData
from core.metrics import compute_metrics, denormalize
from core.utils import resolve_device, set_seed, setup_logger
from common.config import load_project_config
from mamba.mamba_model import TimeSeriesMambaRegressor


def _detect_time_col(df: pd.DataFrame) -> str:
    col_map = {c.lower(): c for c in df.columns}
    if "ts_utc" in col_map:
        return col_map["ts_utc"]
    if "time" in col_map:
        return col_map["time"]
    if "timestamp" in col_map:
        return col_map["timestamp"]
    raise ValueError("Timestamp column not found (ts_utc/time/timestamp).")


def _detect_location_col(df: pd.DataFrame) -> str:
    if "location_key" in df.columns:
        return "location_key"
    if "location" in df.columns:
        return "location"
    raise ValueError("Location column not found (location_key/location).")


def build_time_series_samples(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    horizon: int,
    feature_cols: list[str] | None = None,
    include_target_history: bool = True,
    sample_stride: int = 1,
):
    """Build sliding-window samples from a time-series DataFrame."""
    ts_col = _detect_time_col(df)
    loc_col = _detect_location_col(df)

    for col, label in [(target_col, "target"), (ts_col, "timestamp"), (loc_col, "location")]:
        if col not in df.columns:
            raise ValueError(f"Missing {label} column: {col}")

    if window_size < 1 or horizon < 1 or sample_stride < 1:
        raise ValueError("window_size, horizon, and sample_stride must be >= 1")

    work = df.copy()
    work["_ts"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    missing_required = work[["_ts", loc_col, target_col]].isna().any(axis=1)
    if missing_required.any():
        raise ValueError("Data contains NaN in timestamp/location/target. Clean before training.")

    work["_loc_id"] = work[loc_col].astype("category").cat.codes.astype(np.int64)
    num_locations = int(work["_loc_id"].max()) + 1

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
        raise ValueError("No numeric feature columns found.")

    cols_to_fill = list(dict.fromkeys(numeric_cols + [target_col]))
    for col in cols_to_fill:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    all_nan_cols = [c for c in cols_to_fill if work[c].isna().all()]
    if target_col in all_nan_cols:
        raise ValueError("Target column is all NaN.")
    if all_nan_cols:
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]
        cols_to_fill = [c for c in cols_to_fill if c not in all_nan_cols]

    missing_mask = work[cols_to_fill].isna().any(axis=1)
    work[cols_to_fill] = work.groupby("_loc_id", sort=False)[cols_to_fill].ffill()
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
        raise ValueError("No samples were generated. Check window_size/horizon or data size.")

    return (
        np.concatenate(x_seq_list, axis=0).astype(np.float32),
        np.concatenate(loc_id_list, axis=0).astype(np.int64),
        np.concatenate(y_list, axis=0).astype(np.float32),
        np.concatenate(y_ts_list, axis=0).astype("datetime64[ns]"),
        num_locations,
        numeric_cols,
    )


def split_data_by_timeline(
    x_seq: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[SplitData, SplitData, SplitData]:
    """Split dataset by timeline."""
    if len(y) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split sizes.")

    def _take(idx):
        return SplitData(x_seq=x_seq[idx], loc_ids=loc_ids[idx], y=y[idx])

    return (
        _take(order[:train_end]),
        _take(order[train_end:val_end]),
        _take(order[val_end:]),
    )


def standardize(
    train: SplitData,
    val: SplitData,
    test: SplitData,
) -> tuple[SplitData, SplitData, SplitData, np.ndarray, np.ndarray, float, float]:
    """Standardize X and y using train statistics."""
    x_mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    for split in [train, val, test]:
        split.x_seq = (split.x_seq - x_mean) / x_std

    y_mean = float(train.y.mean())
    y_std = float(train.y.std())
    if y_std < 1e-6:
        y_std = 1.0

    for split in [train, val, test]:
        split.y = (split.y - y_mean) / y_std

    return train, val, test, x_mean, x_std, y_mean, y_std


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    logger,
    epoch_idx: int,
    total_epochs: int,
    log_interval: int,
    use_amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> tuple[float, float]:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    start_t = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)

    for step, (x_seq, loc_ids, y) in enumerate(pbar, start=1):
        x_seq, loc_ids, y = x_seq.to(device), loc_ids.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)
            loss_for_backward = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss at epoch %d step %d", epoch_idx, step)
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
        avg_loss = running_loss / (step * y.size(0))
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_loss:.5f}")

        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            logger.info(
                "Epoch %d/%d | step %d/%d | batch_loss=%.6f | running_avg=%.6f",
                epoch_idx,
                total_epochs,
                step,
                len(loader),
                loss.item(),
                avg_loss,
            )

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, time.time() - start_t


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    amp_enabled = use_amp and device.type == "cuda"

    for x_seq, loc_ids, y in loader:
        x_seq, loc_ids, y = x_seq.to(device), loc_ids.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)
        total_loss += loss.item() * y.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds_arr = denormalize(np.concatenate(preds), y_mean, y_std)
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


def main() -> None:
    cfg = load_project_config()
    data_cfg = cfg.get("data", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    parser = argparse.ArgumentParser(description="Train Mamba AQI forecasting model")
    parser.add_argument("--data-path", type=str, default="dataset/2025.csv")
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "auto"])
    parser.add_argument("--location", type=str, default=None, help="Deprecated; use --locations")
    parser.add_argument("--locations", type=str, default=None, help="Comma-separated locations")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    args.target_col = args.target_col or data_cfg.get("target_col", "aqi")
    args.window_size = int(args.window_size or dataset_cfg.get("seq_len", 96))
    args.horizon = int(args.horizon or dataset_cfg.get("pred_len", 12))
    args.epochs = int(args.epochs or training_cfg.get("epochs", 10))
    args.batch_size = int(args.batch_size or training_cfg.get("batch_size", 128))
    args.lr = float(args.lr or training_cfg.get("learning_rate", 1e-3))
    args.d_model = int(args.d_model or model_cfg.get("d_model", 64))
    args.n_layers = int(args.n_layers or model_cfg.get("n_layers", 2))
    args.patience = int(args.patience or training_cfg.get("patience", 5))
    args.device = args.device or training_cfg.get("device", "auto")

    project_root = PROJECT_ROOT
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    args.data_path = str(data_path)

    if args.out_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (project_root / "runs" / run_id).resolve()
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    args.out_dir = str(out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, name="train_mamba_aqi")
    set_seed(args.seed)

    logger.info("Loading dataset: %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Total rows: %d", len(df))

    selected = []
    if args.locations:
        selected = [x.strip() for x in args.locations.split(",") if x.strip()]
    elif args.location:
        selected = [args.location.strip()]

    if selected:
        loc_col = _detect_location_col(df)
        df = df[df[loc_col].astype(str).isin(selected)].copy()
        if df.empty:
            raise ValueError(f"No data found for locations: {selected}")
        logger.info("Filtered %d locations: %s | rows=%d", len(selected), selected, len(df))

    x_seq, loc_ids, y, y_ts, num_locations, feature_cols = build_time_series_samples(
        df,
        args.target_col,
        args.window_size,
        args.horizon,
        sample_stride=args.sample_stride,
    )
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Samples: %d | Locations: %d | Sample stride: %d", len(y), num_locations, args.sample_stride)

    train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
    train, val, test, x_mean, x_std, y_mean, y_std = standardize(train, val, test)
    logger.info("Split — train: %d | val: %d | test: %d", len(train.y), len(val.y), len(test.y))

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    use_amp = args.amp and device.type == "cuda"
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(AQIDataset(train), shuffle=False, **loader_kwargs)
    val_loader = DataLoader(AQIDataset(val), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(AQIDataset(test), shuffle=False, **loader_kwargs)

    model = TimeSeriesMambaRegressor(
        num_features=train.x_seq.shape[-1],
        num_locations=num_locations,
        d_model=args.d_model,
        n_layers=args.n_layers,
        horizon=args.horizon,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info("Device: %s | AMP: %s | grad_accum: %d", device, use_amp, args.grad_accum_steps)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "best_mamba_aqi.pt")
    history_path = os.path.join(args.out_dir, "metrics_history.csv")
    epochs_without_improvement = 0

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "mae", "rmse", "val_r2", "train_sec"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_sec = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            logger,
            epoch,
            args.epochs,
            args.log_interval,
            use_amp,
            args.grad_accum_steps,
            args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std)

        logger.info(
            "Epoch %02d/%02d | train=%.6f | val_loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | %.1fs",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics.get("mae_norm", float("nan")),
            val_metrics.get("rmse_norm", float("nan")),
            val_metrics["r2"],
            train_sec,
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{val_metrics['loss']:.8f}",
                    f"{val_metrics.get('mae_norm', float('nan')):.8f}",
                    f"{val_metrics.get('rmse_norm', float('nan')):.8f}",
                    f"{val_metrics['r2']:.8f}",
                    f"{train_sec:.2f}",
                ]
            )

        if val_metrics["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
            logger.info("New checkpoint: %s", best_path)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping at epoch %d/%d | best_val_loss=%.6f",
                epoch,
                args.epochs,
                best_val_loss,
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp, y_mean, y_std)

    logger.info(
        "TEST | loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | mae_norm=%.4f | rmse_norm=%.4f",
        test_metrics["loss"],
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        test_metrics.get("mae_norm", float("nan")),
        test_metrics.get("rmse_norm", float("nan")),
    )

    logger.info("Best model: %s", best_path)
    logger.info("History   : %s", history_path)


if __name__ == "__main__":
    main()
