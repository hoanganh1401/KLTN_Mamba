"""
pipeline_mamba.py
-----------------
Mamba model (TabularMambaRegressor), training loop, evaluate, vÃ  train_pipeline.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from Utils import (
    build_future_24h_frame,
    format_time_utc_strings,
    load_train_module,
    normalize_locations,
    parse_time_local,
    split_data_by_timeline,
)


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, y_mean: float, y_std: float) -> dict:
    """Evaluate Mamba model trÃªn má»™t DataLoader."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    preds_norm, targets_norm = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * yb.size(0)
        preds_norm.append(out.detach().cpu().numpy())
        targets_norm.append(yb.detach().cpu().numpy())
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0) * y_std + y_mean
    targets_arr = np.concatenate(targets, axis=0) * y_std + y_mean

    preds_norm_arr = np.concatenate(preds_norm, axis=0)
    targets_norm_arr = np.concatenate(targets_norm, axis=0)

    mse_norm = mean_squared_error(targets_norm_arr, preds_norm_arr)
    mae_norm = mean_absolute_error(targets_norm_arr, preds_norm_arr)
    rmse_norm = float(np.sqrt(mse_norm))

    mse = mean_squared_error(targets_arr, preds_arr)
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": float(mean_absolute_error(targets_arr, preds_arr)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(targets_arr, preds_arr)),
        "mae_norm": float(mae_norm),
        "rmse_norm": float(rmse_norm),
        "preds": preds_arr,
        "targets": targets_arr,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    df: pd.DataFrame,
    forecast_base_df: pd.DataFrame | None,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
    sample_stride: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_layers: int,
    loss_name: str,
    seed: int,
    num_workers: int,
    use_gpu: bool,
    log_interval: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    lr_scheduler_patience: int = 2,
    lr_scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    run_dir: str | None = None,
    forecast_file_name: str = "future_24h_predictions.csv",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train Mamba sequence model vÃ  tráº£ vá» (summary, history_df, future_df)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Cáº§n chá»n Ã­t nháº¥t 1 location Ä‘á»ƒ train Mamba.")

    work_df = df.copy()
    if "location_key" not in work_df.columns and "location" in work_df.columns:
        work_df["location_key"] = work_df["location"].astype(str)
    col_map = {c.lower(): c for c in work_df.columns}
    if "location_key" not in work_df.columns or not any(c in col_map for c in ["time", "ts_utc", "timestamp"]):
        raise ValueError("Dataset train Mamba can co cot 'location'/'location_key' va cot 'time'.")

    work_df = work_df.loc[
        work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])
    ].copy()
    if work_df.empty:
        raise ValueError("KhÃ´ng cÃ³ dá»¯ liá»‡u train cho cÃ¡c location Ä‘Ã£ chá»n.")

    # --- Load helper tá»« scripts/train_mamba_aqi.py ---
    mod = load_train_module()
    if mod is None or not hasattr(mod, "build_time_series_samples"):
        raise RuntimeError(
            "KhÃ´ng load Ä‘Æ°á»£c module mamba/scripts/train_mamba_aqi.py Ä‘á»ƒ cháº¡y Mamba sequence."
        )

    x_seq, loc_ids, y, y_ts, num_locations, ts_feature_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        sample_stride=sample_stride,
        feature_cols=feature_cols,
        include_target_history=True,
    )

    train_split, val_split, test_split = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    # --- Normalize ---
    x_mean = train_split.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train_split.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    for s in [train_split, val_split, test_split]:
        s.x_seq = (s.x_seq - x_mean) / x_std

    y_mean = float(train_split.y.mean())
    y_std = float(train_split.y.std())
    if y_std < 1e-6:
        y_std = 1.0
    for s in [train_split, val_split, test_split]:
        s.y = (s.y - y_mean) / y_std

    train_ds = mod.AQIDataset(train_split)
    val_ds = mod.AQIDataset(val_split)
    test_ds = mod.AQIDataset(test_split)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)

    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    raw_model = mod.TimeSeriesMambaRegressor(
        num_features=train_split.x_seq.shape[-1],
        d_model=d_model,
        n_layers=n_layers,
        horizon=horizon,
    ).to(device)
    model = raw_model

    compile_enabled = False
    try:
        import triton  # noqa: F401
        triton_available = True
    except Exception:
        triton_available = False

    if device.type == "cuda" and triton_available and hasattr(torch, "compile"):
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            compile_enabled = True
        except Exception:
            compile_enabled = False

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=early_stop_min_delta,
        threshold_mode="abs",
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(xb)
                loss = criterion(out, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_for_backward = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * yb.size(0)
            global_step += 1

            if total_steps > 0 and (step % 20 == 0 or step == len(train_loader)):
                prog.progress(min(global_step / total_steps, 1.0))

            if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                avg_loss = running_loss / max(step * yb.size(0), 1)
                line = (
                    f"Epoch {epoch}/{epochs} | step {step}/{len(train_loader)} | "
                    f"batch_loss={loss.item():.6f} | running_avg={avg_loss:.6f}"
                )
                log_lines.append(line)
                log_box.code("\n".join(log_lines[-20:]))

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_sec = time.time() - epoch_start

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | mae={val_metrics.get('mae_norm', float('nan')):.4f} | "
            f"rmse={val_metrics.get('rmse_norm', float('nan')):.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"lr={current_lr:.2e} | sec={epoch_sec:.1f}"
        )
        log_lines.append(epoch_line)
        if current_lr < prev_lr:
            log_lines.append(f"ReduceLROnPlateau: val_loss khong cai thien, lr {prev_lr:.2e} -> {current_lr:.2e}")
        log_box.code("\n".join(log_lines[-20:]))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "learning_rate": current_lr,
                "mae": val_metrics.get("mae_norm"),
                "rmse": val_metrics.get("rmse_norm"),
                "val_r2": val_metrics["r2"],
                "train_sec": epoch_sec,
            }
        )

        if val_metrics["loss"] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            stop_line = (
                f"Early stopping at epoch {epoch}/{epochs} | "
                f"best_val_loss={best_val_loss:.6f}"
            )
            log_lines.append(stop_line)
            log_box.code("\n".join(log_lines[-20:]))
            break

    if best_state is None:
        raise RuntimeError("KhÃ´ng cÃ³ checkpoint há»£p lá»‡ trong quÃ¡ trÃ¬nh train.")

    raw_model.load_state_dict(best_state)
    raw_model.to(device)
    eval_start = time.time()
    val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
    test_metrics = mod.evaluate(model, test_loader, criterion, device, amp_enabled, y_mean, y_std)
    # --- Build province mapping for output/inference metadata only ---
    cleaned = work_df.copy()
    clean_col_map = {c.lower(): c for c in cleaned.columns}
    ts_col = clean_col_map.get("time") or clean_col_map.get("timestamp") or clean_col_map.get("ts_utc")
    eval_sec = time.time() - eval_start
    cleaned["_ts"] = parse_time_local(cleaned[ts_col])
    cleaned = cleaned.dropna(subset=["_ts", "location_key", target_col]).copy()
    cleaned["_loc_id"] = cleaned["location_key"].astype("category").cat.codes.astype(np.int64)
    loc_to_id = (
        cleaned.assign(_loc_key_str=cleaned["location_key"].astype(str))
        .drop_duplicates(subset=["_loc_key_str"])
        .set_index("_loc_key_str")["_loc_id"]
        .to_dict()
    )

    # --- Forecast 24h ---
    base_df = forecast_base_df if forecast_base_df is not None else cleaned.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("KhÃ´ng cÃ³ dá»¯ liá»‡u test lÃ m má»‘c Ä‘á»ƒ dá»± bÃ¡o 24h tiáº¿p theo.")

    base_df = base_df.loc[
        base_df["location_key"].astype(str).isin(list(loc_to_id.keys()))
    ].copy()
    if base_df.empty:
        raise ValueError("Test CSV khÃ´ng cÃ³ location trÃ¹ng vá»›i dá»¯ liá»‡u train Ä‘Ã£ chá»n.")

    base_col_map = {c.lower(): c for c in base_df.columns}
    base_ts_col = base_col_map.get("time") or base_col_map.get("timestamp") or base_col_map.get("ts_utc") or "_ts"
    if base_ts_col not in base_df.columns:
        raise ValueError("Cáº§n cÃ³ cá»™t timestamp ('ts_utc', 'Time', hoáº·c 'timestamp') Ä‘á»ƒ dá»± bÃ¡o 24h tiáº¿p theo.")
    if base_ts_col != "time":
        base_df["time"] = base_df[base_ts_col]

    for col in ts_feature_cols:
        if col not in base_df.columns:
            base_df[col] = np.nan
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        fill_val = base_df[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        base_df[col] = base_df[col].fillna(fill_val)

    future_df = build_future_24h_frame(base_df, feature_cols=ts_feature_cols, target_col=target_col)
    for col in ts_feature_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        fill_val = base_df[col].median() if col in base_df.columns else 0.0
        if pd.isna(fill_val):
            fill_val = 0.0
        future_df[col] = future_df[col].fillna(fill_val)

    # --- Batched inference ---
    forecast_start = time.time()
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
                .assign(time=lambda d: parse_time_local(d["time"]))
                .dropna(subset=["time"])
                .sort_values("time")
            )
            if len(loc_hist) < window_size:
                continue

            rolling_window = loc_hist[ts_feature_cols].tail(window_size).to_numpy(dtype=np.float32)
            loc_future = (
                future_df.loc[future_df["location_key"].astype(str) == loc]
                .copy()
                .assign(time=lambda d: parse_time_local(d["time"]))
                .sort_values("time")
            )

            for _, row in loc_future.iterrows():
                x_norm = (rolling_window - x_mean_2d) / x_std_2d
                infer_x.append(x_norm.astype(np.float32, copy=False))
                infer_meta.append((row["time"], loc))
                next_feats = row[ts_feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

        if infer_x:
            x_all = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred_norm_all = model(x_all).detach().float().cpu().numpy()

            pred_all = pred_norm_all * y_std + y_mean
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                pred_scalar = float(np.asarray(pred_val, dtype=np.float32).reshape(-1)[0])
                preds_rows.append({"time": ts_val, "location": loc_val, "predicted": pred_scalar})

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("KhÃ´ng táº¡o Ä‘Æ°á»£c dá»± bÃ¡o 24h cho Mamba sequence.")

    future_out = (
        pd.DataFrame(preds_rows)
        .assign(time=lambda d: format_time_utc_strings(d["time"]))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    # --- LÆ°u artifacts ---
    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_runs", run_id)
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "best_mamba.pt")
    metrics_path = os.path.join(out_dir, "metrics_history.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)

    io_start = time.time()
    torch.save(raw_model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    future_out.to_csv(future_pred_path, index=False)
    io_sec = time.time() - io_start

    summary = {
        "device": str(device),
        "torch_compile": bool(compile_enabled),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
        "grad_accum_steps": int(grad_accum_steps),
        "n_rows_used": len(y),
        "split_train": len(train_split.y),
        "split_val": len(val_split.y),
        "split_test": len(test_split.y),
        "feature_count_after_encode": train_split.x_seq.shape[-1],
        "sample_stride": int(sample_stride),
        "encoded_features": ts_feature_cols,
        "val_loss": val_metrics["loss"],
        "val_r2": val_metrics["r2"],
        "val_mae_norm": val_metrics.get("mae_norm"),
        "val_rmse_norm": val_metrics.get("rmse_norm"),
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae_norm": test_metrics.get("mae_norm"),
        "test_rmse_norm": test_metrics.get("rmse_norm"),
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "per_location_files": [],
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "best_val_loss": float(best_val_loss),
        "initial_learning_rate": float(lr),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "val_loss",
            "patience": int(lr_scheduler_patience),
            "factor": float(lr_scheduler_factor),
            "min_lr": float(min_lr),
        },
        "train_only_sec": float(pd.DataFrame(history)["train_sec"].sum()) if history else 0.0,
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "io_sec": float(io_sec),
        "run_sec": time.time() - start_all,
    }

    return summary, pd.DataFrame(history), future_out
