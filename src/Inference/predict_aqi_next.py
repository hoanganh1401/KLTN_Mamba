"""Prediction helpers for AQI Mamba inference.

Gold prepares feature windows; this module runs the loaded Mamba model and
builds future timestamps for Streamlit or batch inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.common.time_utils import format_time_local_strings, parse_time_local


def format_time_utc_strings(values: pd.Series) -> pd.Series:
    """Backward-compatible alias: format datetime-like values as Vietnam local time."""
    return format_time_local_strings(values)


def _resolve_time_col(df: pd.DataFrame) -> str:
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get("time") or col_map.get("timestamp") or col_map.get("ts_utc")
    if ts_col is None:
        raise ValueError("Need timestamp column: 'ts_utc', 'time', or 'timestamp'.")
    return ts_col


def build_future_24h_frame(
    df_valid: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    hours: int = 24,
) -> pd.DataFrame:
    """Build future feature rows using each location's latest daily pattern."""
    if df_valid.empty:
        raise ValueError("Cannot build future frame from an empty DataFrame.")

    work = df_valid.copy()
    ts_col = _resolve_time_col(work)
    if ts_col != "time":
        work["time"] = work[ts_col]
    work["time"] = parse_time_local(work["time"])
    work = work.dropna(subset=["time"]).copy()
    if work.empty:
        raise ValueError("No valid timestamps found for future frame.")

    if "location_key" in work.columns:
        groups = [
            (loc, work.loc[work["location_key"].astype(str) == loc].sort_values("time").copy())
            for loc in sorted(work["location_key"].dropna().astype(str).unique().tolist())
        ]
    else:
        groups = [(None, work.sort_values("time").copy())]

    future_rows: list[dict[str, Any]] = []
    for loc, group in groups:
        if group.empty:
            continue

        last_ts = group["time"].iloc[-1]
        start_ts = last_ts + pd.Timedelta(hours=1)
        template = group.tail(hours).copy()
        if template.empty:
            continue
        if len(template) < hours:
            repeat_count = hours // len(template) + 1
            template = pd.concat([template] * repeat_count, ignore_index=True).head(hours)

        for h in range(hours):
            src = template.iloc[h]
            row = {col: src[col] for col in feature_cols if col in template.columns}
            if loc is not None:
                row["location_key"] = loc
            row["time"] = start_ts + pd.Timedelta(hours=h)
            row[target_col] = np.nan
            future_rows.append(row)

    if not future_rows:
        raise ValueError("Could not create future rows for prediction.")
    return pd.DataFrame(future_rows)


@torch.no_grad()
def predict_aqi_next(
    model: torch.nn.Module,
    x_inference: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    y_mean: float | None = None,
    y_std: float | None = None,
    use_amp: bool = False,
) -> np.ndarray:
    """Run a loaded Mamba model on `(batch, seq_len, num_features)` input."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    x_tensor = torch.as_tensor(x_inference, dtype=torch.float32, device=device)
    if x_tensor.ndim != 3:
        raise ValueError("x_inference must have shape (batch, seq_len, num_features).")

    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        pred = model(x_tensor).detach().float().cpu().numpy()

    if y_mean is not None and y_std is not None:
        pred = pred * float(y_std) + float(y_mean)
    return pred
