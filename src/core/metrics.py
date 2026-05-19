"""
core/metrics.py
---------------
Common evaluation metrics helpers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R2, MSE."""
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"mae": mae, "rmse": rmse, "r2": r2, "mse": mse}


def denormalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """De-normalize array back to original scale."""
    return np.asarray(arr, dtype=np.float32) * std + mean
