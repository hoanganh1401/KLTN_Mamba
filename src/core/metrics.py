"""Shared evaluation metrics for AQI sequence models.

Used by Mamba, LSTM, and Transformer training/evaluation code.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, R2, and MSE between true and predicted values."""
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same number of elements, "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    nonfinite_count = int(len(y_true) - finite_mask.sum())
    if nonfinite_count:
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]

    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "mse": float("nan"),
            "nonfinite_count": nonfinite_count,
        }

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mse": mse,
        "nonfinite_count": nonfinite_count,
    }


def denormalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Convert an array from normalized space back to the original scale."""
    return np.asarray(arr, dtype=np.float32) * std + mean
