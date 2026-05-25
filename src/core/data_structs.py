"""Shared data structures for time-series models.

`loc_ids` is kept only as metadata for reporting province/location names in
prediction CSV files. It is not returned by `AQIDataset` and is not fed into
the Mamba model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    """One train/val/test split represented as numpy arrays.

    Attributes
    ----------
    x_seq   : (N, T, F) input feature sequences
    loc_ids : (N,)      optional province/location ids for reporting only
    y       : (N,) or (N, H) target values
    """

    x_seq: np.ndarray
    loc_ids: np.ndarray | None
    y: np.ndarray
    y_ts: np.ndarray | None = None


class AQIDataset(Dataset):
    """PyTorch Dataset that returns `(x_seq, y)` for model training."""

    def __init__(self, split: SplitData) -> None:
        self.x_seq = torch.from_numpy(split.x_seq).float()
        self.y = torch.from_numpy(split.y).float()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_seq[idx], self.y[idx]
