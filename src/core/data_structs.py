"""
core/data_structs.py
--------------------
Shared data structures for model training.

Includes:
- SplitData  : dataclass with x_seq / loc_ids / y arrays
- AQIDataset : PyTorch Dataset wrapping SplitData
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    """Holds one split (train/val/test) as numpy arrays.

    Attributes
    ----------
    x_seq   : (N, T, F) feature sequences
    loc_ids : (N,)      integer location ids
    y       : (N, H)    target values
    """

    x_seq: np.ndarray
    loc_ids: np.ndarray
    y: np.ndarray


class AQIDataset(Dataset):
    """PyTorch Dataset wrapping SplitData."""

    def __init__(self, split: SplitData) -> None:
        self.x_seq = torch.from_numpy(split.x_seq).float()
        self.loc_ids = torch.from_numpy(split.loc_ids).long()
        self.y = torch.from_numpy(split.y).float()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_seq[idx], self.loc_ids[idx], self.y[idx]
