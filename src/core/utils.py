"""
core/utils.py
-------------
Logger, seed, and device helpers.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch


def setup_logger(out_dir: str, name: str = "trainer") -> logging.Logger:
    """Create a logger that writes to console and out_dir/train.log."""
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = os.path.join(out_dir, "train.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    """Set random seed for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
