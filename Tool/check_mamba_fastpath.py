from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "Model"))

import torch
import mamba_ssm.modules.mamba_simple as mamba_simple
import mamba_ssm.ops.selective_scan_interface as selective_scan_interface


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"causal_conv1d_fn: {mamba_simple.causal_conv1d_fn is not None}")
    print(f"selective_scan_cuda: {selective_scan_interface.selective_scan_cuda is not None}")
    fast_ready = (
        mamba_simple.causal_conv1d_fn is not None
        and selective_scan_interface.selective_scan_cuda is not None
    )
    print(f"mamba fused fast path ready: {fast_ready}")


if __name__ == "__main__":
    main()
