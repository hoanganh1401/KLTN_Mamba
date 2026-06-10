"""CLI wrapper for Mamba AQI training."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.Model.Common.train_sequence_aqi import main as _shared_main


def main() -> None:
    if "--model-type" not in sys.argv:
        sys.argv = [sys.argv[0], "--model-type", "mamba", *sys.argv[1:]]
    _shared_main()


if __name__ == "__main__":
    main()
