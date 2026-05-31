"""compare_seasonality.py
Quick A/B test: compare training performance with vs without time features.

Steps per variant:
 - regenerate Gold for given date range
 - run prepare_training_dataset.py to create training dataset (run_id)
 - run train_mamba_aqi.py with small epochs and CPU
 - parse metrics_history.csv and return final val mae/rmse

Usage: run inside project venv:
python tests/compare_seasonality.py --locations DataSet/locations.jsonl --location-keys ha_noi --start-date 2026-05-20 --end-date 2026-05-24 --epochs 2
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

sys.path.insert(0, str(ROOT))
from src.Gold.gold_feature_engineering import run_feature_engineering


def run_gold_for_range(locations_path, location_keys, start_date, end_date, disable_time_features=False):
    cur = start_date
    keys = location_keys.split(",") if isinstance(location_keys, str) else location_keys
    while cur <= end_date:
        for key in keys:
            print(f"Running Gold {cur} {key} disable_time={disable_time_features}")
            run_feature_engineering(locations_path, cur.strftime("%Y-%m-%d"), None, [key], disable_time_features=disable_time_features)
        cur += timedelta(days=1)


def run_prepare_dataset(locations, location_keys, start_date, end_date, run_id):
    cmd = [PY, "src/Gold/prepare_training_dataset.py",
           "--locations", locations,
           "--location-keys", location_keys,
           "--start-date", start_date.strftime("%Y-%m-%d"),
           "--end-date", end_date.strftime("%Y-%m-%d"),
           "--run-id", run_id]
    print("Running prepare_training_dataset:", " ".join(cmd))
    subprocess.check_call(cmd)
    return f"training_dataset/run_id={run_id}"


def run_train(run_id, epochs, out_dir):
    cmd = [PY, "src/Model/train_mamba_aqi.py",
           "--run-id", run_id,
           "--epochs", str(epochs),
        "--device", "cuda",
          "--no-auto-cpu-for-slow-mamba",
          "--force-gpu",
           "--out-dir", out_dir]
    print("Running training:", " ".join(cmd))
    subprocess.check_call(cmd)
    return out_dir


def parse_metrics_csv(out_dir):
    hist = Path(out_dir) / "metrics_history.csv"
    if not hist.exists():
        raise FileNotFoundError(f"metrics_history.csv not found in {out_dir}")
    with open(hist, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("No rows in metrics history")
    last = rows[-1]
    return {k: float(last[k]) if k in last and last[k] not in (None, '') else None for k in ("epoch","train_loss","val_loss","mae","rmse")}


def experiment(locations, location_keys, start_date, end_date, epochs=2):
    results = {}
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for variant, disable in [("with_time", False), ("no_time", True)]:
        run_id = f"exp_{variant}_{ts}"
        # 1) Regenerate gold
        run_gold_for_range(locations, location_keys, start_date, end_date, disable_time_features=disable)
        # 2) Prepare dataset
        dataset_prefix = run_prepare_dataset(locations, location_keys, start_date, end_date, run_id)
        # 3) Train short
        out_dir = ROOT / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_train(run_id, epochs, str(out_dir))
        # 4) parse metrics
        metrics = parse_metrics_csv(str(out_dir))
        results[variant] = {
            "run_id": run_id,
            "dataset_prefix": dataset_prefix,
            "out_dir": str(out_dir),
            "metrics": metrics,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", required=True)
    parser.add_argument("--location-keys", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    sd = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    res = experiment(args.locations, args.location_keys, sd, ed, epochs=args.epochs)
    print("\n=== SUMMARY ===")
    for k, v in res.items():
        print(k, v["metrics"])
    # Decide better by lower val_loss then mae
    a = res["with_time"]["metrics"]
    b = res["no_time"]["metrics"]
    better = "with_time" if (a["val_loss"] or 1e9) < (b["val_loss"] or 1e9) else "no_time"
    print(f"Better variant by val_loss: {better}")


if __name__ == "__main__":
    main()
