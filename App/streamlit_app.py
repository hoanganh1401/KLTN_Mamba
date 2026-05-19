"""
streamlit_app.py
----------------
Lightweight Streamlit UI for viewing saved training results.

Training is disabled in the UI; run training from terminal instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


def _project_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _list_runs(runs_dir: Path) -> list[Path]:
	if not runs_dir.exists():
		return []
	candidates = []
	for item in runs_dir.iterdir():
		if item.is_dir():
			if (item / "metrics_history.csv").exists() or (item / "best_mamba_aqi.pt").exists():
				candidates.append(item)
	return sorted(candidates, key=lambda p: p.name, reverse=True)


def _load_metrics(run_dir: Path) -> pd.DataFrame | None:
	path = run_dir / "metrics_history.csv"
	if not path.exists():
		return None
	return pd.read_csv(path)


def _load_predictions(inference_dir: Path) -> pd.DataFrame | None:
	path = inference_dir / "predictions.csv"
	if not path.exists():
		return None
	return pd.read_csv(path)


def main() -> None:
	st.set_page_config(page_title="AQI Mamba Results", layout="wide")
	st.title("AQI Mamba Results")
	st.caption("Training runs are executed in terminal; this UI only shows saved results.")

	root = _project_root()
	default_runs_dir = root / "runs"
	default_infer_dir = root / "artifacts" / "inference_input"

	st.sidebar.header("Paths")
	runs_dir = Path(st.sidebar.text_input("Runs directory", str(default_runs_dir))).expanduser()
	inference_dir = Path(st.sidebar.text_input("Inference directory", str(default_infer_dir))).expanduser()

	run_dirs = _list_runs(runs_dir)
	if not run_dirs:
		st.warning("No runs found. Train from terminal to generate runs.")
		return

	run_names = [p.name for p in run_dirs]
	selected_name = st.sidebar.selectbox("Select run", run_names, index=0)
	selected_run = run_dirs[run_names.index(selected_name)]

	st.subheader("Run summary")
	st.write(f"Run directory: {selected_run}")

	metrics_df = _load_metrics(selected_run)
	if metrics_df is None or metrics_df.empty:
		st.info("No metrics_history.csv found in this run.")
	else:
		last_row = metrics_df.tail(1).iloc[0].to_dict()
		st.write("Latest metrics")
		st.json(last_row)

		st.write("Training curves")
		chart_cols = [c for c in ["train_loss", "val_loss"] if c in metrics_df.columns]
		if chart_cols:
			st.line_chart(metrics_df[chart_cols])
		else:
			st.info("No train/val loss columns found in metrics_history.csv.")

		st.write("Metrics history")
		st.dataframe(metrics_df, use_container_width=True)

	ckpt_path = selected_run / "best_mamba_aqi.pt"
	st.write("Checkpoint")
	st.write(str(ckpt_path) if ckpt_path.exists() else "Checkpoint not found")

	st.subheader("Predictions")
	pred_df = _load_predictions(inference_dir)
	if pred_df is None or pred_df.empty:
		st.info("No predictions.csv found. Run inference from terminal to generate it.")
	else:
		st.line_chart(pred_df.set_index(pred_df.columns[0]))
		st.dataframe(pred_df, use_container_width=True)


if __name__ == "__main__":
	main()
