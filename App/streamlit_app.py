"""Streamlit app for viewing Mamba AQI prediction results from MinIO."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.common.config import MINIO_ARTIFACTS_BUCKET, MINIO_HOST
from src.common.minio_io import get_client, load_bytes, load_json_object
from src.common.time_utils import parse_time_local

PREDICTION_ROOT = "mamba_inference"


def load_location_names() -> dict[str, str]:
    path = _REPO_ROOT / "DataSet" / "locations.jsonl"
    names: dict[str, str] = {}
    if not path.exists():
        return names
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = str(obj.get("location_key", "")).strip()
            if key:
                names[key] = str(obj.get("name") or key)
    return names


@st.cache_data(ttl=30)
def list_prediction_objects() -> list[str]:
    client = get_client()
    objects = client.list_objects(MINIO_ARTIFACTS_BUCKET, prefix=f"{PREDICTION_ROOT}/", recursive=True)
    return sorted(obj.object_name for obj in objects if obj.object_name.endswith("future_predictions.csv"))


def parse_prediction_path(path: str) -> dict[str, str]:
    parts = path.split("/")
    item = {"path": path, "province": "", "forecast_date": "", "run_id": ""}
    for part in parts:
        if part.startswith("province="):
            item["province"] = part.split("=", 1)[1]
        elif part.startswith("forecast_date="):
            item["forecast_date"] = part.split("=", 1)[1]
        elif part.startswith("run_id="):
            item["run_id"] = part.split("=", 1)[1]
    return item


@st.cache_data(ttl=30)
def load_prediction_csv(path: str) -> pd.DataFrame:
    raw = load_bytes(get_client(), MINIO_ARTIFACTS_BUCKET, path)
    if raw is None:
        raise FileNotFoundError(f"Missing s3://{MINIO_ARTIFACTS_BUCKET}/{path}")
    df = pd.read_csv(io.BytesIO(raw))
    if "forecast_time" in df.columns:
        df["forecast_time"] = parse_time_local(df["forecast_time"])
    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_prediction_metadata(path: str) -> dict:
    meta_path = path.rsplit("/", 1)[0] + "/prediction_metadata.json"
    return load_json_object(get_client(), MINIO_ARTIFACTS_BUCKET, meta_path) or {}


def latest_path_for_province(rows: list[dict[str, str]], province: str) -> str | None:
    candidates = [r for r in rows if r["province"] == province]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r["forecast_date"], r["run_id"]))
    return candidates[-1]["path"]


def render_prediction(df: pd.DataFrame, metadata: dict, path: str) -> None:
    st.caption(f"s3://{MINIO_ARTIFACTS_BUCKET}/{path}")

    clean = df.dropna(subset=["forecast_time", "y_pred"]).copy()
    if clean.empty:
        st.warning("Prediction file is empty or has invalid forecast_time/y_pred values.")
        st.dataframe(df, use_container_width=True)
        return

    province = str(clean["province"].iloc[0]) if "province" in clean.columns else ""
    forecast_start = clean["forecast_time"].min()
    forecast_end = clean["forecast_time"].max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Province", province)
    m2.metric("Rows", f"{len(clean):,}")
    m3.metric("Forecast start", forecast_start.strftime("%Y-%m-%d %H:%M"))
    m4.metric("Forecast end", forecast_end.strftime("%Y-%m-%d %H:%M"))

    chart_df = clean.sort_values("forecast_time").set_index("forecast_time")[["y_pred"]]
    st.line_chart(chart_df, use_container_width=True)

    st.dataframe(
        clean.sort_values(["province", "forecast_time"] if "province" in clean.columns else ["forecast_time"]),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction CSV",
        data=csv_bytes,
        file_name=f"{province or 'prediction'}_future_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Prediction metadata"):
        st.json(metadata)


def load_latest_predictions(rows: list[dict[str, str]], provinces: list[str]) -> pd.DataFrame:
    frames = []
    for province in provinces:
        path = latest_path_for_province(rows, province)
        if not path:
            continue
        df = load_prediction_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_path"] = path
        parsed = parse_prediction_path(path)
        df["forecast_date_partition"] = parsed["forecast_date"]
        df["run_id"] = parsed["run_id"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def render_overview(all_df: pd.DataFrame, location_names: dict[str, str]) -> None:
    clean = all_df.dropna(subset=["forecast_time", "y_pred"]).copy()
    if clean.empty:
        st.warning("No valid prediction rows found.")
        return

    clean["province_name"] = clean["province"].astype(str).map(lambda p: location_names.get(p, p))
    clean = clean.sort_values(["province", "forecast_time"])

    first_rows = clean.groupby("province", as_index=False).first()
    peak_idx = clean.groupby("province")["y_pred"].idxmax()
    peak_rows = clean.loc[peak_idx, ["province", "forecast_time", "y_pred"]].rename(
        columns={"forecast_time": "peak_time", "y_pred": "peak_aqi"}
    )
    summary = first_rows.merge(peak_rows, on="province", how="left")
    summary = summary.rename(columns={"forecast_time": "next_forecast_time", "y_pred": "next_aqi"})
    summary["province_name"] = summary["province"].map(lambda p: location_names.get(str(p), str(p)))
    summary = summary[
        [
            "province_name",
            "province",
            "forecast_date_partition",
            "run_id",
            "next_forecast_time",
            "next_aqi",
            "peak_time",
            "peak_aqi",
            "source_path",
        ]
    ].sort_values("province_name")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Provinces", f"{summary['province'].nunique():,}")
    m2.metric("Prediction rows", f"{len(clean):,}")
    m3.metric("Avg next AQI", f"{summary['next_aqi'].mean():.1f}")
    m4.metric("Max peak AQI", f"{summary['peak_aqi'].max():.1f}")

    top_next = (
        summary[["province_name", "next_aqi"]]
        .sort_values("next_aqi", ascending=False)
        .head(12)
        .set_index("province_name")
    )
    top_peak = (
        summary[["province_name", "peak_aqi"]]
        .sort_values("peak_aqi", ascending=False)
        .head(12)
        .set_index("province_name")
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.write("#### Highest next-hour AQI")
        st.bar_chart(top_next, use_container_width=True)
    with chart_right:
        st.write("#### Highest peak AQI in horizon")
        st.bar_chart(top_peak, use_container_width=True)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "province_name": "Province",
            "province": "Key",
            "forecast_date_partition": "Forecast date",
            "run_id": "Run",
            "next_forecast_time": "Next time",
            "next_aqi": st.column_config.NumberColumn("Next AQI", format="%.1f"),
            "peak_time": "Peak time",
            "peak_aqi": st.column_config.NumberColumn("Peak AQI", format="%.1f"),
            "source_path": "MinIO path",
        },
    )

    st.download_button(
        "Download overview CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="aqi_prediction_overview.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="AQI Mamba Predictions", layout="wide")
    st.title("AQI Mamba Predictions")
    st.caption("View per-province AQI forecasts saved on MinIO. Training controls are intentionally removed.")

    with st.sidebar:
        st.header("MinIO")
        st.caption(f"Endpoint: {MINIO_HOST}")
        st.caption(f"Bucket: {MINIO_ARTIFACTS_BUCKET}")
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    paths = list_prediction_objects()
    if not paths:
        st.info(f"No predictions found under s3://{MINIO_ARTIFACTS_BUCKET}/{PREDICTION_ROOT}/")
        return

    rows = [parse_prediction_path(path) for path in paths]
    location_names = load_location_names()
    provinces = sorted({r["province"] for r in rows if r["province"]})
    all_latest_df = load_latest_predictions(rows, provinces)

    with st.sidebar:
        selected_province = st.selectbox(
            "Province",
            provinces,
            format_func=lambda p: f"{location_names.get(p, p)} ({p})",
        )
        province_rows = [r for r in rows if r["province"] == selected_province]
        forecast_dates = sorted({r["forecast_date"] for r in province_rows if r["forecast_date"]}, reverse=True)
        selected_date = st.selectbox("Forecast date", forecast_dates)
        date_rows = [r for r in province_rows if r["forecast_date"] == selected_date]
        date_rows.sort(key=lambda r: r["run_id"], reverse=True)
        selected_run = st.selectbox("Run", [r["run_id"] for r in date_rows])

    overview_tab, detail_tab = st.tabs(["Overview", "Province detail"])

    with overview_tab:
        st.subheader("Latest Predictions For All Provinces")
        render_overview(all_latest_df, location_names)

    with detail_tab:
        st.subheader("Selected Province Prediction")
        selected_path = next(r["path"] for r in date_rows if r["run_id"] == selected_run)
        selected_df = load_prediction_csv(selected_path)
        selected_meta = load_prediction_metadata(selected_path)
        render_prediction(selected_df, selected_meta, selected_path)


if __name__ == "__main__":
    main()
