"""
ui_components.py
----------------
CÃ¡c component UI tÃ¡i sá»­ dá»¥ng: sidebar, metrics, báº£ng so sÃ¡nh, download, v.v.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import streamlit as st
from minio import Minio
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from Utils import format_time_utc_strings, load_train_module, parse_time_local, split_data_by_timeline

# ThÆ° má»¥c lÆ°u táº¡m file upload (náº±m cáº¡nh app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UPLOAD_TEMP_DIR = _PROJECT_ROOT / "runs" / "uploaded"
_CONFIG_PATH = _PROJECT_ROOT / "Conf" / "air_quality.yaml"
_CONFIG_CACHE: dict | None = None

_MINIO_HOST = os.environ.get("MINIO_HOST", "localhost:9004")
_MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "admin")
_MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "admin123")
_MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
_MINIO_GOLD_BUCKET = os.environ.get("MINIO_GOLD_BUCKET", "air-quality-gold")


def _get_minio_client() -> Minio:
    return Minio(
        _MINIO_HOST,
        access_key=_MINIO_ACCESS,
        secret_key=_MINIO_SECRET,
        secure=_MINIO_SECURE,
    )


def _gold_feature_path(location_key: str, day: date) -> str:
    return (
        f"feature_engineering/province={location_key}"
        f"/year={day.year}/month={day.month:02d}/day={day.day:02d}/data.csv"
    )


def _iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_locations_jsonl(path: str) -> list[str]:
    if not path:
        return []
    jsonl = Path(path)
    if not jsonl.exists():
        return []
    keys: list[str] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get("location_key") or f"{obj['latitude']}_{obj['longitude']}"
            keys.append(str(key))
    return keys


def _load_gold_features_minio(
    location_keys: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    client = _get_minio_client()
    frames: list[pd.DataFrame] = []

    for loc in location_keys:
        for day in _iter_dates(start_date, end_date):
            path = _gold_feature_path(loc, day)
            try:
                resp = client.get_object(_MINIO_GOLD_BUCKET, path)
                try:
                    raw = resp.read()
                finally:
                    resp.close()
                    resp.release_conn()
                df = pd.read_csv(io.BytesIO(raw))
            except Exception:
                continue

            if df.empty:
                continue

            df = df.copy()
            if "location_key" not in df.columns:
                df["location_key"] = df.get("location", loc)
            if "time" in df.columns:
                df["time"] = parse_time_local(df["time"])
            frames.append(df)

    if not frames:
        raise ValueError("No Gold features found for the selected range.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["time"]).copy()
    combined = combined.sort_values(["location_key", "time"]).reset_index(drop=True)

    if "Time" not in combined.columns and "time" in combined.columns:
        combined["Time"] = format_time_utc_strings(combined["time"])

    return combined


def _fmt_metric(value, fallback=np.nan) -> str:
    try:
        if value is None:
            value = fallback
        return f"{float(value):.4f}"
    except Exception:
        return f"{float(fallback):.4f}"


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _get_project_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if not _CONFIG_PATH.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    if yaml is None:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _CONFIG_CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


# ---------------------------------------------------------------------------
# Sidebar & dataset loading
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, object | None]:
    """Render sidebar nguá»“n dá»¯ liá»‡u, tráº£ vá» (source, data_path, uploaded_file)."""
    with st.sidebar:
        st.header("ðŸ“‚ Nguá»“n dá»¯ liá»‡u")

        # â”€â”€ Tráº¡ng thÃ¡i dataset hiá»‡n táº¡i â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _active_path = st.session_state.get("data_path", None)
        _active_df   = st.session_state.get("df", None)
        if _active_df is not None and _active_path:
            st.success(
                f"âœ… Dataset Ä‘ang dÃ¹ng:\n"
                f"`{Path(_active_path).name}`\n"
                f"{_active_df.shape[0]:,} rows Â· {_active_df.shape[1]} cols"
            )
        elif _active_df is not None:
            st.info(
                f"ðŸ“„ Dataset Ä‘ang dÃ¹ng: *(file upload táº¡m)*\n"
                f"{_active_df.shape[0]:,} rows Â· {_active_df.shape[1]} cols"
            )
        else:
            st.warning("âš ï¸ ChÆ°a load dataset. HÃ£y chá»n nguá»“n bÃªn dÆ°á»›i vÃ  nháº¥n **Load**.")

        st.divider()

        # â”€â”€ Chá»n nguá»“n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        source = st.radio(
            "Nguá»“n dataset",
            [
                "ðŸ—„ï¸ MinIO Gold (features)",
                "ðŸ“ ÄÆ°á»ng dáº«n trong workspace",
                "â¬†ï¸ Upload file CSV",
            ],
            index=0,
            help="Chá»n cÃ¡ch cung cáº¥p dá»¯ liá»‡u Ä‘áº§u vÃ o cho Mamba.",
        )
        use_minio = source.startswith("ðŸ—„ï¸")
        use_upload = source.startswith("â¬†ï¸")

        data_path = ""
        uploaded = None
        minio_locations: list[str] = []
        start_date: date | None = None
        end_date: date | None = None

        if use_minio:
            st.caption(f"MinIO: {_MINIO_HOST} Â· bucket: {_MINIO_GOLD_BUCKET}")

            loc_mode = st.radio(
                "Nguá»“n locations",
                ["Tá»± nháº­p", "Tá»« JSONL"],
                horizontal=True,
            )

            minio_locations: list[str] = []
            if loc_mode == "Tá»« JSONL":
                jsonl_path = st.text_input(
                    "Path JSONL locations",
                    value=str(_PROJECT_ROOT / "DataSet" / "locations.jsonl"),
                )
                minio_locations = _load_locations_jsonl(jsonl_path)
                if minio_locations:
                    minio_locations = st.multiselect(
                        "Chá»n locations",
                        options=minio_locations,
                        default=minio_locations[: min(3, len(minio_locations))],
                    )
            else:
                loc_text = st.text_input(
                    "Location keys (comma-separated)",
                    value="",
                )
                minio_locations = [x.strip() for x in loc_text.split(",") if x.strip()]

            end_default = datetime.now().date()
            start_default = end_default - timedelta(days=30)
            start_date = st.date_input("Start date", value=start_default)
            end_date = st.date_input("End date", value=end_default)

        elif not use_upload:
            data_path = st.text_input(
                "Path CSV (tÆ°Æ¡ng Ä‘á»‘i hoáº·c tuyá»‡t Ä‘á»‘i)",
                value=st.session_state.get("_sidebar_path_input", ""),
                help="VÃ­ dá»¥: debug local only; MinIO Gold is the main data source",
                key="_sidebar_path_input",
            )
            st.caption(f"ðŸ“Œ ThÆ° má»¥c gá»‘c: `{_PROJECT_ROOT}`")
        else:
            uploaded = st.file_uploader(
                "Chá»n file CSV Ä‘á»ƒ upload",
                type=["csv"],
                help="File sáº½ Ä‘Æ°á»£c lÆ°u táº¡m vÃ o thÆ° má»¥c `runs/uploaded/`.",
            )
            if uploaded is not None:
                st.caption(f"ðŸ“Ž File Ä‘Ã£ chá»n: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        # â”€â”€ NÃºt Load / Reset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            load_clicked = st.button("â¬‡ï¸ Load", use_container_width=True, type="primary")
        with btn_col2:
            reset_clicked = st.button("ðŸ—‘ï¸ Reset", use_container_width=True)

        if reset_clicked:
            for key in ["df", "data_path", "_upload_saved_path"]:
                st.session_state.pop(key, None)
            st.rerun()

        if load_clicked:
            if use_minio:
                _load_minio_dataset(minio_locations, start_date, end_date)
            else:
                _load_dataset(use_upload, data_path, uploaded)

    return source, data_path, uploaded


def _load_dataset(use_upload: bool, data_path: str, uploaded) -> None:
    """Load dataset vÃ o session_state['df'] vÃ  lÆ°u path tuyá»‡t Ä‘á»‘i vÃ o session_state['data_path']."""
    try:
        if use_upload:
            # â”€â”€ TrÆ°á»ng há»£p upload file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if uploaded is None:
                st.sidebar.error("âŒ Báº¡n chÆ°a chá»n file CSV Ä‘á»ƒ upload.")
                return

            # LÆ°u file táº¡m ra disk Ä‘á»ƒ TFT subprocess Ä‘á»c Ä‘Æ°á»£c Ä‘Æ°á»ng dáº«n thá»±c
            _UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = _UPLOAD_TEMP_DIR / uploaded.name
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())

            df = pd.read_csv(saved_path)
            st.session_state["df"]              = df
            st.session_state["data_path"]       = str(saved_path.resolve())
            st.session_state["_upload_saved_path"] = str(saved_path.resolve())

            st.sidebar.success(
                f"âœ… Upload thÃ nh cÃ´ng: **{uploaded.name}**\n"
                f"{df.shape[0]:,} rows Â· {df.shape[1]} cols"
            )

        else:
            # â”€â”€ TrÆ°á»ng há»£p nháº­p Ä‘Æ°á»ng dáº«n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if not data_path or not data_path.strip():
                st.sidebar.error("âŒ ÄÆ°á»ng dáº«n CSV khÃ´ng Ä‘Æ°á»£c Ä‘á»ƒ trá»‘ng.")
                return

            raw_path = Path(data_path.strip())
            abs_path = raw_path if raw_path.is_absolute() else (_PROJECT_ROOT / raw_path)
            abs_path = abs_path.resolve()
            if not abs_path.exists():
                st.sidebar.error(
                    f"âŒ KhÃ´ng tÃ¬m tháº¥y file:\n`{abs_path}`\n\n"
                    "HÃ£y kiá»ƒm tra láº¡i Ä‘Æ°á»ng dáº«n hoáº·c dÃ¹ng Ä‘Æ°á»ng dáº«n tuyá»‡t Ä‘á»‘i."
                )
                return

            df = pd.read_csv(abs_path)
            st.session_state["df"]        = df
            st.session_state["data_path"] = str(abs_path)

            st.sidebar.success(
                f"âœ… Load thÃ nh cÃ´ng: **{Path(abs_path).name}**\n"
                f"{df.shape[0]:,} rows Â· {df.shape[1]} cols"
            )

    except Exception as e:
        st.sidebar.error(f"âŒ Load dataset lá»—i: {e}")


def _load_minio_dataset(
    location_keys: list[str],
    start_date: date | None,
    end_date: date | None,
) -> None:
    try:
        if not location_keys:
            st.sidebar.error("âŒ ChÆ°a chá»n location Ä‘á»ƒ load tá»« MinIO.")
            return
        if start_date is None or end_date is None:
            st.sidebar.error("âŒ Cáº§n chá»n start/end date.")
            return
        if start_date > end_date:
            st.sidebar.error("âŒ Start date pháº£i <= End date.")
            return

        df = _load_gold_features_minio(location_keys, start_date, end_date)

        st.session_state["df"] = df
        st.session_state["data_path"] = (
            f"minio://{_MINIO_GOLD_BUCKET}/feature_engineering"
            f"?start={start_date.isoformat()}&end={end_date.isoformat()}"
        )

        st.sidebar.success(
            f"âœ… Load MinIO thÃ nh cÃ´ng ({len(location_keys)} locations)\n"
            f"{df.shape[0]:,} rows Â· {df.shape[1]} cols"
        )
    except Exception as e:
        st.sidebar.error(f"âŒ Load MinIO lá»—i: {e}")


# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------

def render_data_preview(df: pd.DataFrame) -> None:
    """Hiá»ƒn thá»‹ preview vÃ  thÃ´ng tin cÆ¡ báº£n cá»§a dataset."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview dá»¯ liá»‡u")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("ThÃ´ng tin")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")


# ---------------------------------------------------------------------------
# Location selector + sample preview
# ---------------------------------------------------------------------------

def render_location_selector(df: pd.DataFrame) -> list[str]:
    """Render location multiselect + sample count preview. Tráº£ vá» selected_locations."""
    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist()) if "location_key" in df.columns else []
    st.subheader("Chá»n Ä‘á»‹a Ä‘iá»ƒm Ä‘á»ƒ train + forecast")
    selected_locations = st.multiselect(
        "Chá»n Ä‘á»‹a Ä‘iá»ƒm Ä‘á»ƒ train + forecast",
        options=locations,
        default=locations[: min(3, len(locations))],
        help="CÃ³ thá»ƒ chá»n 1 hoáº·c nhiá»u Ä‘á»‹a Ä‘iá»ƒm. MÃ´ hÃ¬nh sáº½ train chung theo nhiá»u tá»‰nh.",
    )

    preview_window = int(st.session_state.get("train_window_size", 96))
    preview_horizon = int(st.session_state.get("train_horizon", 24))
    preview_stride = int(st.session_state.get("train_sample_stride", 1))

    if selected_locations:
        _render_sample_count_preview(
            df, selected_locations, int(preview_window), int(preview_horizon), int(preview_stride)
        )

    return selected_locations


def _render_sample_count_preview(
    df: pd.DataFrame, selected_locations: list[str], window: int, horizon: int, sample_stride: int
) -> None:
    """Hiá»ƒn thá»‹ sá»‘ lÆ°á»£ng sample train/val/test theo preview window/horizon."""
    try:
        cfg = _get_project_config()
        target_name = cfg.get("data", {}).get("target_col", "aqi")

        df_sel = df.loc[
            df["location_key"].astype(str).isin([str(x) for x in selected_locations])
        ].copy()
        if target_name in df_sel.columns:
            default_target = target_name
        else:
            default_target = next(
                (c for c in df_sel.select_dtypes(include=["number"]).columns if c != "_loc_id"),
                None,
            )
        if default_target is None:
            st.warning("KhÃ´ng tÃ¬m tháº¥y cá»™t sá»‘ Ä‘á»ƒ preview sample counts.")
            return

        mod = load_train_module()
        if mod is None or not hasattr(mod, "build_time_series_samples"):
            st.warning("KhÃ´ng thá»ƒ load helper 'build_time_series_samples' Ä‘á»ƒ preview samples.")
            return

        col_map = {c.lower(): c for c in df_sel.columns}
        ts_col = col_map.get("time") or col_map.get("timestamp") or col_map.get("ts_utc")
        exclude_cols = [c for c in [ts_col, col_map.get("location_key") or "location_key"] if c]
        feature_cols = [c for c in df_sel.columns if c not in exclude_cols]
        if default_target not in feature_cols:
            feature_cols.append(default_target)

        x_seq, loc_ids, y, y_ts, _, _ = mod.build_time_series_samples(
            df_sel,
            default_target,
            window,
            horizon,
            sample_stride=sample_stride,
            feature_cols=feature_cols,
            include_target_history=True,
        )
        if hasattr(mod, "split_data_by_timeline"):
            train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)
        else:
            train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)

        st.markdown(
            f"**Preview ({len(selected_locations)} locations)**: "
            f"total samples={len(y):,} | sliding step={sample_stride}"
        )
        st.write(f"Train: {len(train.y):,}  |  Val: {len(val.y):,}  |  Test: {len(test.y):,}")
    except Exception as e:
        st.warning(f"KhÃ´ng thá»ƒ tÃ­nh preview samples: {e}")


# ---------------------------------------------------------------------------
# Train config form
# ---------------------------------------------------------------------------

def render_train_config() -> dict:
    """Render toÃ n bá»™ form cáº¥u hÃ¬nh train. Tráº£ vá» dict config."""
    cfg = _get_project_config()
    data_cfg = cfg.get("data", {})
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    df = st.session_state.get("df", pd.DataFrame())
    all_cols = df.columns.tolist()
    reserved_cols = {"y_true", "y_pred", "abs_error"}
    blocked_lower = {"ts_utc", "time", "timestamp", "location_key"}
    feature_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c.lower() not in blocked_lower
        and not c.lower().startswith("unnamed:")
    ]

    st.subheader("Cáº¥u hÃ¬nh train")
    conf1, conf2, conf3 = st.columns(3)

    with conf1:
        target_default = data_cfg.get("target_col", "aqi")
        target_index = feature_options.index(target_default) if target_default in feature_options else 0
        target_col = st.selectbox(
            "Target column (biáº¿n cáº§n dá»± Ä‘oÃ¡n)",
            options=feature_options,
            index=target_index,
        )
        default_features = list(feature_options)
        feature_cols = st.multiselect(
            "Input feature columns",
            options=feature_options,
            default=default_features,
        )
        loss_default = str(training_cfg.get("loss", "huber")).lower()
        loss_options = ["huber", "mse"]
        loss_index = loss_options.index(loss_default) if loss_default in loss_options else 0
        loss_name = st.selectbox("Loss", options=loss_options, index=loss_index)

    with conf2:
        window_size = st.number_input(
            "Window size (timesteps)",
            min_value=1,
            max_value=168,
            value=_as_int(dataset_cfg.get("seq_len"), 96),
            step=1,
        )
        horizon = st.number_input(
            "Horizon (timesteps)",
            min_value=1,
            max_value=168,
            value=_as_int(dataset_cfg.get("pred_len"), 24),
            step=1,
        )
        sample_stride = st.number_input(
            "Sliding step",
            min_value=1,
            max_value=168,
            value=_as_int(dataset_cfg.get("sample_stride"), 1),
            step=1,
        )
        st.session_state["train_window_size"] = int(window_size)
        st.session_state["train_horizon"] = int(horizon)
        st.session_state["train_sample_stride"] = int(sample_stride)
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=200,
            value=_as_int(training_cfg.get("epochs"), 50),
            step=1,
        )
        early_stop_patience = st.number_input(
            "Early stop patience",
            min_value=0,
            max_value=50,
            value=_as_int(training_cfg.get("patience"), 5),
            step=1,
        )
        early_stop_min_delta = st.number_input(
            "Min delta",
            min_value=0.0,
            max_value=1.0,
            value=_as_float(training_cfg.get("min_delta"), 0.0),
            format="%.6f",
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=8,
            max_value=8192,
            value=_as_int(training_cfg.get("batch_size"), 128),
            step=8,
        )
        lr = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-1,
            value=_as_float(training_cfg.get("learning_rate"), 3e-4),
            format="%.6f",
        )
        weight_decay = st.number_input(
            "Weight decay",
            min_value=0.0,
            max_value=1.0,
            value=_as_float(training_cfg.get("weight_decay"), 1e-4),
            format="%.6f",
        )

    with conf3:
        d_model = st.number_input(
            "d_model",
            min_value=16,
            max_value=512,
            value=_as_int(model_cfg.get("d_model"), 64),
            step=16,
        )
        n_layers = st.number_input(
            "n_layers",
            min_value=1,
            max_value=8,
            value=_as_int(model_cfg.get("n_layers"), 2),
            step=1,
        )
        grad_accum_steps = st.number_input(
            "Gradient accumulation",
            min_value=1,
            max_value=64,
            value=_as_int(training_cfg.get("grad_accum_steps"), 1),
            step=1,
        )
        max_grad_norm = st.number_input(
            "Max grad norm",
            min_value=0.0,
            max_value=100.0,
            value=_as_float(training_cfg.get("max_grad_norm"), 1.0),
            step=0.5,
        )
        num_workers = st.number_input(
            "Num workers",
            min_value=0,
            max_value=16,
            value=_as_int(training_cfg.get("num_workers"), 0),
            step=1,
        )

    run1, run2, run3 = st.columns(3)
    with run1:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=_as_int(training_cfg.get("seed"), 42), step=1)
    with run2:
        st.markdown(" ")
    with run3:
        device_default = str(training_cfg.get("device", "cuda")).lower()
        use_gpu = st.checkbox("DÃ¹ng GPU (náº¿u cÃ³)", value=device_default == "cuda")

    import torch
    if use_gpu and not torch.cuda.is_available():
        st.warning(
            "Báº¡n Ä‘ang báº­t GPU nhÆ°ng PyTorch hiá»‡n khÃ´ng nháº­n CUDA (torch+cpu). "
            "Train sáº½ cháº¡y báº±ng CPU nÃªn thá»i gian má»—i epoch sáº½ cao."
        )

    return dict(
        target_col=target_col,
        feature_cols=feature_cols,
        loss_name=loss_name,
        window_size=int(window_size),
        horizon=int(horizon),
        sample_stride=int(sample_stride),
        epochs=int(epochs),
        early_stop_patience=int(early_stop_patience),
        early_stop_min_delta=float(early_stop_min_delta),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        d_model=int(d_model),
        n_layers=int(n_layers),
        grad_accum_steps=int(grad_accum_steps),
        max_grad_norm=float(max_grad_norm),
        num_workers=int(num_workers),
        seed=int(seed),
        use_gpu=bool(use_gpu),
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_mamba_results(summary: dict, hist_df: pd.DataFrame, future_df: pd.DataFrame, df: pd.DataFrame, selected_locations: list[str]) -> None:
    """Hiá»ƒn thá»‹ káº¿t quáº£ sau khi train Mamba."""
    st.success("Train/Test hoÃ n táº¥t")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Val MAE (norm)", _fmt_metric(summary.get("val_mae_norm")))
    met2.metric("Val RMSE (norm)", _fmt_metric(summary.get("val_rmse_norm")))
    met3.metric("Val R2", f"{summary['val_r2']:.4f}")
    met4.metric("Locations done", f"{int(summary['future_locations']):,}")

    st.write("### Sá»‘ dÃ²ng sau khi lá»c theo Ä‘á»‹a Ä‘iá»ƒm")
    train_counts = (
        df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations]), "location_key"]
        .astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="train_source_rows")
    )
    test_counts = pd.DataFrame(
        {
            "location_key": selected_locations,
            "test_source_rows": [int(summary["split_test"])] * len(selected_locations),
        }
    )
    used_counts = (
        future_df["location"].astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="future_rows")
    )
    stats_df = (
        train_counts
        .merge(test_counts, on="location_key", how="outer")
        .merge(used_counts, on="location_key", how="outer")
        .fillna(0)
    )

    merged_stats = stats_df.merge(
        pd.DataFrame(
            [
                {
                    "n_rows_used": summary["n_rows_used"],
                    "split_train": summary["split_train"],
                    "split_val": summary["split_val"],
                    "split_test": summary["split_test"],
                }
            ]
        ),
        how="cross",
    )
    st.dataframe(merged_stats, use_container_width=True)

    st.write("### Thá»‘ng kÃª split")
    st.write(
        {k: round(float(v), 2) if isinstance(v, float) else v
         for k, v in {
             "split_train": summary["split_train"],
             "split_val": summary["split_val"],
             "split_test": summary["split_test"],
             "n_rows_used": summary["n_rows_used"],
             "future_rows": summary["future_rows"],
             "future_locations": summary["future_locations"],
             "train_only_sec": summary.get("train_only_sec", np.nan),
             "eval_sec": summary.get("eval_sec", np.nan),
             "forecast_sec": summary.get("forecast_sec", np.nan),
             "io_sec": summary.get("io_sec", np.nan),
             "run_sec": summary["run_sec"],
         }.items()}
    )

    st.write("### Lá»‹ch sá»­ train Mamba")
    st.dataframe(hist_df, use_container_width=True)


def render_forecast_download(future_df: pd.DataFrame, summary: dict) -> None:
    """Hiá»ƒn thá»‹ báº£ng dá»± bÃ¡o 24h vÃ  nÃºt download."""
    st.write("### Dá»± bÃ¡o 24 giá» tiáº¿p theo (tá»«ng Ä‘á»‹a Ä‘iá»ƒm)")
    st.dataframe(future_df.head(300), use_container_width=True)
    st.download_button(
        "Download file tá»•ng (má»i location)",
        data=future_df.to_csv(index=False).encode("utf-8"),
        file_name="future_24h_predictions.csv",
        mime="text/csv",
    )
    st.info("Mamba xuáº¥t 1 file tá»•ng trong run_dir: future_24h_predictions.csv (gá»“m time, location, predicted).")
    st.code(
        f"run_dir: {os.path.dirname(summary['future_pred_path'])}\n"
        "Files: future_24h_predictions.csv"
    )

