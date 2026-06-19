"""Public-facing Streamlit dashboard for AQI forecast results stored on MinIO."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # Keep the dashboard usable in lightweight local envs.
    go = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.common.config import MINIO_ARTIFACTS_BUCKET, MINIO_HOST
from src.common.minio_io import get_client, load_bytes, load_csv_object, load_json_object
from src.common.time_utils import now_local, parse_time_local

PREDICTION_ROOT = "mamba_inference"
MINIO_RAW_BUCKETS = list(dict.fromkeys([os.environ.get("MINIO_BUCKET", "air-quality"), "air-quality"]))

AQI_LEVELS = [
    {"max": 50, "label": "Good", "vi": "Tốt", "color": "#22c55e", "text": "#052e16"},
    {"max": 100, "label": "Moderate", "vi": "Trung bình", "color": "#eab308", "text": "#422006"},
    {"max": 150, "label": "Unhealthy for sensitive groups", "vi": "Kém cho nhóm nhạy cảm", "color": "#f97316", "text": "#431407"},
    {"max": 200, "label": "Unhealthy", "vi": "Xấu", "color": "#ef4444", "text": "#450a0a"},
    {"max": 300, "label": "Very unhealthy", "vi": "Rất xấu", "color": "#a855f7", "text": "#3b0764"},
    {"max": 500, "label": "Hazardous", "vi": "Nguy hại", "color": "#7f1d1d", "text": "#ffffff"},
]

AQI_ADVICE = {
    "Good": "Không khí phù hợp cho hầu hết hoạt động ngoài trời.",
    "Moderate": "Người nhạy cảm nên theo dõi triệu chứng nếu hoạt động lâu ngoài trời.",
    "Unhealthy for sensitive groups": "Trẻ em, người cao tuổi và người có bệnh hô hấp nên giảm hoạt động ngoài trời.",
    "Unhealthy": "Nên hạn chế hoạt động ngoài trời kéo dài; cân nhắc khẩu trang khi ra đường.",
    "Very unhealthy": "Tránh vận động ngoài trời; đóng cửa sổ và dùng lọc không khí nếu có.",
    "Hazardous": "Hạn chế ra ngoài tối đa; ưu tiên ở trong nhà và theo dõi khuyến cáo địa phương.",
}




def render_html(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)
def repair_mojibake(text: str) -> str:
    if not isinstance(text, str) or not any(marker in text for marker in ("Ã", "Ä", "Æ", "á»", "áº")):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def aqi_info(value: float | int | None) -> dict[str, str | int | float]:
    if value is None or pd.isna(value):
        return {"label": "Unknown", "vi": "Chưa có dữ liệu", "color": "#94a3b8", "text": "#0f172a", "max": 0}
    value = max(0, float(value))
    for level in AQI_LEVELS:
        if value <= level["max"]:
            return level
    return AQI_LEVELS[-1]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --aqi-bg: #f8fafc;
            --aqi-card: #ffffff;
            --aqi-ink: #0f172a;
            --aqi-muted: #64748b;
            --aqi-line: #e2e8f0;
        }
        .stApp { background: var(--aqi-bg); color: var(--aqi-ink); }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--aqi-line); }
        [data-testid="stHeader"] { background: rgba(248, 250, 252, 0.86); backdrop-filter: blur(10px); }
        div.block-container { padding-top: 1.4rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: 0; color: var(--aqi-ink); }
        .hero {
            padding: 22px 24px; border: 1px solid var(--aqi-line); border-radius: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); margin-bottom: 18px;
        }
        .eyebrow { color: #2563eb; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
        .hero-title { font-size: 2.05rem; line-height: 1.15; font-weight: 760; margin: 0; }
        .hero-subtitle { color: var(--aqi-muted); font-size: 1rem; margin-top: 8px; max-width: 780px; }
        .hero-meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
        .pill { display: inline-flex; align-items: center; padding: 7px 10px; border-radius: 999px; border: 1px solid var(--aqi-line); background: #ffffff; color: #334155; font-size: .86rem; font-weight: 650; }
        .card { background: var(--aqi-card); border: 1px solid var(--aqi-line); border-radius: 8px; padding: 16px; min-height: 118px; box-shadow: 0 10px 30px rgba(15,23,42,.04); }
        .card-label { color: var(--aqi-muted); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
        .card-value { color: var(--aqi-ink); font-size: 1.72rem; font-weight: 780; margin-top: 8px; line-height: 1.1; }
        .card-help { color: var(--aqi-muted); font-size: .86rem; margin-top: 8px; }
        .status-card { border-radius: 8px; padding: 18px; border: 1px solid rgba(15,23,42,.08); min-height: 180px; }
        .status-aqi { font-size: 3.4rem; line-height: 1; font-weight: 820; margin: 10px 0 4px; }
        .status-label { display: inline-flex; padding: 7px 10px; border-radius: 999px; background: rgba(255,255,255,.72); font-weight: 760; font-size: .9rem; }
        .section-title { font-size: 1.15rem; font-weight: 760; margin: 8px 0 10px; color: var(--aqi-ink); }
        .risk-row { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 10px; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--aqi-line); }
        .risk-row:last-child { border-bottom: 0; }
        .risk-name { font-weight: 720; color: var(--aqi-ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .risk-time { color: var(--aqi-muted); font-size: .82rem; margin-top: 2px; }
        .aqi-badge { min-width: 58px; text-align: center; border-radius: 999px; padding: 6px 9px; font-weight: 800; }
        .legend { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 8px; margin: 8px 0 14px; }
        .legend-item { border-radius: 8px; padding: 9px; font-size: .78rem; font-weight: 740; min-height: 58px; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--aqi-line); border-radius: 8px; padding: 12px 14px; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 12px; background: #ffffff; border: 1px solid var(--aqi-line); color: #334155 !important; }
        .stTabs [data-baseweb="tab"] * { color: #334155 !important; opacity: 1 !important; }
        .stTabs [role="tab"] p, .stTabs [role="tab"] span { color: #334155 !important; opacity: 1 !important; }
        .stTabs [aria-selected="true"] { background: #dbeafe; border-color: #2563eb; color: #0f172a !important; }
        .stTabs [aria-selected="true"] * { color: #0f172a !important; font-weight: 760 !important; opacity: 1 !important; }
        .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span { color: #0f172a !important; font-weight: 760 !important; opacity: 1 !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #0f172a !important; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #334155 !important; }
        [data-baseweb="select"] > div { background: #ffffff !important; border-color: #cbd5e1 !important; }
        [data-baseweb="select"] span { color: #0f172a !important; }
        .stButton button { background: #2563eb !important; color: #ffffff !important; border: 0 !important; }
        .status-card .card-help { color: inherit; opacity: .78; }
        @media (max-width: 900px) { .hero-title { font-size: 1.55rem; } .legend { grid-template-columns: repeat(2,minmax(0,1fr)); } .status-aqi { font-size: 2.6rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                names[key] = repair_mojibake(str(obj.get("name") or key))
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
        raise FileNotFoundError("Prediction file is missing from artifact storage")
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



def format_time(value: pd.Timestamp | None, fmt: str = "%d/%m %H:%M") -> str:
    if value is None or pd.isna(value):
        return "--"
    return pd.Timestamp(value).strftime(fmt)


def format_aqi(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.0f}"


def prepare_clean_predictions(df: pd.DataFrame, location_names: dict[str, str] | None = None) -> pd.DataFrame:
    clean = df.dropna(subset=["forecast_time", "y_pred"]).copy()
    if clean.empty:
        return clean
    clean["y_pred"] = clean["y_pred"].clip(lower=0)
    if "province" not in clean.columns:
        clean["province"] = ""
    location_names = location_names or {}
    clean["province_name"] = clean["province"].astype(str).map(lambda p: location_names.get(p, p))
    clean["level"] = clean["y_pred"].map(lambda v: str(aqi_info(v)["vi"]))
    clean["level_en"] = clean["y_pred"].map(lambda v: str(aqi_info(v)["label"]))
    clean["level_color"] = clean["y_pred"].map(lambda v: str(aqi_info(v)["color"]))
    return clean.sort_values(["province", "forecast_time"])


def build_summary(all_df: pd.DataFrame, location_names: dict[str, str]) -> pd.DataFrame:
    clean = prepare_clean_predictions(all_df, location_names)
    if clean.empty:
        return clean
    first_rows = clean.groupby("province", as_index=False).first()
    peak_idx = clean.groupby("province")["y_pred"].idxmax()
    peak_rows = clean.loc[peak_idx, ["province", "forecast_time", "y_pred"]].rename(columns={"forecast_time": "peak_time", "y_pred": "peak_aqi"})
    avg_rows = clean.groupby("province", as_index=False)["y_pred"].mean().rename(columns={"y_pred": "avg_aqi"})
    summary = first_rows.merge(peak_rows, on="province", how="left").merge(avg_rows, on="province", how="left")
    summary = summary.rename(columns={"forecast_time": "next_forecast_time", "y_pred": "next_aqi"})
    summary["province_name"] = summary["province"].map(lambda p: location_names.get(str(p), str(p)))
    summary["next_level"] = summary["next_aqi"].map(lambda v: str(aqi_info(v)["vi"]))
    summary["peak_level"] = summary["peak_aqi"].map(lambda v: str(aqi_info(v)["vi"]))
    summary["next_color"] = summary["next_aqi"].map(lambda v: str(aqi_info(v)["color"]))
    return summary[["province_name", "province", "forecast_date_partition", "run_id", "next_forecast_time", "next_aqi", "next_level", "avg_aqi", "peak_time", "peak_aqi", "peak_level", "source_path", "next_color"]].sort_values(["next_aqi", "peak_aqi"], ascending=False)


def render_hero(summary: pd.DataFrame) -> None:
    if summary.empty:
        subtitle = "Chưa có dữ liệu dự báo hợp lệ để hiển thị."
        meta = ""
    else:
        horizon_start = summary["next_forecast_time"].min()
        horizon_end = summary["peak_time"].max()
        meta = f"""<div class="hero-meta">
<span class="pill">Cập nhật: {format_time(horizon_start, "%d/%m/%Y %H:%M")}</span>
<span class="pill">Số tỉnh/thành: {summary['province'].nunique()}</span>
<span class="pill">Khung dự báo: {format_time(horizon_start)} - {format_time(horizon_end)}</span>
</div>"""
        subtitle = "Theo dõi dự báo chỉ số chất lượng không khí theo tỉnh/thành, nhận diện điểm nóng và thời điểm AQI tăng cao."
    render_html(f"""
        <div class="hero">
            <div class="eyebrow">Air quality forecast</div>
            <div class="hero-title">Dự báo AQI Việt Nam</div>
            <div class="hero-subtitle">{subtitle}</div>
            {meta}
        </div>
        """)


def render_aqi_legend() -> None:
    items = []
    for idx, level in enumerate(AQI_LEVELS):
        lower = 0 if idx == 0 else int(AQI_LEVELS[idx - 1]["max"]) + 1
        upper = int(level["max"])
        items.append(
            f"<div class='legend-item' style='background:{level['color']}; color:{level['text']}'>"
            f"<div class='range'>{lower}-{upper}</div><div>{level['vi']}</div></div>"
        )
    components.html(
        "<style>body{margin:0;font-family:Arial,sans-serif;}"
        ".legend{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;}"
        ".legend-item{border-radius:8px;padding:10px;min-height:58px;font-size:13px;font-weight:700;box-sizing:border-box;}"
        ".range{font-weight:800;margin-bottom:4px;}"
        "@media(max-width:900px){.legend{grid-template-columns:repeat(2,minmax(0,1fr));}}"
        "</style>"
        f"<div class='legend'>{''.join(items)}</div>",
        height=86,
    )


def render_card(label: str, value: str, help_text: str) -> None:
    render_html(f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-help">{help_text}</div>
        </div>
        """)


def render_status_card(title: str, aqi_value: float, timestamp: pd.Timestamp | None, province: str = "") -> None:
    info = aqi_info(aqi_value)
    advice = AQI_ADVICE.get(str(info["label"]), "Theo dõi thêm khi có bản tin mới.")
    place = f"<div class='card-help'>{province}</div>" if province else ""
    render_html(f"""
        <div class="status-card" style="background:{info['color']}; color:{info['text']}">
            <div style="font-weight:760; opacity:.86">{title}</div>
            {place}
            <div class="status-aqi">{format_aqi(aqi_value)}</div>
            <div class="status-label">{info['vi']}</div>
            <div style="margin-top:14px; font-size:.94rem; line-height:1.45">{advice}</div>
            <div style="margin-top:10px; font-size:.82rem; opacity:.72">{format_time(timestamp, "%d/%m/%Y %H:%M")}</div>
        </div>
        """)


def render_risk_list(summary: pd.DataFrame, value_col: str, time_col: str, limit: int = 7) -> None:
    html_rows = []
    for _, row in summary.sort_values(value_col, ascending=False).head(limit).iterrows():
        info = aqi_info(row[value_col])
        html_rows.append(f"""
            <div class="risk-row">
                <div><div class="risk-name">{row['province_name']}</div><div class="risk-time">{format_time(row[time_col], "%d/%m %H:%M")} - {info['vi']}</div></div>
                <div class="aqi-badge" style="background:{info['color']}; color:{info['text']}">{format_aqi(row[value_col])}</div>
            </div>
        """)
    render_html("".join(html_rows))


def add_aqi_bands(fig: go.Figure, y_max: float) -> None:
    for y0, y1, color in [(0, 50, "#dcfce7"), (50, 100, "#fef9c3"), (100, 150, "#ffedd5"), (150, 200, "#fee2e2"), (200, 300, "#f3e8ff"), (300, max(500, y_max), "#fee2e2")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.34, line_width=0, layer="below")


def style_chart(fig: go.Figure, title: str, y_max: float) -> go.Figure:
    add_aqi_bands(fig, y_max)
    axis_color = "#1e293b"
    grid_color = "#cbd5e1"
    fig.update_layout(
        title=None,
        height=430,
        margin=dict(l=14, r=18, t=58, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=axis_color, size=13),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
            font=dict(color="#0f172a", size=13),
            bgcolor="rgba(255,255,255,0.92)",
        ),
        xaxis=dict(
            showgrid=False,
            title=None,
            tickfont=dict(color=axis_color, size=12),
            linecolor=axis_color,
            linewidth=1,
            mirror=False,
        ),
        yaxis=dict(
            title=dict(text="AQI", font=dict(color=axis_color, size=14)),
            range=[0, y_max],
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(color=axis_color, size=12),
            linecolor=axis_color,
            linewidth=1,
        ),
    )
    return fig


def render_overview_chart(clean: pd.DataFrame, summary: pd.DataFrame) -> None:
    top_provinces = summary.sort_values("peak_aqi", ascending=False).head(6)["province"].tolist()
    chart_df = clean[clean["province"].isin(top_provinces)].copy()
    if go is None:
        fallback = chart_df.pivot_table(index="forecast_time", columns="province_name", values="y_pred", aggfunc="mean")
        st.line_chart(fallback, width="stretch")
        return

    fig = go.Figure()
    for province, group in chart_df.groupby("province"):
        fig.add_trace(go.Scatter(x=group["forecast_time"], y=group["y_pred"], mode="lines", name=str(group["province_name"].iloc[0]), line=dict(width=3), hovertemplate="%{y:.0f} AQI<extra></extra>"))
    y_max = max(220, float(clean["y_pred"].max()) * 1.16)
    render_html("<div class='section-title'>Xu hướng AQI của các điểm nóng</div>")
    st.plotly_chart(style_chart(fig, "", y_max), width="stretch")


def render_province_chart(clean: pd.DataFrame, title: str) -> None:
    if go is None:
        st.line_chart(clean.set_index("forecast_time")[["y_pred"]], width="stretch")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean["forecast_time"], y=clean["y_pred"], mode="lines+markers", line=dict(width=4, color="#2563eb"), marker=dict(size=8, color=clean["level_color"], line=dict(width=1, color="#ffffff")), name="Dự báo AQI", hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.0f} AQI<extra></extra>"))
    y_max = max(220, float(clean["y_pred"].max()) * 1.18)
    render_html(f"<div class='section-title'>{title}</div>")
    st.plotly_chart(style_chart(fig, "", y_max), width="stretch")


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


def raw_air_quality_path(location_key: str, value: pd.Timestamp) -> str:
    return (
        f"air_quality/province={location_key}"
        f"/year={value.year}/month={value.month:02d}/day={value.day:02d}/data.csv"
    )


@st.cache_data(ttl=60)
def load_current_raw_aqi(provinces: tuple[str, ...], target_hour: str) -> pd.DataFrame:
    client = get_client()
    target_time = pd.Timestamp(target_hour)
    frames = []
    dates_to_try = [target_time, target_time - pd.Timedelta(days=1)]

    for province in provinces:
        chosen = None
        for date_value in dates_to_try:
            path = raw_air_quality_path(province, date_value)
            for bucket in MINIO_RAW_BUCKETS:
                df = load_csv_object(client, bucket, path)
                if df is None or df.empty or "time" not in df.columns or "aqi" not in df.columns:
                    continue
                work = df.copy()
                work["aqi"] = pd.to_numeric(work["aqi"], errors="coerce")
                work = work.dropna(subset=["time", "aqi"])
                work = work[work["time"] <= target_time]
                if work.empty:
                    continue
                chosen = work.sort_values("time").iloc[-1]
                break
            if chosen is not None:
                break
        if chosen is None:
            continue
        frames.append(
            {
                "province": province,
                "observed_time": chosen["time"],
                "current_aqi": float(chosen["aqi"]),
            }
        )

    if not frames:
        return pd.DataFrame(columns=["province", "observed_time", "current_aqi"])
    return pd.DataFrame(frames)


def render_current_aqi(provinces: list[str], location_names: dict[str, str], selected_province: str) -> pd.DataFrame:
    target_time = pd.Timestamp(now_local()).floor("h")
    current = load_current_raw_aqi(tuple(provinces), target_time.isoformat())
    if current.empty:
        st.warning("Chưa có dữ liệu AQI đã cào cho giờ hiện tại.")
        return pd.DataFrame()

    current = current.copy()
    current["province_name"] = current["province"].map(lambda p: location_names.get(str(p), str(p)))
    current["current_level"] = current["current_aqi"].map(lambda v: str(aqi_info(v)["vi"]))
    current["current_color"] = current["current_aqi"].map(lambda v: str(aqi_info(v)["color"]))
    current = current.sort_values(["current_aqi", "province_name"], ascending=[False, True])

    selected_rows = current[current["province"] == selected_province]
    current_row = selected_rows.iloc[0] if not selected_rows.empty else current.iloc[0]
    current_time = current_row["observed_time"]
    avg_current = float(current["current_aqi"].mean())
    alert_count = int((current["current_aqi"] > 100).sum())

    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])
    with c1:
        render_status_card("AQI giờ hiện tại", current_row["current_aqi"], current_time, str(current_row["province_name"]))
    with c2:
        render_card("Thời điểm", format_time(current_time, "%d/%m/%Y %H:%M"), "Dữ liệu đã cào gần nhất")
    with c3:
        render_card("AQI trung bình", f"{avg_current:.0f}", str(aqi_info(avg_current)["vi"]))
    with c4:
        render_card("Vượt ngưỡng 100", f"{alert_count}", "Tỉnh/thành cần lưu ý")

    left_col, right_col = st.columns([1.35, 1])
    with left_col:
        render_html("<div class='section-title'>AQI hiện tại theo tỉnh/thành</div>")
        display = current[["province_name", "province", "observed_time", "current_aqi", "current_level"]].copy()
        display["observed_time"] = display["observed_time"].dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "province_name": "Tỉnh/thành",
                "province": "Mã",
                "observed_time": "Thời điểm",
                "current_aqi": st.column_config.NumberColumn("AQI hiện tại", format="%.0f"),
                "current_level": "Mức AQI",
            },
        )
    with right_col:
        render_html("<div class='section-title'>Khu vực cần lưu ý</div>")
        render_risk_list(current, "current_aqi", "observed_time", limit=8)

    return current

def render_overview(all_df: pd.DataFrame, location_names: dict[str, str]) -> pd.DataFrame:
    clean = prepare_clean_predictions(all_df, location_names)
    if clean.empty:
        st.warning("Không tìm thấy dòng dự báo hợp lệ.")
        return pd.DataFrame()
    summary = build_summary(clean, location_names)
    worst_now = summary.sort_values("next_aqi", ascending=False).iloc[0]
    avg_next = float(summary["next_aqi"].mean())
    unhealthy_count = int((summary["next_aqi"] > 100).sum())

    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])
    with c1:
        render_status_card("AQI cao nhất sắp tới", worst_now["next_aqi"], worst_now["next_forecast_time"], str(worst_now["province_name"]))
    with c2:
        render_card("Tỉnh/thành có dự báo", f"{summary['province'].nunique():,}", "Dữ liệu mới nhất theo từng tỉnh/thành")
    with c3:
        render_card("AQI trung bình sắp tới", f"{avg_next:.0f}", str(aqi_info(avg_next)["vi"]))
    with c4:
        render_card("Vượt ngưỡng 100", f"{unhealthy_count}", "Cần lưu ý cho nhóm nhạy cảm")

    render_html("<div class='section-title'>Bảng màu AQI</div>")
    render_aqi_legend()
    chart_col, risk_col = st.columns([2.1, 1])
    with chart_col:
        render_overview_chart(clean, summary)
    with risk_col:
        render_html("<div class='section-title'>Điểm nóng trong kỳ dự báo</div>")
        render_risk_list(summary, "peak_aqi", "peak_time")

    render_html("<div class='section-title'>Tổng quan theo tỉnh/thành</div>")
    display = summary.copy()
    display["next_forecast_time"] = display["next_forecast_time"].dt.strftime("%d/%m/%Y %H:%M")
    display["peak_time"] = display["peak_time"].dt.strftime("%d/%m/%Y %H:%M")
    st.dataframe(
        display.drop(columns=["next_color", "source_path", "run_id"]),
        width="stretch",
        hide_index=True,
        column_config={
            "province_name": "Tỉnh/thành",
            "province": "Mã",
            "forecast_date_partition": "Ngày dự báo",
            "next_forecast_time": "Mốc gần nhất",
            "next_aqi": st.column_config.NumberColumn("AQI gần nhất", format="%.0f"),
            "next_level": "Mức gần nhất",
            "avg_aqi": st.column_config.NumberColumn("AQI TB", format="%.0f"),
            "peak_time": "Thời điểm cao nhất",
            "peak_aqi": st.column_config.NumberColumn("AQI cao nhất", format="%.0f"),
            "peak_level": "Mức cao nhất",
        },
    )
    return summary


def render_prediction(df: pd.DataFrame, metadata: dict, path: str, location_names: dict[str, str]) -> None:
    clean = prepare_clean_predictions(df, location_names)
    if clean.empty:
        st.warning("File dự báo rỗng hoặc thiếu cột forecast_time/y_pred hợp lệ.")
        st.dataframe(df, width="stretch")
        return
    province = str(clean["province"].iloc[0]) if "province" in clean.columns else ""
    province_name = location_names.get(province, province)
    first_row = clean.iloc[0]
    peak_row = clean.loc[clean["y_pred"].idxmax()]
    avg_aqi = float(clean["y_pred"].mean())
    risky_hours = int((clean["y_pred"] > 100).sum())

    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])
    with c1:
        render_status_card("Dự báo gần nhất", first_row["y_pred"], first_row["forecast_time"], province_name)
    with c2:
        render_status_card("Đỉnh AQI", peak_row["y_pred"], peak_row["forecast_time"], "Cao nhất trong kỳ")
    with c3:
        render_card("AQI trung bình", f"{avg_aqi:.0f}", str(aqi_info(avg_aqi)["vi"]))
    with c4:
        render_card("Số mốc vượt 100", f"{risky_hours}", "Khung giờ cần cảnh báo")

    render_province_chart(clean, f"Dự báo AQI - {province_name}")
    timeline = clean[["forecast_time", "y_pred", "level"]].copy()
    timeline["time_label"] = timeline["forecast_time"].dt.strftime("%d/%m %H:%M")
    render_html("<div class='section-title'>Chi tiết theo mốc thời gian</div>")
    st.dataframe(timeline[["time_label", "y_pred", "level"]], width="stretch", hide_index=True, column_config={"time_label": "Thời gian", "y_pred": st.column_config.NumberColumn("AQI dự báo", format="%.0f"), "level": "Mức AQI"})


def main() -> None:
    st.set_page_config(page_title="Dự báo AQI", page_icon="AQI", layout="wide")
    inject_css()

    paths = list_prediction_objects()
    rows = [parse_prediction_path(path) for path in paths]
    location_names = load_location_names()
    provinces = sorted({r["province"] for r in rows if r["province"]})
    all_latest_df = load_latest_predictions(rows, provinces) if provinces else pd.DataFrame()
    summary = build_summary(all_latest_df, location_names) if not all_latest_df.empty else pd.DataFrame()
    render_hero(summary)

    with st.sidebar:
        st.markdown("### Bộ lọc dự báo")
        if st.button("Làm mới dữ liệu", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()

    if not paths or not provinces:
        st.info("Chưa có dữ liệu dự báo để hiển thị. Vui lòng chạy pipeline dự báo trước.")
        return

    with st.sidebar:
        selected_province = st.selectbox("Tỉnh/thành", provinces, format_func=lambda p: f"{location_names.get(p, p)}")
        province_rows = [r for r in rows if r["province"] == selected_province]
        forecast_dates = sorted({r["forecast_date"] for r in province_rows if r["forecast_date"]}, reverse=True)
        selected_date = st.selectbox("Ngày dự báo", forecast_dates)
        date_rows = [r for r in province_rows if r["forecast_date"] == selected_date]
        date_rows.sort(key=lambda r: r["run_id"], reverse=True)
        selected_run_row = date_rows[0]

    current_tab, overview_tab, detail_tab = st.tabs(["AQI hiện tại", "Tổng Quan Dự Đoán", "Chi tiết tỉnh/thành"])
    with current_tab:
        render_current_aqi(provinces, location_names, selected_province)
    with overview_tab:
        render_overview(all_latest_df, location_names)
    with detail_tab:
        selected_path = selected_run_row["path"]
        selected_df = load_prediction_csv(selected_path)
        selected_meta = load_prediction_metadata(selected_path)
        render_prediction(selected_df, selected_meta, selected_path, location_names)


if __name__ == "__main__":
    main()
