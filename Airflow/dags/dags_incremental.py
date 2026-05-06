"""
Airflow DAG – Air Quality Bronze Ingestion
==========================================
Chạy hourly. Khi máy bị tắt và khởi động lại:
  - catchup=False     → Airflow KHÔNG tạo hàng nghìn DagRun backlog
  - gap detection     → bronze_raw.py tự scan MinIO và backfill dữ liệu thực tế còn thiếu
  - max_active_runs=1 → tránh nhiều run chạy song song tranh nhau API

Tại sao KHÔNG dùng catchup=True?
  @hourly từ 2025-01-01 → Airflow tạo ~12.000+ DagRun ngay lập tức,
  scheduler bị nghẹt queue → DAG không hiện trên UI hoặc chạy rất chậm.
  Gap detection trong bronze_raw.py đã xử lý việc bù dữ liệu thiếu tốt hơn.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
# Điều chỉnh cho phù hợp với mount path trong Docker của bạn
SCRIPT_PATH    = "/opt/Project/Dataset/data_scraper.py"
LOCATIONS_PATH = "/opt/Project/Dataset/locations.jsonl"

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "airflow",
    "depends_on_past":  False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="air_quality_bronze_ingestion",
    description="Hourly air quality ingestion with auto gap-fill on restart",
    default_args=default_args,
    schedule="@hourly",

    # Đặt start_date = ngày bắt đầu thực tế gần đây
    # KHÔNG đặt quá xa trong quá khứ khi catchup=False
    start_date=datetime(2025, 4, 1),

    # ✅ False để tránh scheduler bị nghẹt do backlog quá lớn
    # Việc bù dữ liệu thiếu do bronze_raw.py gap detection xử lý
    catchup=False,

    max_active_runs=1,

    tags=["bronze", "air-quality", "open-meteo"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_incremental",
        bash_command=(
            f"python {SCRIPT_PATH} "
            "--mode incremental "
            f"--locations {LOCATIONS_PATH} "
            "--lookback-days 30 "
        ),
        env={
            "MINIO_HOST":       "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
            "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
            "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
            "MINIO_BUCKET":     "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
            "MINIO_SECURE":     "{{ var.value.get('MINIO_SECURE', 'false') }}",
        },
        execution_timeout=timedelta(minutes=30),
    )