"""
Airflow DAGs — Air Quality Pipeline
=====================================

DAG 1: air_quality_hourly  — @hourly
    ├── ingest_incremental      (data_scraper.py)
    ├── validate_hourly         (data_validation.py --mode hourly)
    │       R1 Schema, R2 Duplicate, R4 AQI range, R5 Physical bounds
    └── process_silver          (data_processing.py)
            Clean + normalize + time features (intra-day)

DAG 2: air_quality_daily   — 01:00 UTC hàng ngày
    ├── validate_daily          (data_validation.py --mode daily)
    │       R3 Timestamp gap, R6 Missing rate, R7 Coverage >= 90%
    │       Đọc Silver ngày hôm qua, ghi đè với daily flags
    └── feature_engineering / train_mamba / evaluate / save_model
            [chưa viết — sẽ thêm vào sau]

Luồng dữ liệu:
  Bronze (MinIO/air-quality)
    ↓ ingest           [@hourly]
    ↓ validate_hourly  [@hourly] → Silver + hourly flags
    ↓ process_silver   [@hourly] → Silver cleaned + normalized
    ↓ validate_daily   [01:00]   → Silver + daily completeness flags
    ↓ feature_engineering → train → evaluate → save_model  [01:00]
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_PATH    = "/opt/Project/Dataset"
LOCATIONS_PATH = "/opt/Project/Dataset/locations.jsonl"

SCRAPER_SCRIPT     = f"{SCRIPT_PATH}/data_scraper.py"
VALIDATION_SCRIPT = f"{SCRIPT_PATH}/Data_Validation.py"
PROCESSING_SCRIPT = f"{SCRIPT_PATH}/Data_processing.py"

# ── Common MinIO env ───────────────────────────────────────────────────────────
MINIO_ENV = {
    "MINIO_HOST":          "{{ var.value.get('MINIO_HOST',          'minio:9000') }}",
    "MINIO_ACCESS_KEY":    "{{ var.value.get('MINIO_ACCESS_KEY',    'admin') }}",
    "MINIO_SECRET_KEY":    "{{ var.value.get('MINIO_SECRET_KEY',    'admin123') }}",
    "MINIO_BUCKET":        "{{ var.value.get('MINIO_BUCKET',        'air-quality') }}",
    "MINIO_SILVER_BUCKET": "{{ var.value.get('MINIO_SILVER_BUCKET', 'air-quality-silver') }}",
    "MINIO_SECURE":        "{{ var.value.get('MINIO_SECURE',        'false') }}",
}

default_args = {
    "owner":            "airflow",
    "depends_on_past":  False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# DAG 1 — Hourly: Ingest → Validate → Process
# ══════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="air_quality_hourly",
    description="Hourly: ingest + batch validation + processing",
    default_args=default_args,
    schedule="@hourly",
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bronze", "silver", "air-quality", "hourly"],
) as dag_hourly:

    ingest = BashOperator(
        task_id="ingest_incremental",
        bash_command=(
            f"python {SCRAPER_SCRIPT} "
            "--mode incremental "
            f"--locations {LOCATIONS_PATH} "
            "--lookback-days 30"
        ),
        env=MINIO_ENV,
        execution_timeout=timedelta(minutes=30),
    )

    # Batch quality check ngay sau ingest
    # Checks: R1 schema, R2 duplicate, R4 AQI range, R5 physical bounds
    # Không check: R3 gap, R6 missing rate, R7 coverage (ngày chưa xong)
    validate_hourly = BashOperator(
        task_id="validate_hourly",
        bash_command=(
            f"python {VALIDATION_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date {{ ds }} "
            "--mode hourly"
        ),
        env=MINIO_ENV,
        execution_timeout=timedelta(minutes=20),
    )

    # Process ngay sau validate để Silver luôn được cập nhật mỗi giờ
    process = BashOperator(
        task_id="process_silver",
        bash_command=(
            f"python {PROCESSING_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date {{ ds }}"
        ),
        env=MINIO_ENV,
        execution_timeout=timedelta(minutes=30),
    )

    ingest >> validate_hourly >> process


# ══════════════════════════════════════════════════════════════════════════════
# DAG 2 — Daily: Validate completeness + ML Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="air_quality_daily",
    description="Daily: completeness validation + ML pipeline",
    default_args=default_args,
    schedule="0 1 * * *",      # 01:00 UTC — data ngày hôm qua đã đủ 24h
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["silver", "gold", "air-quality", "daily", "ml"],
) as dag_daily:

    # {{ ds }} = ngày hôm qua (01:00 ngày 9/5 → ds = "2025-05-08")
    # Checks: R3 continuity, R6 missing rate, R7 coverage >= 90%
    validate_daily = BashOperator(
        task_id="validate_daily",
        bash_command=(
            f"python {VALIDATION_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date {{ ds }} "
            "--mode daily"
        ),
        env=MINIO_ENV,
        execution_timeout=timedelta(minutes=30),
    )

    # feature_engineering, train_mamba, evaluate, save_model
    # sẽ được thêm vào sau khi implement xong