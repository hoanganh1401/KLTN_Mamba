"""
Hourly Air Quality DAG.

Runs the near-real-time path:
Bronze ingestion -> hourly validation -> Silver processing -> Gold features.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/Project"
DATASET_DIR = f"{PROJECT_ROOT}/Dataset"
SRC_DIR = f"{PROJECT_ROOT}/src"

LOCATIONS_PATH = f"{DATASET_DIR}/locations.jsonl"
PROJECT_CONFIG = f"{PROJECT_ROOT}/Conf/air_quality.yaml"
LOOKBACK_DAYS = "{{ var.value.get('AQI_PIPELINE_LOOKBACK_DAYS', var.value.get('AQI_INGEST_LOOKBACK_DAYS', '30')) }}"

SCRAPER_SCRIPT = f"{DATASET_DIR}/data_scraper.py"
VALIDATION_SCRIPT = f"{SRC_DIR}/Silver/Data_Validation.py"
SILVER_PROCESSING_SCRIPT = f"{SRC_DIR}/Silver/silver_processing.py"
GOLD_FEATURE_SCRIPT = f"{SRC_DIR}/Gold/gold_feature_engineering.py"

COMMON_ENV = {
    "PYTHONPATH": f"{PROJECT_ROOT}:{SRC_DIR}",
    "MINIO_HOST": "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
    "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
    "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
    "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
    "MINIO_SILVER_BUCKET": "{{ var.value.get('MINIO_SILVER_BUCKET', 'air-quality-silver') }}",
    "MINIO_GOLD_BUCKET": "{{ var.value.get('MINIO_GOLD_BUCKET', 'air-quality-gold') }}",
    "MINIO_ARTIFACTS_BUCKET": "{{ var.value.get('MINIO_ARTIFACTS_BUCKET', 'air-quality-artifacts') }}",
    "MINIO_SECURE": "{{ var.value.get('MINIO_SECURE', 'false') }}",
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="air_quality_hourly",
    description="Hourly Bronze ingestion, Silver processing, and Gold feature refresh",
    default_args=default_args,
    schedule="@hourly",
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["air-quality", "hourly", "bronze", "silver", "gold"],
) as dag:
    ingest_incremental = BashOperator(
        task_id="ingest_incremental",
        bash_command=(
            f"python {SCRAPER_SCRIPT} "
            "--mode incremental "
            f"--locations {LOCATIONS_PATH} "
            f"--lookback-days {LOOKBACK_DAYS}"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(minutes=45),
    )

    validate_hourly = BashOperator(
        task_id="validate_hourly_lookback",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"base=datetime.strptime('{{{{ ds }}}}', '%Y-%m-%d').date(); "
            f"days=int('{LOOKBACK_DAYS}'); "
            "print(' '.join((base - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)))\"); do "
            "echo \"Validating hourly data for ${target_date}\"; "
            f"python {VALIDATION_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date ${target_date} "
            "--mode hourly; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=2),
    )

    process_silver = BashOperator(
        task_id="process_silver_lookback",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"base=datetime.strptime('{{{{ ds }}}}', '%Y-%m-%d').date(); "
            f"days=int('{LOOKBACK_DAYS}'); "
            "print(' '.join((base - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)))\"); do "
            "echo \"Processing Silver data for ${target_date}\"; "
            f"python {SILVER_PROCESSING_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date ${target_date}; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=2),
    )

    build_gold_features = BashOperator(
        task_id="build_gold_features_lookback",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"base=datetime.strptime('{{{{ ds }}}}', '%Y-%m-%d').date(); "
            f"days=int('{LOOKBACK_DAYS}'); "
            "print(' '.join((base - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)))\"); do "
            "echo \"Building Gold features for ${target_date}\"; "
            f"python {GOLD_FEATURE_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            f"--config {PROJECT_CONFIG} "
            "--date ${target_date}; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=2),
    )

    ingest_incremental >> validate_hourly >> process_silver >> build_gold_features
