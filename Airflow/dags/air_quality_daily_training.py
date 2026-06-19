"""
Manual Air Quality training DAG.

Runs the offline training path on demand:
daily validation -> Silver processing -> Gold features -> training dataset
-> Mamba API training.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = "/opt/Project"
DATASET_DIR = f"{PROJECT_ROOT}/Dataset"
SRC_DIR = f"{PROJECT_ROOT}/src"

LOCATIONS_PATH = f"{DATASET_DIR}/locations.jsonl"
PROJECT_CONFIG = f"{PROJECT_ROOT}/Conf/air_quality.yaml"
MAMBA_CONFIG = "/workspace/KLTN_Mamba/Conf/air_quality.yaml"

VALIDATION_SCRIPT = f"{SRC_DIR}/Silver/Data_Validation.py"
SILVER_PROCESSING_SCRIPT = f"{SRC_DIR}/Silver/silver_processing.py"
GOLD_FEATURE_SCRIPT = f"{SRC_DIR}/Gold/gold_feature_engineering.py"
PREPARE_TRAINING_SCRIPT = f"{SRC_DIR}/Gold/prepare_training_dataset.py"

PRODUCTION_RUN_PREFIX = "{{ var.value.get('AQI_PRODUCTION_RUN_PREFIX', 'manual_latest') }}"
TRAIN_START_DATE = "{{ var.value.get('AQI_TRAIN_START_DATE', '2025-01-01') }}"
TRAIN_END_DATE = "{{ var.value.get('AQI_TRAIN_END_DATE', macros.ds_add(ds, -1)) }}"


def load_location_keys(path: str) -> list[str]:
    """Read province keys at DAG-parse time to create one training chain per province."""
    dag_project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(path),
        dag_project_root / "DataSet" / "locations.jsonl",
        dag_project_root / "Dataset" / "locations.jsonl",
    ]
    location_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if location_path is None:
        return []

    keys: list[str] = []
    with location_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("location_key")
            if key:
                keys.append(str(key))
    return keys


def task_suffix(location_key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", location_key).strip("_").lower()


LOCATION_KEYS = load_location_keys(LOCATIONS_PATH)

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
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="air_quality_manual_training",
    description="Manual daily validation, dataset build, and Mamba API training",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["air-quality", "manual", "daily", "silver", "gold", "mamba", "training"],
) as dag:
    validate_daily_range = BashOperator(
        task_id="validate_daily_range",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"start=datetime.strptime('{TRAIN_START_DATE}', '%Y-%m-%d').date(); "
            f"end=datetime.strptime('{TRAIN_END_DATE}', '%Y-%m-%d').date(); "
            "print(' '.join((start + timedelta(days=i)).isoformat() for i in range((end-start).days + 1)))\"); do "
            "echo \"Running daily validation for ${target_date}\"; "
            f"python {VALIDATION_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date ${target_date} "
            "--mode daily; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=4),
    )

    reprocess_silver_range = BashOperator(
        task_id="reprocess_silver_range",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"start=datetime.strptime('{TRAIN_START_DATE}', '%Y-%m-%d').date(); "
            f"end=datetime.strptime('{TRAIN_END_DATE}', '%Y-%m-%d').date(); "
            "print(' '.join((start + timedelta(days=i)).isoformat() for i in range((end-start).days + 1)))\"); do "
            "echo \"Reprocessing Silver data for ${target_date}\"; "
            f"python {SILVER_PROCESSING_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date ${target_date}; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=4),
    )

    rebuild_gold_features_range = BashOperator(
        task_id="rebuild_gold_features_range",
        bash_command=(
            "set -euo pipefail; "
            f"for target_date in $(python -c \"from datetime import datetime, timedelta; "
            f"start=datetime.strptime('{TRAIN_START_DATE}', '%Y-%m-%d').date(); "
            f"end=datetime.strptime('{TRAIN_END_DATE}', '%Y-%m-%d').date(); "
            "print(' '.join((start + timedelta(days=i)).isoformat() for i in range((end-start).days + 1)))\"); do "
            "echo \"Rebuilding Gold features for ${target_date}\"; "
            f"python {GOLD_FEATURE_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            f"--config {PROJECT_CONFIG} "
            "--date ${target_date}; "
            "done"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=4),
    )

    if not LOCATION_KEYS:
        raise FileNotFoundError(f"No location keys found at {LOCATIONS_PATH}")

    previous_tail = rebuild_gold_features_range
    with TaskGroup(group_id="train_each_province") as per_province:
        for location_key in LOCATION_KEYS:
            suffix = task_suffix(location_key)
            province_run_id = f"{PRODUCTION_RUN_PREFIX}__{location_key}"
            model_run_id = f"mamba_{province_run_id}"

            prepare_training_dataset = BashOperator(
                task_id=f"prepare_training_dataset_{suffix}",
                bash_command=(
                    f"python {PREPARE_TRAINING_SCRIPT} "
                    f"--locations {LOCATIONS_PATH} "
                    f"--location-keys {location_key} "
                    f"--config {PROJECT_CONFIG} "
                    f"--start-date {TRAIN_START_DATE} "
                    f"--end-date {TRAIN_END_DATE} "
                    f"--run-id {province_run_id}"
                ),
                env=COMMON_ENV,
                execution_timeout=timedelta(hours=1),
            )

            train_mamba = HttpOperator(
                task_id=f"train_mamba_{suffix}",
                http_conn_id="mamba_api",
                endpoint="/train",
                method="POST",
                data=json.dumps(
                    {
                        "run_id": province_run_id,
                        "model_run_id": model_run_id,
                        "config": MAMBA_CONFIG,
                        "keep_local": True,
                    }
                ),
                headers={"Content-Type": "application/json"},
                log_response=True,
                extra_options={"timeout": 21600},
                execution_timeout=timedelta(hours=6),
            )

            previous_tail >> prepare_training_dataset
            prepare_training_dataset >> train_mamba
            previous_tail = train_mamba

    validate_daily_range >> reprocess_silver_range >> rebuild_gold_features_range >> per_province
