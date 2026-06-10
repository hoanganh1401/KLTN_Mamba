"""
Daily Air Quality training and inference DAG.

Runs the completed-day path:
daily validation -> Gold features -> training dataset -> Mamba API training
-> inference input -> Mamba API prediction artifacts.
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
GOLD_FEATURE_SCRIPT = f"{SRC_DIR}/Gold/gold_feature_engineering.py"
PREPARE_TRAINING_SCRIPT = f"{SRC_DIR}/Gold/prepare_training_dataset.py"
PREPARE_INFERENCE_SCRIPT = f"{SRC_DIR}/Gold/prepare_inference_input.py"

RUN_ID = "daily_{{ ds_nodash }}"


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
    dag_id="air_quality_daily_training",
    description="Daily validation, dataset build, Mamba API training, and Mamba API inference",
    default_args=default_args,
    schedule="0 1 * * *",
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["air-quality", "daily", "silver", "gold", "mamba", "inference"],
) as dag:
    validate_daily = BashOperator(
        task_id="validate_daily",
        bash_command=(
            f"python {VALIDATION_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            "--date {{ ds }} "
            "--mode daily"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(minutes=45),
    )

    refresh_gold_features = BashOperator(
        task_id="refresh_gold_features",
        bash_command=(
            f"python {GOLD_FEATURE_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            f"--config {PROJECT_CONFIG} "
            "--date {{ ds }}"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(minutes=45),
    )

    if not LOCATION_KEYS:
        raise FileNotFoundError(f"No location keys found at {LOCATIONS_PATH}")

    previous_tail = refresh_gold_features
    with TaskGroup(group_id="train_infer_each_province") as per_province:
        for location_key in LOCATION_KEYS:
            suffix = task_suffix(location_key)
            province_run_id = f"{RUN_ID}__{location_key}"
            model_run_id = f"mamba_{province_run_id}"

            prepare_training_dataset = BashOperator(
                task_id=f"prepare_training_dataset_{suffix}",
                bash_command=(
                    f"python {PREPARE_TRAINING_SCRIPT} "
                    f"--locations {LOCATIONS_PATH} "
                    f"--location-keys {location_key} "
                    f"--config {PROJECT_CONFIG} "
                    "--start-date {{ var.value.get('AQI_TRAIN_START_DATE', '2025-01-01') }} "
                    "--end-date {{ ds }} "
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

            prepare_inference_input = BashOperator(
                task_id=f"prepare_inference_input_{suffix}",
                bash_command=(
                    f"python {PREPARE_INFERENCE_SCRIPT} "
                    f"--locations {LOCATIONS_PATH} "
                    f"--location-keys {location_key} "
                    f"--config {PROJECT_CONFIG} "
                    "--end-date {{ ds }} "
                    "--lookback-days {{ var.value.get('AQI_INFERENCE_LOOKBACK_DAYS', '14') }} "
                    f"--run-id {province_run_id} "
                    f"--output-run-id {province_run_id}"
                ),
                env=COMMON_ENV,
                execution_timeout=timedelta(minutes=45),
            )

            run_inference = HttpOperator(
                task_id=f"run_inference_{suffix}",
                http_conn_id="mamba_api",
                endpoint="/inference",
                method="POST",
                data=json.dumps(
                    {
                        "inference_run_id": province_run_id,
                        "model_run_id": model_run_id,
                        "artifact_run_id": province_run_id,
                    }
                ),
                headers={"Content-Type": "application/json"},
                log_response=True,
                extra_options={"timeout": 3600},
                execution_timeout=timedelta(hours=1),
            )

            previous_tail >> prepare_training_dataset
            (
                prepare_training_dataset
                >> train_mamba
                >> prepare_inference_input
                >> run_inference
            )
            previous_tail = run_inference

    validate_daily >> refresh_gold_features >> per_province
