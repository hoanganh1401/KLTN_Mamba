"""
Daily Air Quality training and inference DAG.

Runs the completed-day path:
daily validation -> Gold features -> training dataset -> Mamba training
-> inference input -> prediction artifacts.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/Project"
DATASET_DIR = f"{PROJECT_ROOT}/Dataset"
SRC_DIR = f"{PROJECT_ROOT}/src"
AIRFLOW_TMP_DIR = "/opt/airflow/aqi_artifacts"

LOCATIONS_PATH = f"{DATASET_DIR}/locations.jsonl"
PROJECT_CONFIG = f"{PROJECT_ROOT}/Conf/air_quality.yaml"

VALIDATION_SCRIPT = f"{SRC_DIR}/Silver/Data_Validation.py"
GOLD_FEATURE_SCRIPT = f"{SRC_DIR}/Gold/gold_feature_engineering.py"
PREPARE_TRAINING_SCRIPT = f"{SRC_DIR}/Gold/prepare_training_dataset.py"
TRAIN_MAMBA_SCRIPT = f"{SRC_DIR}/Model/train_mamba_aqi.py"
PREPARE_INFERENCE_SCRIPT = f"{SRC_DIR}/Gold/prepare_inference_input.py"
RUN_INFERENCE_SCRIPT = f"{SRC_DIR}/Inference/run_mamba_inference.py"

RUN_ID = "daily_{{ ds_nodash }}"
MODEL_RUN_ID = f"mamba_{RUN_ID}"
MODEL_OUT_DIR = f"{AIRFLOW_TMP_DIR}/{RUN_ID}/model"

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
    description="Daily validation, training dataset build, Mamba training, and inference",
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

    prepare_training_dataset = BashOperator(
        task_id="prepare_training_dataset",
        bash_command=(
            f"python {PREPARE_TRAINING_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            f"--config {PROJECT_CONFIG} "
            "--end-date {{ ds }} "
            "--days {{ var.value.get('AQI_TRAIN_LOOKBACK_DAYS', '180') }} "
            f"--run-id {RUN_ID}"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=1),
    )

    train_mamba = BashOperator(
        task_id="train_mamba",
        bash_command=(
            f"mkdir -p {MODEL_OUT_DIR} && "
            f"python {TRAIN_MAMBA_SCRIPT} "
            f"--config {PROJECT_CONFIG} "
            f"--run-id {RUN_ID} "
            f"--model-run-id {MODEL_RUN_ID} "
            f"--out-dir {MODEL_OUT_DIR} "
            "--keep-local"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=6),
    )

    prepare_inference_input = BashOperator(
        task_id="prepare_inference_input",
        bash_command=(
            f"python {PREPARE_INFERENCE_SCRIPT} "
            f"--locations {LOCATIONS_PATH} "
            f"--config {PROJECT_CONFIG} "
            "--end-date {{ ds }} "
            "--lookback-days {{ var.value.get('AQI_INFERENCE_LOOKBACK_DAYS', '14') }} "
            f"--run-id {RUN_ID} "
            f"--output-run-id {RUN_ID}"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(minutes=45),
    )

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=(
            f"python {RUN_INFERENCE_SCRIPT} "
            f"--inference-run-id {RUN_ID} "
            f"--checkpoint {MODEL_OUT_DIR}/best_mamba_aqi.pt "
            f"--metadata {MODEL_OUT_DIR}/training_metadata.json "
            f"--artifact-run-id {RUN_ID}"
        ),
        env=COMMON_ENV,
        execution_timeout=timedelta(hours=1),
    )

    (
        validate_daily
        >> refresh_gold_features
        >> prepare_training_dataset
        >> train_mamba
        >> prepare_inference_input
        >> run_inference
    )
