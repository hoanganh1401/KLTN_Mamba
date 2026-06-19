"""
Realtime Air Quality inference DAG.

Runs the near-real-time path:
Bronze ingestion -> hourly validation -> Silver processing -> Gold features
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
LOOKBACK_DAYS = "{{ var.value.get('AQI_PIPELINE_LOOKBACK_DAYS', var.value.get('AQI_INGEST_LOOKBACK_DAYS', '30')) }}"

SCRAPER_SCRIPT = f"{DATASET_DIR}/data_scraper.py"
VALIDATION_SCRIPT = f"{SRC_DIR}/Silver/Data_Validation.py"
SILVER_PROCESSING_SCRIPT = f"{SRC_DIR}/Silver/silver_processing.py"
GOLD_FEATURE_SCRIPT = f"{SRC_DIR}/Gold/gold_feature_engineering.py"
PREPARE_INFERENCE_SCRIPT = f"{SRC_DIR}/Gold/prepare_inference_input.py"

PRODUCTION_RUN_PREFIX = "{{ var.value.get('AQI_PRODUCTION_RUN_PREFIX', 'manual_latest') }}"


def load_location_keys(path: str) -> list[str]:
    """Read province keys at DAG-parse time to create one inference chain per province."""
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="air_quality_hourly",
    description="Hourly Bronze ingestion, Silver/Gold refresh, and Mamba inference",
    default_args=default_args,
    schedule="@hourly",
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["air-quality", "hourly", "bronze", "silver", "gold", "inference"],
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

    if not LOCATION_KEYS:
        raise FileNotFoundError(f"No location keys found at {LOCATIONS_PATH}")

    previous_tail = build_gold_features
    with TaskGroup(group_id="infer_each_province") as per_province:
        for location_key in LOCATION_KEYS:
            suffix = task_suffix(location_key)
            production_run_id = f"{PRODUCTION_RUN_PREFIX}__{location_key}"
            model_run_id = f"mamba_{production_run_id}"
            inference_run_id = f"infer_{{{{ ts_nodash }}}}__{location_key}"

            prepare_inference_input = BashOperator(
                task_id=f"prepare_inference_input_{suffix}",
                bash_command=(
                    f"python {PREPARE_INFERENCE_SCRIPT} "
                    f"--locations {LOCATIONS_PATH} "
                    f"--location-keys {location_key} "
                    f"--config {PROJECT_CONFIG} "
                    "--end-date {{ ds }} "
                    "--lookback-days {{ var.value.get('AQI_INFERENCE_LOOKBACK_DAYS', '14') }} "
                    f"--run-id {production_run_id} "
                    f"--output-run-id {inference_run_id}"
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
                        "inference_run_id": inference_run_id,
                        "model_run_id": model_run_id,
                        "artifact_run_id": inference_run_id,
                    }
                ),
                headers={"Content-Type": "application/json"},
                log_response=True,
                extra_options={"timeout": 3600},
                execution_timeout=timedelta(hours=1),
            )

            previous_tail >> prepare_inference_input >> run_inference
            previous_tail = run_inference

    ingest_incremental >> validate_hourly >> process_silver >> build_gold_features >> per_province
