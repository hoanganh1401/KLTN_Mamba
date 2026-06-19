"""
Manual benchmark training DAG.

Trains LSTM + Transformer models for each province on demand using the same
prepared training dataset artifacts that the production Mamba run uses. This
DAG does not run inference and does not replace the production Mamba checkpoint.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = "/opt/Project"
DATASET_DIR = f"{PROJECT_ROOT}/Dataset"

LOCATIONS_PATH = f"{DATASET_DIR}/locations.jsonl"
MODEL_CONFIG = "/workspace/KLTN_Mamba/Conf/air_quality.yaml"

PRODUCTION_RUN_PREFIX = "{{ var.value.get('AQI_PRODUCTION_RUN_PREFIX', 'manual_latest') }}"


def load_location_keys(path: str) -> list[str]:
    """Read province keys at DAG-parse time to create one benchmark chain per province."""
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

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="air_quality_benchmark_training",
    description="Manual LSTM and Transformer benchmark training",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["air-quality", "manual", "benchmark", "lstm", "transformer", "training"],
) as dag:
    if not LOCATION_KEYS:
        raise FileNotFoundError(f"No location keys found at {LOCATIONS_PATH}")

    previous_tail = None
    with TaskGroup(group_id="benchmark_each_province") as per_province:
        for location_key in LOCATION_KEYS:
            suffix = task_suffix(location_key)
            province_run_id = f"{PRODUCTION_RUN_PREFIX}__{location_key}"

            train_lstm = HttpOperator(
                task_id=f"train_lstm_{suffix}",
                http_conn_id="mamba_api",
                endpoint="/train",
                method="POST",
                data=json.dumps(
                    {
                        "run_id": province_run_id,
                        "model_run_id": f"lstm_{province_run_id}",
                        "model_type": "lstm",
                        "config": MODEL_CONFIG,
                        "keep_local": True,
                    }
                ),
                headers={"Content-Type": "application/json"},
                log_response=True,
                extra_options={"timeout": 21600},
                execution_timeout=timedelta(hours=6),
            )

            train_transformer = HttpOperator(
                task_id=f"train_transformer_{suffix}",
                http_conn_id="mamba_api",
                endpoint="/train",
                method="POST",
                data=json.dumps(
                    {
                        "run_id": province_run_id,
                        "model_run_id": f"transformer_{province_run_id}",
                        "model_type": "transformer",
                        "config": MODEL_CONFIG,
                        "keep_local": True,
                    }
                ),
                headers={"Content-Type": "application/json"},
                log_response=True,
                extra_options={"timeout": 21600},
                execution_timeout=timedelta(hours=6),
            )

            train_lstm >> train_transformer
            if previous_tail is not None:
                previous_tail >> train_lstm
            previous_tail = train_transformer
