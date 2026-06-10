"""Optional MLflow helpers for AQI model training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_mlflow(logger=None):
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
    try:
        import mlflow
    except Exception as exc:
        if logger:
            logger.warning("MLflow is not available; tracking is skipped: %s", exc)
        return None
    return mlflow


def start_mlflow_run(
    *,
    logger,
    experiment_name: str,
    run_name: str,
    tracking_uri: str | None = None,
    tags: dict[str, Any] | None = None,
):
    mlflow = get_mlflow(logger)
    if mlflow is None:
        return None, None
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is not None and getattr(experiment, "lifecycle_stage", None) == "deleted":
                try:
                    client.restore_experiment(experiment.experiment_id)
                    logger.info("Restored deleted MLflow experiment: %s", experiment_name)
                except Exception as restore_exc:
                    fallback_name = f"{experiment_name} Active"
                    logger.warning(
                        "Could not restore deleted MLflow experiment '%s': %s. Using '%s' instead.",
                        experiment_name,
                        restore_exc,
                        fallback_name,
                    )
                    experiment_name = fallback_name
        except Exception as lookup_exc:
            logger.warning("Could not inspect MLflow experiment '%s': %s", experiment_name, lookup_exc)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
        logger.info("MLflow run started: experiment=%s | run_name=%s", experiment_name, run_name)
        return mlflow, run
    except Exception as exc:
        logger.warning("MLflow tracking is skipped because the tracking server is not ready: %s", exc)
        return None, None


def log_artifact_dir(mlflow, out_dir: str | Path, logger=None) -> None:
    if mlflow is None:
        return
    out_path = Path(out_dir)
    if not out_path.exists():
        return
    try:
        mlflow.log_artifacts(str(out_path))
    except Exception as exc:
        if logger:
            logger.warning("Could not log MLflow artifacts from %s: %s", out_path, exc)
