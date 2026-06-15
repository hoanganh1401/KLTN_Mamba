from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path("/workspace/KLTN_Mamba")
CONFIG_PATH = PROJECT_ROOT / "Conf" / "air_quality.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "Model" / "Mamba" / "train_mamba_aqi.py"
INFERENCE_SCRIPT = PROJECT_ROOT / "src" / "Inference" / "run_mamba_inference.py"
API_RUNS_DIR = PROJECT_ROOT / "runs" / "mamba_api"

app = FastAPI(title="AQI Mamba API")


class TrainRequest(BaseModel):
    run_id: str = Field(..., description="Training dataset run id, e.g. daily_20260608")
    model_run_id: str | None = Field(None, description="Model artifact run id")
    config: str = Field(str(CONFIG_PATH), description="Project YAML config path")
    keep_local: bool = Field(True, description="Keep checkpoint locally for follow-up inference")


class InferenceRequest(BaseModel):
    inference_run_id: str = Field(..., description="Inference input run id")
    model_run_id: str = Field(..., description="Model run id created by /train")
    artifact_run_id: str | None = Field(None, description="Prediction artifact run id")
    province: str | None = Field(None, description="Province key for loading model artifacts from MinIO")


def _run_command(cmd: list[str], timeout: int | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
    }
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=payload)
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model(req: TrainRequest) -> dict[str, Any]:
    model_run_id = req.model_run_id or f"mamba_{req.run_id}"
    out_dir = API_RUNS_DIR / model_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(TRAIN_SCRIPT),
        "--config",
        req.config,
        "--run-id",
        req.run_id,
        "--model-run-id",
        model_run_id,
        "--out-dir",
        str(out_dir),
    ]
    if req.keep_local:
        cmd.append("--keep-local")

    result = _run_command(cmd)
    result.update(
        {
            "run_id": req.run_id,
            "model_run_id": model_run_id,
            "out_dir": str(out_dir),
            "checkpoint": str(out_dir / "best_mamba_aqi.pt"),
            "metadata": str(out_dir / "training_metadata.json"),
        }
    )
    return result


@app.post("/inference")
def run_inference(req: InferenceRequest) -> dict[str, Any]:
    artifact_run_id = req.artifact_run_id or req.inference_run_id
    model_dir = API_RUNS_DIR / req.model_run_id
    checkpoint = model_dir / "best_mamba_aqi.pt"
    metadata = model_dir / "training_metadata.json"

    if checkpoint.exists() and metadata.exists():
        model_args = [
            "--checkpoint",
            str(checkpoint),
            "--metadata",
            str(metadata),
        ]
    elif req.province:
        model_args = [
            "--province",
            req.province,
            "--model-run-id",
            req.model_run_id,
        ]
    else:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Local checkpoint or metadata not found. Run /train first, keep local artifacts, or provide province for MinIO fallback.",
                "checkpoint": str(checkpoint),
                "metadata": str(metadata),
            },
        )

    cmd = [
        "python3",
        str(INFERENCE_SCRIPT),
        "--inference-run-id",
        req.inference_run_id,
        *model_args,
        "--artifact-run-id",
        artifact_run_id,
    ]
    result = _run_command(cmd)
    result.update(
        {
            "inference_run_id": req.inference_run_id,
            "model_run_id": req.model_run_id,
            "artifact_run_id": artifact_run_id,
            "province": req.province,
        }
    )
    return result
