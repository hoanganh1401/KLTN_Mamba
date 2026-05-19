# KLTN_Mamba

## Ubuntu (WSL) setup

See [UBUNTU_SETUP.md](UBUNTU_SETUP.md) for the recommended CUDA + Mamba training setup.

## Run guide (A to Z)

### 0) Start Docker services (MinIO + Airflow)

```bash
cd Tool
docker compose up -d
```

Check:
- Airflow UI: http://localhost:8081
- MinIO UI: http://localhost:9005 (user: admin, pass: admin123)

### 1) (Optional) WSL Ubuntu environment

If you run on Windows and want faster Mamba training, use WSL Ubuntu:
- See [UBUNTU_SETUP.md](UBUNTU_SETUP.md)

### 2) Silver processing

```bash
python src/Silver/silver_processing.py \
	--locations DataSet/locations.jsonl \
	--date 2026-05-16
```

### 3) Gold feature engineering

```bash
python src/Gold/gold_feature_engineering.py \
	--locations DataSet/locations.jsonl \
	--date 2026-05-16
```

### 4) Prepare training dataset (Gold -> train/val/test)

```bash
python src/Gold/prepare_training_dataset.py \
	--locations DataSet/locations.jsonl \
	--start-date 2026-05-01 \
	--end-date 2026-05-16 \
	--seq-len 96 \
	--pred-len 12 \
	--train-ratio 0.7 \
	--val-ratio 0.1
```

Outputs are written to `artifacts/datasets/<timestamp>/` and include:
- X_train.npy, y_train.npy, loc_ids_train.npy
- X_val.npy, y_val.npy, loc_ids_val.npy
- X_test.npy, y_test.npy, loc_ids_test.npy
- scaler.pkl, dataset_metadata.json

### 5) Train Mamba (CSV-based example)

```bash
python src/mamba/train_mamba_aqi.py \
	--data-path dataset/air_quality.csv \
	--epochs 10 \
	--window-size 72 \
	--horizon 12 \
	--batch-size 128 \
	--device cuda \
	--amp
```

### 6) Prepare inference input

```bash
python src/Gold/prepare_inference_input.py \
	--locations DataSet/locations.jsonl \
	--seq-len 96 \
	--pred-len 12 \
	--dataset-dir artifacts/datasets/<timestamp>
```

### 7) Predict next AQI steps

```bash
python src/Inference/predict_aqi_next.py \
	--model-path runs/<run_id>/best_mamba_aqi.pt \
	--inference-dir artifacts/inference_input \
	--dataset-dir artifacts/datasets/<timestamp> \
	--horizon 12 \
	--device cuda
```

## Notes

- If you use MinIO, pass `--minio-prefix` in the Gold steps to upload outputs.
- For Windows + CUDA issues, prefer the WSL setup in [UBUNTU_SETUP.md](UBUNTU_SETUP.md).