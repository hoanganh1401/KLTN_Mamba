# Huong dan chay pipeline Mamba AQI

Tai lieu nay dung cho project `D:\KLTN\KLTN_Mamba` sau khi code da chuyen sang luong:

```text
Bronze/Airflow -> Silver -> Gold -> Prepare training dataset -> Train Mamba -> Inference
```

Du lieu train/inference lay tu MinIO, khong lay tu folder dataset local cu.

## 0) Mo terminal va kich hoat moi truong

```powershell
cd D:\KLTN\KLTN_Mamba
.\.venv\Scripts\Activate.ps1
```

Kiem tra PyTorch GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Neu chua co torch:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel packaging
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 1) Build/chay Docker services

Dung Docker cho MinIO/Airflow/Streamlit. Neu Docker Desktop bi loi `_ping` hoac `500 Internal Server Error`, restart Docker Desktop truoc.

```powershell
cd D:\KLTN\KLTN_Mamba
docker compose -f Tool\docker-compose.yaml build
docker compose -f Tool\docker-compose.yaml up -d
```

MinIO:

```text
API     : http://localhost:9004
Console : http://localhost:9005
User    : admin
Pass    : admin123
```

Airflow:

```text
http://localhost:8081
```

Neu chi muon build service Mamba/Streamlit:

```powershell
docker compose -f Tool\docker-compose.yaml build ts-mamba
```

Kiem tra Mamba fastpath trong Docker:

```powershell
docker compose -f Tool\docker-compose.yaml run --rm ts-mamba python3 Tool/check_mamba_fastpath.py
```

## 2) Bronze layer

Bronze duoc chay bang Airflow DAG de cao du lieu raw len MinIO bucket `air-quality`.

Sau khi DAG Bronze chay xong, moi chay Silver cho cung ngay.

## 3) Silver layer

Silver doc Bronze tu MinIO, validate va ghi processed vao bucket `air-quality-silver`.

Chay cho 1 ngay:

```powershell
$d = "2026-05-23"

.\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode hourly --date $d
.\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode daily --date $d
.\.venv\Scripts\python.exe src\Silver\silver_processing.py --locations DataSet\locations.jsonl --date $d
```

Chay nhieu ngay:

```powershell
$dates = @("2026-05-20","2026-05-21","2026-05-22","2026-05-23")

foreach ($d in $dates) {
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode hourly --date $d
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode daily --date $d
  .\.venv\Scripts\python.exe src\Silver\silver_processing.py --locations DataSet\locations.jsonl --date $d
}
```

Luu y: config hien tai `seq_len=72`, `pred_len=12`, nen can toi thieu 84 gio du lieu lien tuc de tao sample train. Nen chay toi thieu 4 ngay, thuc te nen chay 30-90 ngay.

## 4) Gold feature engineering

Gold doc Silver processed tu MinIO, tao feature thoi gian va giu cac cot flag de quality gate:

```text
_invalid_segment
_flag_duplicate
_flag_physical_bound
_flag_aqi_range
_imputed
_flag_low_coverage
```

Flag chi dung de loc window xau, khong dua vao model.

Chay Gold cho nhieu ngay:

```powershell
$dates = @("2026-05-20","2026-05-21","2026-05-22","2026-05-23")

foreach ($d in $dates) {
  .\.venv\Scripts\python.exe src\Gold\gold_feature_engineering.py --locations DataSet\locations.jsonl --date $d --config Conf\air_quality.yaml
}
```

Output:

```text
s3://air-quality-gold/feature_engineering/province=<province>/year=YYYY/month=MM/day=DD/data.csv
```

## 5) Prepare training dataset

Buoc nay doc Gold features tu MinIO, tao sliding window:

```text
X = 72 gio qua khu
y = 12 gio AQI tiep theo
```

No chia train/val/test theo thoi gian va luu dataset len MinIO.

```powershell
.\.venv\Scripts\python.exe src\Gold\prepare_training_dataset.py `
  --locations DataSet\locations.jsonl `
  --config Conf\air_quality.yaml `
  --start-date 2026-05-20 `
  --end-date 2026-05-23 `
  --run-id gold_train_20260523
```

Output:

```text
s3://air-quality-gold/training_dataset/run_id=gold_train_20260523/
```

Trong do co:

```text
X_train.npy, y_train.npy
X_val.npy, y_val.npy
X_test.npy, y_test.npy
scaler.pkl
dataset_metadata.json
```

## 6) Train Mamba

Train doc dataset tu MinIO Gold, khong doc CSV local.

Dung run moi, khong ghi de run cu:

```powershell
.\.venv\Scripts\python.exe src\Model\train_mamba_aqi.py `
  --config Conf\air_quality.yaml `
  --run-id gold_train_20260523 `
  --out-dir runs\mamba_20260523_y_scaled `
  --device cuda `
  --amp
```

Output local:

```text
runs\mamba_20260523_y_scaled\best_mamba_aqi.pt
runs\mamba_20260523_y_scaled\metrics_history.csv
runs\mamba_20260523_y_scaled\test_predictions.csv
runs\mamba_20260523_y_scaled\training_metadata.json
```

Output MinIO:

```text
s3://air-quality-artifacts/mamba/run_id=mamba_20260523_y_scaled/
```

Giai thich file test:

```text
test_predictions.csv = dung de danh gia model tren test set lich su.
y_true = AQI that trong y_test.npy
y_pred = AQI model du doan
error = y_pred - y_true
abs_error = |error|
```

Day khong phai du doan tuong lai that. Day la backtest/evaluation vi test set da co dap an that.

## 7) Real inference: du doan 12 gio tiep theo

Inference that gom 2 buoc:

```text
prepare_inference_input.py -> run_mamba_inference.py
```

### 7.1) Tao X_inference tu Gold

Lenh nay lay `72` gio gan nhat trong Gold, kiem tra quality gate, scale bang scaler cua training, roi luu `X_inference.npy`.

```powershell
.\.venv\Scripts\python.exe src\Gold\prepare_inference_input.py `
  --locations DataSet\locations.jsonl `
  --config Conf\air_quality.yaml `
  --run-id gold_train_20260523 `
  --end-date 2026-05-23 `
  --lookback-days 4 `
  --output-run-id infer_20260523
```

Output:

```text
s3://air-quality-gold/inference_input/run_id=infer_20260523/X_inference.npy
s3://air-quality-gold/inference_input/run_id=infer_20260523/inference_metadata.json
```

Neu loi thieu `X_inference.npy`, nghia la chua chay buoc 7.1 hoac `--output-run-id` khac voi `--inference-run-id`.

Neu loi khong du history, tang:

```powershell
--lookback-days 7
```

### 7.2) Chay model de predict

```powershell
.\.venv\Scripts\python.exe src\Inference\run_mamba_inference.py `
  --inference-run-id infer_20260523 `
  --checkpoint runs\mamba_20260523_y_scaled\best_mamba_aqi.pt `
  --metadata runs\mamba_20260523_y_scaled\training_metadata.json `
  --output-path runs\inference\infer_20260523\future_predictions.csv `
  --artifact-run-id infer_20260523 `
  --device cuda `
  --amp
```

Output local:

```text
runs\inference\infer_20260523\future_predictions.csv
```

Output MinIO:

```text
s3://air-quality-artifacts/mamba_inference/run_id=infer_20260523/future_predictions.csv
```

File inference that chi co:

```text
province
horizon_step
forecast_time
y_pred
```

Khong co `y_true`, `error`, `abs_error` vi tuong lai chua xay ra. Sau nay khi du lieu that duoc crawl ve, co the join lai de tinh monitoring:

```text
y_true
error
abs_error
```

## 8) Luong ngan gon de chay lai tu dau

```powershell
cd D:\KLTN\KLTN_Mamba
.\.venv\Scripts\Activate.ps1

$dates = @("2026-05-20","2026-05-21","2026-05-22","2026-05-23")

foreach ($d in $dates) {
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode hourly --date $d
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode daily --date $d
  .\.venv\Scripts\python.exe src\Silver\silver_processing.py --locations DataSet\locations.jsonl --date $d
  .\.venv\Scripts\python.exe src\Gold\gold_feature_engineering.py --locations DataSet\locations.jsonl --date $d --config Conf\air_quality.yaml
}

.\.venv\Scripts\python.exe src\Gold\prepare_training_dataset.py `
  --locations DataSet\locations.jsonl `
  --config Conf\air_quality.yaml `
  --start-date 2026-05-20 `
  --end-date 2026-05-23 `
  --run-id gold_train_20260523

.\.venv\Scripts\python.exe src\Model\train_mamba_aqi.py `
  --config Conf\air_quality.yaml `
  --run-id gold_train_20260523 `
  --out-dir runs\mamba_20260523_y_scaled `
  --device cuda `
  --amp

.\.venv\Scripts\python.exe src\Gold\prepare_inference_input.py `
  --locations DataSet\locations.jsonl `
  --config Conf\air_quality.yaml `
  --run-id gold_train_20260523 `
  --end-date 2026-05-23 `
  --lookback-days 4 `
  --output-run-id infer_20260523

.\.venv\Scripts\python.exe src\Inference\run_mamba_inference.py `
  --inference-run-id infer_20260523 `
  --checkpoint runs\mamba_20260523_y_scaled\best_mamba_aqi.pt `
  --metadata runs\mamba_20260523_y_scaled\training_metadata.json `
  --output-path runs\inference\infer_20260523\future_predictions.csv `
  --artifact-run-id infer_20260523 `
  --device cuda `
  --amp
```

## 9) Loi thuong gap

### No module named torch

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### NoSuchBucket: air-quality-gold hoac air-quality-artifacts

Code upload da tu tao bucket khi ghi. Neu van gap loi, kiem tra MinIO co dang chay khong:

```powershell
docker compose -f Tool\docker-compose.yaml ps minio
```

### Missing X_inference.npy

Chay buoc `prepare_inference_input.py` truoc, va dam bao:

```text
--output-run-id infer_20260523
```

trung voi:

```text
--inference-run-id infer_20260523
```

### Ket qua train xau, R2 am

Dung run da normalize target:

```text
runs\mamba_20260523_y_scaled
```

Khong dung run cu:

```text
runs\mamba_20260523
```

Muon ket qua tot hon, can nhieu ngay du lieu hon. 4 ngay chi de test pipeline, khong du cho model on dinh.

### CUDA out of memory

Giam batch size trong `Conf\air_quality.yaml`:

```yaml
training:
  batch_size: 16
```

Sau do train lai.
