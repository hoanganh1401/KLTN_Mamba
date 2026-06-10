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

Neu train tu ngay nao den ngay nao thi Silver cung phai chay du dung khoang ngay do. Vi du chay tu `2026-05-20` den `2026-05-25`:

```powershell
cd D:\KLTN\KLTN_Mamba
.\.venv\Scripts\Activate.ps1

$start = [datetime]"2026-01-01"
$end = [datetime]"2026-05-25"
$dates = for ($d = $start; $d -le $end; $d = $d.AddDays(1)) { $d.ToString("yyyy-MM-dd") }

foreach ($d in $dates) {
  Write-Host "`n===== SILVER $d ====="

  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py `
    --locations DataSet\locations.jsonl `
    --mode hourly `
    --date $d

  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py `
    --locations DataSet\locations.jsonl `
    --mode daily `
    --date $d

  .\.venv\Scripts\python.exe src\Silver\silver_processing.py `
    --locations DataSet\locations.jsonl `
    --date $d
}
```

Luu y: config hien tai `seq_len=96`, `pred_len=12`, nen can toi thieu 108 gio du lieu lien tuc moi tao duoc sample. De train Mamba on dinh, nen dung toi thieu 60-180 ngay; train 5-6 ngay se chi co vai chuc sample va rat de cho R2 am.

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

Gold phai chay cung khoang ngay voi Silver:

```powershell
$dates = @("2026-05-20","2026-05-21","2026-05-22","2026-05-23","2026-05-24","2026-05-25")

foreach ($d in $dates) {
  Write-Host "`n===== GOLD $d ====="

  .\.venv\Scripts\python.exe src\Gold\gold_feature_engineering.py `
    --locations DataSet\locations.jsonl `
    --date $d `
    --config Conf\air_quality.yaml
}
```

Output:

```text
s3://air-quality-gold/feature_engineering/province=<province>/year=YYYY/month=MM/day=DD/data.csv
```

## 5) Prepare training dataset va train tung tinh

Luong hien tai dung:

```text
moi tinh -> 1 training dataset rieng -> 1 model rieng -> 1 metrics rieng
```

Moi tinh se tao sliding window rieng:

```text
X = 96 gio qua khu
y = 12 gio AQI tiep theo
```

Lenh duoi day tao dataset tung tinh va train tung tinh. `train_mamba_aqi.py` tu tao thu muc tam, train xong upload artifacts len MinIO, roi tu xoa local tam. Khong can truyen `--out-dir`.

```powershell
$start = "2026-01-01"
$end = "2026-05-25"
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"
  $modelRun = "mamba_${loc}_${tag}"

  Write-Host "`n===== TRAIN $loc ====="

  .\.venv\Scripts\python.exe src\Gold\prepare_training_dataset.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --start-date $start `
    --end-date $end `
    --run-id $datasetRun

  .\.venv\Scripts\python.exe src\Model\Mamba\train_mamba_aqi.py `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --model-run-id $modelRun `
    --device cuda `
    --amp
}
```

Output dataset train tung tinh:

```text
s3://air-quality-gold/training_dataset/run_id=gold_train_<province>_20260525/
```

Output model/metrics tung tinh:

```text
s3://air-quality-artifacts/mamba/province=<province>/run_id=mamba_<province>_20260525/
```

Trong moi folder model co:

```text
best_model.pt
best_mamba_aqi.pt
metrics_history.csv
test_predictions.csv
training_metadata.json
artifact_manifest.json
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

Chay predict tung tinh bang model da upload tren MinIO:

```powershell
$end = "2026-05-25"
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"
  $modelRun = "mamba_${loc}_${tag}"
  $inferRun = "infer_${loc}_${tag}"

  Write-Host "`n===== PREDICT $loc ====="

  .\.venv\Scripts\python.exe src\Gold\prepare_inference_input.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --end-date $end `
    --lookback-days 7 `
    --output-run-id $inferRun

  .\.venv\Scripts\python.exe src\Inference\run_mamba_inference.py `
    --inference-run-id $inferRun `
    --province $loc `
    --model-run-id $modelRun `
    --artifact-run-id $inferRun `
    --device cuda `
    --amp
}
```

Output input inference tung tinh:

```text
s3://air-quality-gold/inference_input/run_id=infer_<province>_20260525/X_inference.npy
s3://air-quality-gold/inference_input/run_id=infer_<province>_20260525/inference_metadata.json
```

Neu loi thieu `X_inference.npy`, nghia la chua chay buoc 7.1 hoac `--output-run-id` khac voi `--inference-run-id`.

Neu loi khong du history, tang:

```powershell
--lookback-days 7
```

### 7.2) Chay model de predict

Trong flow tung tinh, `run_mamba_inference.py` load checkpoint va metadata truc tiep tu MinIO bang:

```text
--province <province>
--model-run-id mamba_<province>_20260525
```

Neu khong truyen `--output-path`, file predict chi duoc ghi tam, upload len MinIO xong se tu xoa local. Chi them `--keep-local` khi can debug.

Output MinIO:

```text
s3://air-quality-artifacts/mamba_inference/province=<province>/forecast_date=<YYYY-MM-DD>/run_id=infer_<province>_20260525/future_predictions.csv
```

Ví dụ input kết thúc ngày `2026-05-25`, forecast bắt đầu sang `2026-05-26`, output sẽ nằm kiểu:

```text
s3://air-quality-artifacts/mamba_inference/province=an_giang/forecast_date=2026-05-26/run_id=infer_an_giang_20260525/future_predictions.csv
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

$dates = @("2026-05-20","2026-05-21","2026-05-22","2026-05-23","2026-05-24","2026-05-25")
$start = "2026-05-20"
$end = "2026-05-25"
$tag = "20260525"

foreach ($d in $dates) {
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode hourly --date $d
  .\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode daily --date $d
  .\.venv\Scripts\python.exe src\Silver\silver_processing.py --locations DataSet\locations.jsonl --date $d
  .\.venv\Scripts\python.exe src\Gold\gold_feature_engineering.py --locations DataSet\locations.jsonl --date $d --config Conf\air_quality.yaml
}

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"
  $modelRun = "mamba_${loc}_${tag}"
  $inferRun = "infer_${loc}_${tag}"

  .\.venv\Scripts\python.exe src\Gold\prepare_training_dataset.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --start-date $start `
    --end-date $end `
    --run-id $datasetRun

  .\.venv\Scripts\python.exe src\Model\Mamba\train_mamba_aqi.py `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --model-run-id $modelRun `
    --device cuda `
    --amp

  .\.venv\Scripts\python.exe src\Gold\prepare_inference_input.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --end-date $end `
    --lookback-days 7 `
    --output-run-id $inferRun

  .\.venv\Scripts\python.exe src\Inference\run_mamba_inference.py `
    --inference-run-id $inferRun `
    --province $loc `
    --model-run-id $modelRun `
    --artifact-run-id $inferRun `
    --device cuda `
    --amp
}
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
--output-run-id infer_<province>_20260525
```

trung voi:

```text
--inference-run-id infer_<province>_20260525
```

### Ket qua train xau, R2 am

Dung run tung tinh da upload len MinIO:

```text
s3://air-quality-artifacts/mamba/province=<province>/run_id=mamba_<province>_20260525/
```

Khong dung global run cu:

```text
s3://air-quality-artifacts/mamba/run_id=<global_run>/
```

Muon ket qua tot hon, can nhieu ngay du lieu hon. 4 ngay chi de test pipeline, khong du cho model on dinh.

### CUDA out of memory

Giam batch size trong `Conf\air_quality.yaml`:

```yaml
training:
  batch_size: 16
```

Sau do train lai.
