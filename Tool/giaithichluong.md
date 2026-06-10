# Giải thích luồng AQI Mamba

Tài liệu này giải thích bản chất pipeline hiện tại của project `KLTN_Mamba`: dữ liệu được crawl lên MinIO, đi qua Bronze/Silver/Gold, sau đó tạo dataset cho Mamba, train model và dự đoán AQI 12 giờ tiếp theo.

Luồng chính:

```text
Bronze/Airflow
  -> Silver validation + processing
  -> Gold feature engineering
  -> Prepare training dataset
  -> Train Mamba
  -> Prepare inference input
  -> Run inference
  -> Lưu kết quả predict/metrics lên MinIO
```

Điểm quan trọng nhất: model Mamba hiện tại không dùng location embedding nữa. Tỉnh/thành được xử lý bằng cách group theo `province/location`, tạo chuỗi thời gian riêng cho từng tỉnh. `loc_ids` nếu còn tồn tại trong dataset chỉ dùng để report lại tên tỉnh trong file kết quả, không đưa vào model.

## 1. Bronze Layer

Bronze là lớp dữ liệu thô.

Nguồn dữ liệu được Airflow crawl và lưu lên MinIO bucket:

```text
s3://air-quality/
```

Vai trò:

- Lưu dữ liệu vừa crawl, gần với dữ liệu gốc nhất.
- Chưa làm sạch sâu.
- Là đầu vào cho Silver.

Bronze hiện được chạy bằng Airflow DAG. Sau khi DAG Bronze chạy xong cho một ngày cụ thể, mới chạy tiếp Silver cho cùng ngày đó.

## 2. Silver Layer

Silver là lớp dữ liệu đã được validate và làm sạch.

Các file chính:

```text
src/Silver/Data_Validation.py
src/Silver/silver_processing.py
```

### 2.1. Data_Validation.py

File này đọc dữ liệu Bronze từ MinIO, kiểm tra chất lượng dữ liệu và gắn các cờ chất lượng.

Ví dụ các cờ:

```text
_flag_any
_flag_aqi_range
_flag_duplicate
_flag_low_coverage
_flag_physical_bound
_imputed
_invalid_segment
```

Ý nghĩa:

- Các cờ này không phải feature cho model.
- Chúng dùng để biết dòng dữ liệu hoặc đoạn dữ liệu có đáng tin hay không.
- Sau này Gold dùng các cờ này để quyết định window nào được dùng để train/predict.

Chạy validate hourly:

```powershell
.\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode hourly --date 2026-05-23
```

Chạy validate daily:

```powershell
.\.venv\Scripts\python.exe src\Silver\Data_Validation.py --locations DataSet\locations.jsonl --mode daily --date 2026-05-23
```

Output lưu ở:

```text
s3://air-quality-silver/
```

### 2.2. silver_processing.py

File này xử lý dữ liệu sau validation:

- Chuẩn hóa cột thời gian về `time`.
- Xử lý missing.
- Xử lý duplicate.
- Xử lý outlier theo rule/strategy nếu có.
- Impute những khoảng thiếu có thể chấp nhận.
- Lưu dữ liệu processed theo partition tỉnh/ngày.

Lệnh chạy:

```powershell
.\.venv\Scripts\python.exe src\Silver\silver_processing.py --locations DataSet\locations.jsonl --date 2026-05-23
```

Output processed có dạng:

```text
s3://air-quality-silver/processed/province=<province>/year=YYYY/month=MM/day=DD/data.csv
```

Ví dụ:

```text
s3://air-quality-silver/processed/province=an_giang/year=2026/month=05/day=23/data.csv
```

Sau Silver, dữ liệu đã sạch hơn nhưng chưa phải dataset train cho model.

## 3. Gold Layer

Gold là lớp dữ liệu phục vụ trực tiếp cho model.

Gold gồm 3 việc:

```text
gold_feature_engineering.py
prepare_training_dataset.py
prepare_inference_input.py
```

## 4. Gold Feature Engineering

File:

```text
src/Gold/gold_feature_engineering.py
```

Đầu vào:

```text
s3://air-quality-silver/processed/
```

Đầu ra:

```text
s3://air-quality-gold/feature_engineering/
```

File này tạo thêm time features:

```text
hour_sin
hour_cos
month_sin
month_cos
day_of_week
is_weekend
```

Các feature này giúp model hiểu chu kỳ theo giờ, tháng, ngày trong tuần.

Ví dụ:

```text
Silver processed data
  -> thêm time features
  -> Gold feature data
```

Lệnh chạy:

```powershell
.\.venv\Scripts\python.exe src\Gold\gold_feature_engineering.py --locations DataSet\locations.jsonl --date 2026-05-23 --config Conf\air_quality.yaml
```

Output:

```text
s3://air-quality-gold/feature_engineering/province=<province>/year=YYYY/month=MM/day=DD/data.csv
```

Lưu ý: các cột flag vẫn có thể còn ở Gold feature để phục vụ quality gate, nhưng không đưa vào model.

## 5. Prepare Training Dataset

File:

```text
src/Gold/prepare_training_dataset.py
```

Đây là bước biến dữ liệu Gold feature thành tensor train/val/test cho Mamba.

Đầu vào:

```text
s3://air-quality-gold/feature_engineering/
```

Đầu ra:

```text
s3://air-quality-gold/training_dataset/run_id=<run_id>/
```

Theo config hiện tại:

```yaml
dataset:
  seq_len: 72
  pred_len: 12
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

Nghĩa là:

```text
X = 72 giờ quá khứ
y = 12 giờ AQI tương lai
```

Ví dụ một sample:

```text
X:
2026-05-20 00:00 -> 2026-05-22 23:00

y:
2026-05-23 00:00 -> 2026-05-23 11:00
```

### Quality gate trong prepare_training_dataset.py

Các cột flag không đưa vào model, nhưng được dùng để loại window xấu.

Config:

```yaml
quality:
  hard_fail_cols:
    - _invalid_segment
    - _flag_duplicate
    - _flag_physical_bound
    - _flag_aqi_range
  imputed_col: _imputed
  max_imputed_ratio: 0.3
  low_coverage_col: _flag_low_coverage
  max_low_coverage_ratio: 0.5
```

Cách hiểu:

- Nếu window có `_invalid_segment`, `_flag_duplicate`, `_flag_physical_bound`, `_flag_aqi_range` thì bỏ.
- Nếu tỷ lệ `_imputed` vượt 30% thì bỏ.
- Nếu tỷ lệ `_flag_low_coverage` vượt 50% thì bỏ.
- Sau khi kiểm tra xong, các cờ này bị loại khỏi feature model.

### Vì sao không dùng location embedding?

Dữ liệu hiện đã được partition theo tỉnh:

```text
province=an_giang/year=2026/month=05/day=23
province=ha_noi/year=2026/month=05/day=23
...
```

Khi tạo sliding window, code group theo tỉnh để đảm bảo chuỗi thời gian không bị trộn giữa các tỉnh. Vì vậy model chỉ cần học từ feature thời gian và chỉ số môi trường trong từng window.

`loc_ids` chỉ còn để biết sample đó thuộc tỉnh nào khi xuất `test_predictions.csv`, không còn là input của Mamba.

Lệnh chạy đúng hiện tại là tạo dataset riêng cho từng tỉnh. Ví dụ với `2026-05-20` đến `2026-05-25`:

```powershell
$start = "2026-05-20"
$end = "2026-05-25"
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"

  .\.venv\Scripts\python.exe src\Gold\prepare_training_dataset.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --start-date $start `
    --end-date $end `
    --run-id $datasetRun
}
```

Output chính:

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
X_test.npy
y_test.npy
loc_ids_train.npy
loc_ids_val.npy
loc_ids_test.npy
y_ts_train.npy
y_ts_val.npy
y_ts_test.npy
scaler.pkl
dataset_metadata.json
```

Trong đó:

- `X_*`: input cho model.
- `y_*`: AQI thật tương ứng với horizon dự đoán.
- `loc_ids_*`: metadata để map lại province, không đưa vào model.
- `y_ts_*`: thời gian tương ứng với y.
- `scaler.pkl`: scaler fit trên train set.
- `dataset_metadata.json`: metadata của dataset.

## 6. Train Mamba

File:

```text
src/Model/Mamba/train_mamba_aqi.py
```

Đầu vào:

```text
s3://air-quality-gold/training_dataset/run_id=<run_id>/
```

Model không đọc dataset local cũ trong `TS_MAMBA/dataset`.

Lệnh train đúng hiện tại là train từng tỉnh từ dataset riêng. `train_mamba_aqi.py` tự tạo thư mục tạm, train xong upload artifacts lên MinIO, rồi tự xóa local tạm nếu không truyền `--keep-local`.

```powershell
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"
  $modelRun = "mamba_${loc}_${tag}"

  .\.venv\Scripts\python.exe src\Model\Mamba\train_mamba_aqi.py `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --model-run-id $modelRun `
    --device cuda `
    --amp
}
```

Trong train:

- Model nhận `X_train`.
- Target là `y_train`.
- Train bằng sliding window.
- Validation dùng `X_val/y_val`.
- Test dùng `X_test/y_test`.
- Target được normalize bằng thống kê của train split, rồi denormalize khi tính metrics.

Output MinIO mỗi tỉnh:

```text
s3://air-quality-artifacts/mamba/province=<province>/run_id=mamba_<province>_20260525/
```

Trong mỗi folder tỉnh có:

```text
best_model.pt
best_mamba_aqi.pt
metrics_history.csv
test_predictions.csv
training_metadata.json
artifact_manifest.json
```

### metrics_history.csv

File này ghi quá trình train theo epoch.

Các cột chính:

```text
epoch
train_loss
val_loss
mae
rmse
mae_norm
rmse_norm
val_r2
train_sec
```

Ý nghĩa:

- `train_loss`: loss trên train.
- `val_loss`: loss trên validation.
- `mae`: sai số tuyệt đối trung bình theo đơn vị AQI thật.
- `rmse`: căn bậc hai MSE theo đơn vị AQI thật.
- `mae_norm`, `rmse_norm`: metric trên target đã normalize.
- `val_r2`: R2 trên validation.
- `train_sec`: thời gian train epoch.

### test_predictions.csv

File này dùng để đánh giá model trên dữ liệu lịch sử đã biết đáp án.

Các cột quan trọng:

```text
province
time
horizon_step
y_true
y_pred
error
abs_error
```

Ý nghĩa:

```text
y_true = AQI thật trong test set
y_pred = AQI model dự đoán
error = y_pred - y_true
abs_error = |y_pred - y_true|
```

Ví dụ:

```text
y_true = 120
y_pred = 110
error = -10
abs_error = 10
```

Nghĩa là model dự đoán thấp hơn thực tế 10 AQI.

Test set là 15% dữ liệu cuối theo thời gian, tức dữ liệu quá khứ đã có AQI thật. Vì vậy `test_predictions.csv` có `y_true`.

Đây không phải dự đoán tương lai thật cho người dùng.

## 7. Prepare Inference Input

File:

```text
src/Gold/prepare_inference_input.py
```

Đây là bước chuẩn bị input mới nhất để dự đoán tương lai.

Đầu vào:

```text
s3://air-quality-gold/feature_engineering/
s3://air-quality-gold/training_dataset/run_id=<training_run_id>/scaler.pkl
```

Đầu ra:

```text
s3://air-quality-gold/inference_input/run_id=<infer_run_id>/
```

File này làm:

- Đọc Gold feature các ngày gần nhất.
- Group theo tỉnh.
- Lấy `seq_len = 72` giờ gần nhất cho mỗi tỉnh.
- Kiểm tra quality gate.
- Load `scaler.pkl` từ training dataset.
- Transform bằng scaler cũ.
- Lưu `X_inference.npy`.

Điểm rất quan trọng:

```text
prepare_inference_input.py không fit scaler mới.
Nó phải dùng scaler.pkl đã fit ở prepare_training_dataset.py.
```

Lệnh chạy hiện tại cũng chạy theo từng tỉnh, dùng đúng scaler của dataset tỉnh đó:

```powershell
$end = "2026-05-25"
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $datasetRun = "gold_train_${loc}_${tag}"
  $inferRun = "infer_${loc}_${tag}"

  .\.venv\Scripts\python.exe src\Gold\prepare_inference_input.py `
    --locations DataSet\locations.jsonl `
    --location-keys $loc `
    --config Conf\air_quality.yaml `
    --run-id $datasetRun `
    --end-date $end `
    --lookback-days 7 `
    --output-run-id $inferRun
}
```

Output:

```text
X_inference.npy
inference_metadata.json
```

Nếu báo không đủ history, nghĩa là Gold chưa có đủ 72 giờ liên tục hoặc bị quality gate loại. Khi đó cần:

- Chạy Bronze/Silver/Gold thêm ngày.
- Tăng `--lookback-days`.
- Kiểm tra dữ liệu thiếu ở MinIO.

## 8. Run Inference

File:

```text
src/Inference/run_mamba_inference.py
```

Đầu vào:

```text
X_inference.npy
inference_metadata.json
best_mamba_aqi.pt
training_metadata.json
```

Lệnh chạy hiện tại load model trực tiếp từ MinIO theo `province` và `model-run-id`, không cần checkpoint local:

```powershell
$tag = "20260525"

$locations = Get-Content DataSet\locations.jsonl | ForEach-Object {
  (ConvertFrom-Json $_).location_key
}

foreach ($loc in $locations) {
  $modelRun = "mamba_${loc}_${tag}"
  $inferRun = "infer_${loc}_${tag}"

  .\.venv\Scripts\python.exe src\Inference\run_mamba_inference.py `
    --inference-run-id $inferRun `
    --province $loc `
    --model-run-id $modelRun `
    --artifact-run-id $inferRun `
    --device cuda `
    --amp
}
```

Nếu không truyền `--output-path`, file predict chỉ được ghi tạm, upload lên MinIO xong sẽ tự xóa local. Chỉ thêm `--keep-local` khi cần debug.

Output MinIO:

```text
s3://air-quality-artifacts/mamba_inference/province=<province>/forecast_date=<YYYY-MM-DD>/run_id=infer_<province>_20260525/future_predictions.csv
```

Ví dụ nếu lấy dữ liệu đến ngày `2026-05-25` và dự đoán bắt đầu sang `2026-05-26`, kết quả của An Giang sẽ lưu ở:

```text
s3://air-quality-artifacts/mamba_inference/province=an_giang/forecast_date=2026-05-26/run_id=infer_an_giang_20260525/future_predictions.csv
```

### future_predictions.csv

File này là dự đoán thật cho tương lai.

Nó có dạng:

```text
province
forecast_time
horizon_step
y_pred
```

Ở thời điểm dự đoán thật, chưa có `y_true`, vì tương lai chưa xảy ra.

Sau này khi dữ liệu thật của các giờ đó được crawl về, đi qua Silver/Gold, mới có thể join lại để tạo monitoring:

```text
province
forecast_time
horizon_step
y_true
y_pred
error
abs_error
```

## 9. Phân biệt train, test và inference thật

### Training

Training dùng dữ liệu lịch sử.

```text
X_train = quá khứ
y_train = AQI thật sau X_train
```

Model học bằng nhiều sliding window.

### Validation

Validation dùng dữ liệu lịch sử nhưng nằm sau train theo thời gian.

Dùng để chọn checkpoint tốt nhất và early stopping.

### Test

Test là phần dữ liệu cuối theo thời gian.

Vì đây vẫn là dữ liệu lịch sử nên có `y_true`.

File:

```text
test_predictions.csv
```

Dùng để đánh giá model.

### Real inference

Real inference lấy 72 giờ gần nhất đã có trong Gold để dự đoán 12 giờ tiếp theo.

Ở thời điểm predict, chỉ có:

```text
y_pred
```

Chưa có:

```text
y_true
error
abs_error
```

## 10. Direct Forecast hay Recursive Forecast?

Pipeline hiện tại là direct multi-step forecast.

Nghĩa là một lần forward model sẽ dự đoán thẳng 12 bước:

```text
Input: 72 giờ quá khứ
Output: 12 giờ tương lai
```

Không phải recursive forecast.

Recursive forecast là kiểu dự đoán giờ thứ 1, rồi lấy dự đoán đó nhét lại vào input để dự đoán giờ thứ 2, lặp tiếp cho tới giờ thứ 12. Pipeline hiện tại không làm như vậy.

Vì vậy:

```text
Train: sliding window để tạo nhiều sample.
Predict: direct 12-hour forecast.
```

Hai ý này không mâu thuẫn nhau.

## 11. Các bucket MinIO đang dùng

Theo config:

```yaml
minio:
  bronze_bucket: air-quality
  silver_bucket: air-quality-silver
  gold_bucket: air-quality-gold
  artifacts_bucket: air-quality-artifacts
```

Vai trò:

```text
air-quality
  dữ liệu raw Bronze

air-quality-silver
  dữ liệu validate/processed

air-quality-gold
  feature engineering, training dataset, inference input

air-quality-artifacts
  checkpoint, metrics_history, test_predictions, future_predictions
```

## 12. Luồng chạy nhiều ngày

Ví dụ chạy từ `2026-05-20` đến `2026-05-25` theo đúng yêu cầu mỗi tỉnh một dataset, một model, một metrics:

```powershell
cd D:\KLTN\KLTN_Mamba
.\.venv\Scripts\Activate.ps1

$dateStart = [datetime]"2026-01-01"
$dateEnd = [datetime]"2026-05-25"
$dates = for ($d = $dateStart; $d -le $dateEnd; $d = $d.AddDays(1)) { $d.ToString("yyyy-MM-dd") }
$start = "2026-01-01"
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

## 13. Luồng Airflow dự kiến sau này

Khi đưa vào Airflow, thứ tự task hợp lý:

```text
bronze_crawl
  -> silver_validate_hourly
  -> silver_validate_daily
  -> silver_processing
  -> gold_feature_engineering
  -> for each province:
       prepare_training_dataset
       train_mamba
       prepare_inference_input
       run_mamba_inference
```

Nếu không muốn train mỗi ngày, có thể tách:

```text
Daily DAG:
  bronze -> silver -> gold -> prepare_inference_input -> run_inference

Retrain DAG:
  prepare_training_dataset -> train_mamba
```

Cách này thực tế hơn:

- Inference có thể chạy hằng ngày hoặc hằng giờ.
- Retrain có thể chạy theo lịch riêng, ví dụ mỗi tuần hoặc mỗi tháng.

## 14. Tóm tắt ngắn gọn

```text
Silver = dữ liệu sạch
Gold feature = dữ liệu sạch + feature thời gian
Training dataset = Gold feature -> sliding window train/val/test
Train Mamba = mỗi tỉnh một model, học từ X_train/y_train, đánh giá bằng val/test
test_predictions.csv = đánh giá lịch sử, có y_true
prepare_inference_input = lấy 72 giờ mới nhất của từng tỉnh, scale bằng scaler cũ của tỉnh đó
future_predictions.csv = dự đoán tương lai thật theo từng tỉnh, chỉ có y_pred
```

Với pipeline hiện tại:

```text
Train dùng sliding window.
Predict dùng direct multi-step forecast 12 giờ.
Location không còn embedding.
MinIO là nguồn dữ liệu chính.
Artifact train lưu theo s3://air-quality-artifacts/mamba/province=<province>/run_id=<model_run_id>/.
```
