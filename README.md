# KLTN_Mamba

Pipeline xu ly chat luong khong khi: thu thap du lieu (Bronze) -> xu ly Silver/Gold -> du doan. Du an su dung Airflow + MinIO, co the chay bang Docker hoac chay script thu cong.

## 1. Yeu cau
- Docker Desktop (khuyen nghi de chay Airflow/MinIO)
- Python 3.10+ (neu chay local)

docker compose -f Tool\docker-compose.yaml build ts-mamba
docker compose -f Tool\docker-compose.yaml up -d

## 2. Chay nhanh bang Docker (Airflow + MinIO + Postgres + Redis)
1. Mo terminal tai thu muc co file compose: [Tool/docker-compose.yaml](Tool/docker-compose.yaml)
2. Chay:
	```bash
	docker compose up -d

	(Neu may dung docker-compose cu: `docker-compose up -d`)
3. Truy cap Airflow UI: http://localhost:8081
	- Tai khoan mac dinh: `airflow` / `airflow`
4. Truy cap MinIO Console: http://localhost:9005
	- Tai khoan mac dinh: `admin` / `admin123`
	- MinIO S3 endpoint: http://localhost:9004

Ghi chu:
- Thay doi bien moi truong o [Tool/.env](Tool/.env) va [Tool/docker-compose.yaml](Tool/docker-compose.yaml) neu can (MINIO_*, WAQI_API_KEY...).

## 3. Chay DAG trong Airflow
Trong repo hien co DAG chinh o:
- [Airflow/dags/air_quality_hourly.py](Airflow/dags/air_quality_hourly.py): `air_quality_hourly`
  - Chay moi gio: ingest incremental -> validate hourly -> process Silver -> build Gold features.
- [Airflow/dags/air_quality_daily_training.py](Airflow/dags/air_quality_daily_training.py): `air_quality_daily_training`
  - Chay 01:00 hang ngay: validate daily -> refresh Gold features -> prepare training dataset -> train Mamba -> prepare inference input -> run inference.

[Airflow/dags/dags_incremental.py](Airflow/dags/dags_incremental.py) va [Airflow/dags/test.py](Airflow/dags/test.py) chi con la file deprecated, khong tao DAG rieng de tranh trung `dag_id`.

De chay:
1. Vao Airflow UI
2. Unpause DAG
3. Trigger DAG bang tay hoac cho scheduler chay theo lich

## 4. Chay data scraper thu cong (khong can Airflow)
Script: [DataSet/data_scraper.py](DataSet/data_scraper.py)

Tao file .env o root (D:\KLTN\KLTN_Mamba) neu can tuy chinh:
```
MINIO_HOST=localhost:9004
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_BUCKET=air-quality
MINIO_SECURE=false
```

Chay incremental (gap-fill + hom nay):
```bash
python DataSet/data_scraper.py --mode incremental --locations DataSet/locations.jsonl --lookback-days 30
```

Chay backfill:
```bash
python DataSet/data_scraper.py --mode backfill --start-date 2025-01-01 --end-date 2025-02-01 --locations DataSet/locations.jsonl
```

## 5. Chay Streamlit app
File app hien dang trong: [App/streamlit_app.py](App/streamlit_app.py). Ban co the them UI, sau do chay:
```bash
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run App/streamlit_app.py
```
Mo trinh duyet: http://localhost:8501

## 6. Dung dich vu Docker
```bash
docker compose down
```
