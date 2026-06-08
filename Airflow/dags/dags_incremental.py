"""
Deprecated DAG file.

Bronze incremental ingestion is now scheduled as the first task of
``air_quality_hourly`` in ``air_quality_hourly.py``. This file intentionally
does not define a DAG, so Airflow will not create a duplicate ingestion DAG.
"""
