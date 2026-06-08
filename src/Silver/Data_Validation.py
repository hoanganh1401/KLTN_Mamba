"""
data_validation.py — Silver Layer Data Validation
===================================================
Hỗ trợ 2 mode validation độc lập:

  --mode hourly  (chạy sau mỗi lần ingest)
    Mục tiêu: phát hiện lỗi batch vừa cào ngay lập tức
    R1  Schema          : đủ cột bắt buộc
    R2  Duplicate ts    : (time, location) không trùng
    R4  AQI range       : 0 <= AQI <= 500
    R5  Physical bounds : từng chỉ số trong giới hạn sensor

  --mode daily  (chạy lúc 01:00 UTC hôm sau)
    Mục tiêu: validate tính đầy đủ của cả ngày hôm qua
    R3  Timestamp gap   : không có gap > 1.5h trong chuỗi
    R6  Missing rate    : mỗi cột < 30% null
    R7  Coverage        : số giờ thực tế >= 90% (>= 21/24h)

Input  : MinIO bronze (air-quality)        → raw/province={key}/year={Y}/month={MM}/day={DD}/data.csv
Output :
  - MinIO silver (air-quality-silver)    → validated/province={key}/year={Y}/month={MM}/day={DD}/data.csv
  - MinIO reports (air-quality-silver)   → validation_reports/{mode}/year={Y}/month={MM}/day={DD}/summary.json
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime, timedelta, date

import pandas as pd
from minio import Minio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from common.time_utils import now_local, parse_time_local


# =============================
# 0) Config & Constants
# =============================
MINIO_HOST          = os.environ.get("MINIO_HOST",          "localhost:9004")
MINIO_ACCESS        = os.environ.get("MINIO_ACCESS_KEY",    "admin")
MINIO_SECRET        = os.environ.get("MINIO_SECRET_KEY",    "admin123")
MINIO_BUCKET        = os.environ.get("MINIO_BUCKET",        "air-quality")
MINIO_SILVER_BUCKET = os.environ.get("MINIO_SILVER_BUCKET", "air-quality-silver")
MINIO_SECURE        = os.environ.get("MINIO_SECURE",        "false").lower() == "true"

METRIC_COLS   = ["pm25","pm10","no2","o3","so2","co","aod","dust","uv_index","co2","aqi"]
REQUIRED_COLS = ["time","location","latitude","longitude"] + METRIC_COLS

EDA_RULES_OBJECT = "eda_outputs/validation_rules.json"

_DEFAULT_PHYSICAL_BOUNDS: dict[str, list] = {
    "pm25":     [0,    1000],  "pm10":     [0,    2000],
    "no2":      [0,    1000],  "o3":       [0,     500],
    "so2":      [0,    1000],  "co":       [0,   50000],
    "aod":      [0,       5],  "dust":     [0,    2000],
    "uv_index": [0,      20],  "co2":      [300,  5000],
    "aqi":      [0,     500],
}
_DEFAULT_RULES = {
    "missing_rate_threshold": 0.30,
    "physical_bounds":        _DEFAULT_PHYSICAL_BOUNDS,
    "min_coverage_pct":       90.0,
    "aqi_valid_range":        [0, 500],
    "required_columns":       REQUIRED_COLS,
}


# =============================
# 1) MinIO helpers
# =============================
def get_client() -> Minio:
    return Minio(MINIO_HOST, access_key=MINIO_ACCESS,
                 secret_key=MINIO_SECRET, secure=MINIO_SECURE)


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def bronze_path(location_key: str, year: int, month: int, day: int) -> str:
    """Bronze: raw data từ scraper, chưa qua xử lý."""
    return (f"air_quality/province={location_key}"
            f"/year={year}/month={month:02d}/day={day:02d}/data.csv")


def silver_path(location_key: str, year: int, month: int, day: int) -> str:
    """Silver: data đã validated, có cột _flag_*."""
    return (f"validated/province={location_key}"
            f"/year={year}/month={month:02d}/day={day:02d}/data.csv")


def load_bronze(client: Minio, location_key: str,
                year: int, month: int, day: int) -> pd.DataFrame | None:
    path = bronze_path(location_key, year, month, day)
    try:
        resp = client.get_object(MINIO_BUCKET, path)
        try:
            raw = resp.read()
        finally:
            resp.close(); resp.release_conn()
        df = pd.read_csv(io.BytesIO(raw))
        df["time"] = parse_time_local(df["time"])
        return df
    except Exception as exc:
        print(f"  [WARN] Bronze not found: {path} — {exc}")
        return None


def load_silver(client: Minio, location_key: str,
                year: int, month: int, day: int) -> pd.DataFrame | None:
    """Load Silver đã có (dùng cho daily validation merge flags)."""
    path = silver_path(location_key, year, month, day)
    try:
        resp = client.get_object(MINIO_SILVER_BUCKET, path)
        try:
            raw = resp.read()
        finally:
            resp.close(); resp.release_conn()
        df = pd.read_csv(io.BytesIO(raw))
        df["time"] = parse_time_local(df["time"])
        return df
    except Exception:
        return None


def upload_csv(client: Minio, bucket: str, path: str, df: pd.DataFrame) -> None:
    ensure_bucket(client, bucket)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(bucket, path,
                      data=io.BytesIO(csv_bytes),
                      length=len(csv_bytes),
                      content_type="text/csv")


def upload_json(client: Minio, bucket: str, path: str, data: dict) -> None:
    ensure_bucket(client, bucket)
    j_bytes = json.dumps(data, ensure_ascii=False, indent=2,
                         default=str).encode("utf-8")
    client.put_object(bucket, path,
                      data=io.BytesIO(j_bytes),
                      length=len(j_bytes),
                      content_type="application/json")


def load_rules(client: Minio) -> dict:
    """Load validation rules từ EDA output. Fallback về defaults nếu chưa có."""
    try:
        resp = client.get_object(MINIO_SILVER_BUCKET, EDA_RULES_OBJECT)
        try:
            raw = resp.read()
        finally:
            resp.close(); resp.release_conn()
        rules = json.loads(raw.decode("utf-8"))
        rules = normalize_rules(rules)
        print(f"✅ Loaded validation rules from MinIO: {EDA_RULES_OBJECT}")
        return rules
    except Exception:
        print(
            f"⚠️  validation_rules.json not found at "
            f"s3://{MINIO_SILVER_BUCKET}/{EDA_RULES_OBJECT}\n"
            f"   Using fallback defaults."
        )
        return _DEFAULT_RULES


def normalize_rules(rules: dict) -> dict:
    """
    Làm mềm các threshold sinh từ EDA để pipeline vận hành ổn định hơn.

    Khi dữ liệu mẫu quá sạch, EDA có thể sinh missing_rate_threshold=0.0
    hoặc min_coverage_pct=100.0. Các giá trị đó dễ gây warning trong vận hành
    thật dù chỉ thiếu 1 giá trị hoặc 1 giờ dữ liệu về trễ.
    """
    normalized = {**_DEFAULT_RULES, **rules}

    missing_threshold = float(normalized.get("missing_rate_threshold", 0.30))
    normalized["missing_rate_threshold"] = max(missing_threshold, 0.05)

    coverage_pct = float(normalized.get("min_coverage_pct", 90.0))
    normalized["min_coverage_pct"] = min(max(coverage_pct, 80.0), 95.0)

    return normalized


# =============================
# 2) Report helper
# =============================
class ValidationReport:
    def __init__(self, location_key: str, target_date: str, mode: str):
        self.location_key = location_key
        self.target_date  = target_date
        self.mode         = mode
        self.checks: list[dict] = []
        self.has_critical = False

    def add(self, rule_id: str, passed: bool,
            critical: bool, detail: str, n_affected: int = 0) -> None:
        status = "PASS" if passed else ("FAIL_CRITICAL" if critical else "FAIL_WARNING")
        self.checks.append({
            "rule": rule_id, "status": status,
            "detail": detail, "n_affected": n_affected,
        })
        if not passed and critical:
            self.has_critical = True
        icon = "✅" if passed else ("❌" if critical else "⚠️")
        print(f"  {icon} [{rule_id}] {detail}")

    def to_dict(self) -> dict:
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        return {
            "location_key": self.location_key,
            "target_date":  self.target_date,
            "mode":         self.mode,
            "validated_at": now_local().isoformat(),
            "summary": {
                "total_checks":   len(self.checks),
                "passed":         n_pass,
                "warnings":       sum(1 for c in self.checks if c["status"] == "FAIL_WARNING"),
                "critical_fails": sum(1 for c in self.checks if c["status"] == "FAIL_CRITICAL"),
                "has_critical":   self.has_critical,
            },
            "checks": self.checks,
        }


# =============================
# 3) Hourly checks (batch quality)
# =============================
def run_hourly_checks(df: pd.DataFrame, report: ValidationReport,
                      rules: dict) -> pd.DataFrame:
    """
    Kiểm tra batch vừa cào về:
      R1 Schema, R2 Duplicate, R4 AQI range, R5 Physical bounds
    Không check R3/R6/R7 vì ngày chưa xong.
    """
    df = df.copy()
    required_cols   = rules.get("required_columns", REQUIRED_COLS)
    physical_bounds = rules.get("physical_bounds", _DEFAULT_PHYSICAL_BOUNDS)
    aqi_range       = rules.get("aqi_valid_range", [0, 500])

    # ── R1: Schema ────────────────────────────────────────────────────────────
    missing_cols = [c for c in required_cols if c not in df.columns]
    r1_pass = len(missing_cols) == 0
    report.add("R1_schema", r1_pass, critical=True,
               detail=f"Missing columns: {missing_cols}" if missing_cols
                      else f"All {len(required_cols)} required columns present")
    if not r1_pass:
        return df

    # ── R2: Duplicate timestamps ───────────────────────────────────────────────
    dup_mask = df.duplicated(subset=["time", "location"], keep=False)
    df["_flag_duplicate"] = dup_mask
    report.add("R2_duplicate_ts", dup_mask.sum() == 0, critical=False,
               detail=f"{dup_mask.sum()} duplicate (time, location) rows",
               n_affected=int(dup_mask.sum()))

    # ── R4: AQI range ──────────────────────────────────────────────────────────
    aqi_lo, aqi_hi = aqi_range[0], aqi_range[1]
    aqi_bad = df["aqi"].notna() & ((df["aqi"] < aqi_lo) | (df["aqi"] > aqi_hi))
    df["_flag_aqi_range"] = aqi_bad
    report.add("R4_aqi_range", aqi_bad.sum() == 0, critical=False,
               detail=f"{aqi_bad.sum()} rows with AQI outside [{aqi_lo}, {aqi_hi}]",
               n_affected=int(aqi_bad.sum()))

    # ── R5: Physical bounds ────────────────────────────────────────────────────
    flag_physical = pd.Series(False, index=df.index)
    phys_details  = []
    for col, bounds in physical_bounds.items():
        if col not in df.columns:
            continue
        lo, hi = bounds[0], bounds[1]
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        flag_physical |= bad
        if bad.sum() > 0:
            phys_details.append(f"{col}:{bad.sum()}")
    df["_flag_physical_bound"] = flag_physical
    report.add("R5_physical_bounds", flag_physical.sum() == 0, critical=False,
               detail=f"{flag_physical.sum()} out-of-bound rows"
                      + (f" [{', '.join(phys_details)}]" if phys_details else ""),
               n_affected=int(flag_physical.sum()))

    flag_cols = [c for c in df.columns if c.startswith("_flag_")]
    df["_flag_any"] = df[flag_cols].any(axis=1)
    return df


# =============================
# 4) Daily checks (completeness)
# =============================
def run_daily_checks(df: pd.DataFrame, report: ValidationReport,
                     rules: dict) -> pd.DataFrame:
    """
    Kiểm tra tính đầy đủ của cả ngày hôm qua:
      R3 Timestamp gap, R6 Missing rate, R7 Coverage
    Chạy trên Silver đã có (đã qua hourly checks).
    """
    df = df.copy()
    missing_threshold = rules.get("missing_rate_threshold", 0.30)
    min_coverage      = rules.get("min_coverage_pct", 90.0) / 100

    # ── R3: Timestamp continuity (full-day) ───────────────────────────────────
    ts_sorted = df["time"].dropna().sort_values()
    if len(ts_sorted) > 1:
        gaps_h   = ts_sorted.diff().dt.total_seconds().dropna() / 3600
        big_gaps = gaps_h[gaps_h > 1.5]
        n_gaps   = len(big_gaps)
        max_gap  = round(big_gaps.max(), 1) if n_gaps > 0 else 0
    else:
        n_gaps, max_gap = 0, 0
    report.add("R3_ts_continuity", n_gaps == 0, critical=False,
               detail=f"{n_gaps} gap(s) > 1.5h in full-day series, max={max_gap}h",
               n_affected=n_gaps)

    # ── R6: Missing rate per column ────────────────────────────────────────────
    high_miss = []
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        rate = df[col].isnull().mean()
        if rate > missing_threshold:
            high_miss.append(f"{col}={rate:.1%}")
    report.add("R6_missing_rate", len(high_miss) == 0, critical=False,
               detail=f"Columns > {missing_threshold:.0%} missing: "
                      + (", ".join(high_miss) if high_miss else "none"))

    # ── R7: Coverage (full-day) ────────────────────────────────────────────────
    actual_hours   = df["time"].dropna().dt.floor("h").nunique()
    expected_hours = 24
    coverage       = actual_hours / expected_hours
    df["_flag_low_coverage"] = coverage < min_coverage
    report.add("R7_coverage", coverage >= min_coverage, critical=False,
               detail=f"Coverage: {actual_hours}/{expected_hours}h = {coverage:.1%}"
                      f" (threshold: {min_coverage:.0%})")

    flag_cols = [c for c in df.columns if c.startswith("_flag_")]
    df["_flag_any"] = df[flag_cols].any(axis=1)
    return df


# =============================
# 5) Validate per location
# =============================
def validate_hourly(client: Minio, location_key: str,
                    target_date: date, rules: dict) -> dict:
    """Hourly validation: load Bronze → check batch → save Silver."""
    year, month, day = target_date.year, target_date.month, target_date.day
    print(f"\n── [HOURLY] {location_key} | {target_date} ──")

    report = ValidationReport(location_key, str(target_date), mode="hourly")

    df = load_bronze(client, location_key, year, month, day)
    if df is None:
        report.add("R0_data_exists", False, critical=True,
                   detail="Bronze data not found")
        return report.to_dict()

    report.add("R0_data_exists", True, critical=True,
               detail=f"Loaded {len(df)} rows from Bronze")

    df_flagged = run_hourly_checks(df, report, rules)

    if not report.has_critical:
        s_path = silver_path(location_key, year, month, day)
        upload_csv(client, MINIO_SILVER_BUCKET, s_path, df_flagged)
        n_flagged = int(df_flagged.get("_flag_any", pd.Series(False)).sum())
        print(f"  ✅ Silver saved ({n_flagged} flagged rows)")
    else:
        print(f"  ❌ Critical fail — Silver NOT saved")

    return report.to_dict()


def validate_daily(client: Minio, location_key: str,
                   target_date: date, rules: dict) -> dict:
    """Daily validation: load Silver → check completeness → update Silver."""
    year, month, day = target_date.year, target_date.month, target_date.day
    print(f"\n── [DAILY] {location_key} | {target_date} ──")

    report = ValidationReport(location_key, str(target_date), mode="daily")

    # Daily validation đọc từ Silver (đã qua hourly checks)
    df = load_silver(client, location_key, year, month, day)
    if df is None:
        report.add("R0_data_exists", False, critical=True,
                   detail="Silver data not found (hourly validation may have failed)")
        return report.to_dict()

    report.add("R0_data_exists", True, critical=True,
               detail=f"Loaded {len(df)} rows from Silver")

    df_flagged = run_daily_checks(df, report, rules)

    # Ghi đè Silver với daily flags bổ sung
    s_path = silver_path(location_key, year, month, day)
    upload_csv(client, MINIO_SILVER_BUCKET, s_path, df_flagged)
    print(f"  ✅ Silver updated with daily flags")

    return report.to_dict()


# =============================
# 6) Entry points
# =============================
def run_validation(locations_path: str, target_date_str: str, mode: str) -> None:
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    with open(locations_path, encoding="utf-8") as f:
        locations = [json.loads(line) for line in f if line.strip()]

    client = get_client()
    rules  = load_rules(client)

    print(f"\n{'='*55}")
    print(f"  VALIDATION [{mode.upper()}] — {target_date_str}")
    print(f"{'='*55}")

    all_reports      = []
    has_any_critical = False

    for loc in locations:
        loc_key = loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
        if mode == "hourly":
            rep = validate_hourly(client, loc_key, target_date, rules)
        else:
            rep = validate_daily(client, loc_key, target_date, rules)
        all_reports.append(rep)
        if rep["summary"]["has_critical"]:
            has_any_critical = True

    # Upload summary report
    year, month, day = target_date.year, target_date.month, target_date.day
    r_path = (f"validation_reports/{mode}"
              f"/year={year}/month={month:02d}/day={day:02d}/summary.json")
    summary = {
        "mode":             mode,
        "target_date":      target_date_str,
        "validated_at":     now_local().isoformat(),
        "total_locations":  len(all_reports),
        "passed":           sum(1 for r in all_reports if r["summary"]["critical_fails"] == 0
                                and r["summary"]["warnings"] == 0),
        "warnings":         sum(1 for r in all_reports if r["summary"]["warnings"] > 0
                                and r["summary"]["critical_fails"] == 0),
        "critical_fails":   sum(1 for r in all_reports if r["summary"]["critical_fails"] > 0),
        "location_reports": all_reports,
    }
    upload_json(client, MINIO_SILVER_BUCKET, r_path, summary)

    print(f"\n  ✅ Clean    : {summary['passed']}")
    print(f"  ⚠️  Warnings : {summary['warnings']}")
    print(f"  ❌ Critical : {summary['critical_fails']}")
    print(f"  📄 Report   : s3://{MINIO_SILVER_BUCKET}/{r_path}")

    if has_any_critical:
        print("\n❌ Critical failures detected — check report above")
        sys.exit(1)


# =============================
# 7) CLI
# =============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data validation — hourly (batch quality) hoặc daily (completeness)"
    )
    parser.add_argument("--locations", required=True,
                        help="Path to JSONL locations file")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (hourly default: today, daily default: yesterday)")
    parser.add_argument("--mode", choices=["hourly", "daily"], required=True,
                        help="hourly: check batch quality | daily: check full-day completeness")
    args = parser.parse_args()

    if args.date:
        target = args.date
    elif args.mode == "hourly":
        target = now_local().strftime("%Y-%m-%d")                 # hôm nay (+7)
    else:
        target = (now_local() - timedelta(days=1)).strftime("%Y-%m-%d")  # hôm qua (+7)

    run_validation(args.locations, target, mode=args.mode)


if __name__ == "__main__":
    main()
