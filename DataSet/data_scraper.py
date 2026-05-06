"""
Bronze ingestion: Open-Meteo Air Quality API → MinIO
Storage layout:
  {bucket}/air_quality/province={location_key}/year={YYYY}/month={MM}/data.csv
"""

import io
import json
import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Iterator, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from minio import Minio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================
# 0) Exceptions & Config
# =============================
class RateLimitError(Exception):
    """Raised when Open-Meteo returns 429 after all retries are exhausted."""


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

MINIO_HOST   = os.environ.get("MINIO_HOST",       "localhost:9004")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "admin123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET",     "air-quality")
MINIO_SECURE = os.environ.get("MINIO_SECURE",     "false").lower() == "true"

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Columns to fetch from Open-Meteo
HOURLY_VARS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
    "sulphur_dioxide", "carbon_monoxide",
    "aerosol_optical_depth", "dust", "uv_index",
    "carbon_dioxide", "us_aqi",
]

# Rename map: API field → project field
RENAME_MAP = {
    "pm2_5":              "pm25",
    "nitrogen_dioxide":   "no2",
    "ozone":              "o3",
    "sulphur_dioxide":    "so2",
    "carbon_monoxide":    "co",
    "aerosol_optical_depth": "aod",
    "carbon_dioxide":     "co2",
    "us_aqi":             "aqi",
}

# Final column order saved to CSV
OUTPUT_COLS = [
    "time", "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2", "aqi",
    "location", "latitude", "longitude",
]


# =============================
# 1) Helpers
# =============================
def build_http_session() -> requests.Session:
    """Session with automatic retry on transient network errors (not 429)."""
    session = requests.Session()
    retry = Retry(
        total=0,          # we handle retries manually for full control
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


def get_minio_client() -> Minio:
    return Minio(
        MINIO_HOST,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def load_locations(path: str) -> list[dict]:
    """Load a JSONL or JSON file of locations."""
    if not path:
        raise ValueError("--locations path is required.")

    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    # Try JSONL first (multiple lines)
    if len(lines) > 1:
        parsed = []
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if parsed:
            return parsed

    # Fallback: plain JSON
    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "locations" in data:
        return data["locations"]

    raise ValueError(f"Unsupported locations format in: {path}")


def generate_date_chunks(
    start: str, end: str, days: int = 90
) -> Iterator[tuple[str, str]]:
    """Yield (chunk_start, chunk_end) pairs of at most `days` each."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    current = s
    while current <= e:
        chunk_end = min(current + timedelta(days=days - 1), e)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + timedelta(days=1)


def backoff_sleep(attempt: int, base: float = 2.0, cap: float = 60.0) -> None:
    delay = min(base ** attempt, cap)
    print(f"    Sleeping {delay:.0f}s before retry …")
    time.sleep(delay)


def object_path(location_key: str, year: int, month: int, day: int) -> str:
    """MinIO object path for a given location/year/month/day partition."""
    return (
        f"air_quality/province={location_key}"
        f"/year={year}/month={month:02d}/day={day:02d}/data.csv"
    )


# =============================
# 2) API Fetch
# =============================
def fetch_openmeteo(
    session: requests.Session,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Fetch hourly air-quality data from Open-Meteo.
    Raises RateLimitError on persistent 429.
    Returns an empty DataFrame on other unrecoverable errors.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     HOURLY_VARS,
        "timezone":   "UTC",
        "start_date": start_date,
        "end_date":   end_date,
    }

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            r = session.get(API_URL, params=params, timeout=30)

            if r.status_code == 429:
                print(
                    f"    [WARN] 429 Too Many Requests "
                    f"({lat},{lon} {start_date}→{end_date}) "
                    f"attempt {attempt+1}/{max_retries}"
                )
                if attempt == max_retries - 1:
                    raise RateLimitError(
                        f"Rate limit exhausted for ({lat},{lon} {start_date}→{end_date})"
                    )
                backoff_sleep(attempt + 1)
                continue

            r.raise_for_status()
            js: dict = r.json()
            break

        except RateLimitError:
            raise

        except Exception as exc:
            last_exc = exc
            print(f"    [ERROR] attempt {attempt+1}/{max_retries}: {exc}")
            if attempt == max_retries - 1:
                return pd.DataFrame()
            backoff_sleep(attempt + 1)
    else:
        # all retries exhausted without breaking
        print(f"    [ERROR] All retries failed: {last_exc}")
        return pd.DataFrame()

    hourly = js.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df.rename(columns=RENAME_MAP, inplace=True)

    # Parse & gate timestamps
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df[df["time"].notna() & (df["time"] <= pd.Timestamp.now(tz="UTC"))]

    df["latitude"]  = js.get("latitude")
    df["longitude"] = js.get("longitude")

    # Keep only the desired output columns (fill missing with NaN)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[OUTPUT_COLS].reset_index(drop=True)


# =============================
# 3) MinIO Storage (partition by location + year + month + day)
# =============================
def save_to_minio(
    client: Minio,
    df: pd.DataFrame,
    location_key: str,
    incremental: bool = False,
) -> None:
    """
    Partition data by (year, month, day) and merge into the corresponding object.

    incremental=True  → append only rows newer than the stored max(time).
    incremental=False → full upsert / dedup on (time, location).
    """
    ensure_bucket(client, MINIO_BUCKET)

    if df.empty:
        return

    # Ensure time is tz-aware
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    for (year, month, day), group in df.groupby(
        [df["time"].dt.year, df["time"].dt.month, df["time"].dt.day]
    ):
        path = object_path(location_key, year, month, day)
        new_rows = group.copy()

        # --- Load existing data (if any) ---
        existing: Optional[pd.DataFrame] = None
        try:
            resp = client.get_object(MINIO_BUCKET, path)
            try:
                raw = resp.read()
            finally:
                resp.close()
                resp.release_conn()

            existing = pd.read_csv(io.BytesIO(raw))
            existing["time"] = pd.to_datetime(
                existing["time"], utc=True, errors="coerce"
            )
        except Exception:
            pass  # object doesn't exist yet

        # --- Incremental: skip rows already stored ---
        if incremental and existing is not None and not existing.empty:
            max_stored = existing["time"].max()
            new_rows = new_rows[new_rows["time"] > max_stored]

        # --- Merge & dedup ---
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, new_rows], ignore_index=True)
        else:
            merged = new_rows

        merged = (
            merged
            .drop_duplicates(subset=["time", "location"], keep="last")
            .sort_values("time", kind="mergesort")
            .reset_index(drop=True)
        )

        csv_bytes = merged.to_csv(index=False).encode("utf-8")
        client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=path,
            data=io.BytesIO(csv_bytes),
            length=len(csv_bytes),
            content_type="text/csv",
        )
        print(
            f"    ✅ s3://{MINIO_BUCKET}/{path} "
            f"(rows={len(merged)}, new={len(new_rows)})"
        )



# =============================
# 4) Gap Detection
# =============================
def get_stored_dates(client: Minio, location_key: str) -> set:
    """
    Scan MinIO prefix for a location and return the set of dates (datetime.date)
    that already have a data.csv stored.
    Path pattern: air_quality/province={key}/year={Y}/month={MM}/day={DD}/data.csv
    """
    prefix = f"air_quality/province={location_key}/"
    stored: set = set()

    try:
        objects = client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
        for obj in objects:
            parts = obj.object_name.split("/")
            try:
                year  = int(next(p for p in parts if p.startswith("year=")).split("=")[1])
                month = int(next(p for p in parts if p.startswith("month=")).split("=")[1])
                day   = int(next(p for p in parts if p.startswith("day=")).split("=")[1])
                stored.add(datetime(year, month, day).date())
            except (StopIteration, ValueError):
                continue
    except Exception as exc:
        print(f"  [WARN] Could not scan MinIO for {location_key}: {exc}")

    return stored


def find_missing_dates(
    client: Minio,
    location_key: str,
    lookback_days: int = 30,
) -> list:
    """
    Return sorted list of missing dates in [today - lookback_days, yesterday].
    Today is excluded because it is handled by the normal incremental fetch.
    """
    today     = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    start     = today - timedelta(days=lookback_days)

    expected = {start + timedelta(days=i) for i in range((yesterday - start).days + 1)}
    stored   = get_stored_dates(client, location_key)
    return sorted(expected - stored)


def _fetch_and_save(
    session: requests.Session,
    client: Minio,
    loc: dict,
    start_date: str,
    end_date: str,
    incremental: bool,
) -> bool:
    """Shared fetch → save helper. Returns False on RateLimitError."""
    lat     = loc["latitude"]
    lon     = loc["longitude"]
    loc_key = loc.get("location_key") or f"{lat}_{lon}"

    try:
        df = fetch_openmeteo(session, lat, lon, start_date, end_date)
    except RateLimitError as exc:
        print(f"  ⛔ {exc}")
        return False

    if df.empty:
        print(f"  ⚠️  No data for {loc_key} {start_date}→{end_date}")
        return True

    df["location"] = loc_key
    save_to_minio(client, df, loc_key, incremental=incremental)
    time.sleep(0.4)
    return True


# =============================
# 5) Ingestion Modes
# =============================
def run_backfill(
    locations: list[dict],
    start_date: str,
    end_date: str,
    chunk_days: int = 90,
) -> None:
    client  = get_minio_client()
    session = build_http_session()

    for loc in locations:
        loc_key  = loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
        loc_name = loc.get("name", loc_key)
        print(f"\n==> Backfill: {loc_name} ({loc_key})")

        for s, e in generate_date_chunks(start_date, end_date, days=chunk_days):
            print(f"  Fetching {s} → {e} …")
            ok = _fetch_and_save(session, client, loc, s, e, incremental=False)
            if not ok:
                print("  Stopping backfill. Re-run later to continue.")
                return

    print("\n✔ BACKFILL COMPLETE.")


def run_incremental(
    locations: list[dict],
    lookback_days: int = 30,
) -> None:
    """
    For each location:
      1. Detect missing days in the last `lookback_days` window → gap-fill them.
      2. Fetch today's data normally.
    """
    client  = get_minio_client()
    session = build_http_session()
    today   = datetime.utcnow().strftime("%Y-%m-%d")

    for loc in locations:
        loc_key  = loc.get("location_key") or f"{loc['latitude']}_{loc['longitude']}"
        loc_name = loc.get("name", loc_key)

        print(f"\n==> Incremental: {loc_name} ({loc_key})  date={today}")

        # ── Step 1: Gap detection & auto backfill ──────────────────────────
        missing = find_missing_dates(client, loc_key, lookback_days=lookback_days)

        if missing:
            print(f"  🔍 Found {len(missing)} missing day(s): "
                  f"{missing[0]} … {missing[-1]}")

            # Batch consecutive missing days into chunks to minimise API calls
            for s, e in generate_date_chunks(
                missing[0].strftime("%Y-%m-%d"),
                missing[-1].strftime("%Y-%m-%d"),
                days=90,
            ):
                print(f"  ↩️  Gap-fill {s} → {e} …")
                ok = _fetch_and_save(session, client, loc, s, e, incremental=False)
                if not ok:
                    print("  Stopping gap-fill due to rate limit.")
                    break
        else:
            print("  ✔ No gaps found in the lookback window.")

        # ── Step 2: Fetch today ────────────────────────────────────────────
        print(f"  Fetching today ({today}) …")
        ok = _fetch_and_save(session, client, loc, today, today, incremental=True)
        if not ok:
            print("  Stopping incremental run due to rate limit.")
            break

    print("\n✔ INCREMENTAL COMPLETE.")


# =============================
# 6) CLI
# =============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bronze ingestion: Open-Meteo Air Quality → MinIO"
    )
    parser.add_argument(
        "--mode", choices=["backfill", "incremental"], required=True,
        help="backfill: historical range | incremental: today + auto gap-fill"
    )
    parser.add_argument("--start-date",    help="YYYY-MM-DD  (required for backfill)")
    parser.add_argument("--end-date",      help="YYYY-MM-DD  (default: today)")
    parser.add_argument("--chunk-days",    type=int, default=90,
                        help="Days per API request chunk (default: 90)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Days to look back for gap detection (default: 30)")
    parser.add_argument("--locations",     required=True,
                        help="Path to JSONL/JSON file with location records")
    args = parser.parse_args()

    locations = load_locations(args.locations)

    if args.mode == "backfill":
        if not args.start_date:
            parser.error("--start-date is required for backfill mode")
        end_date = args.end_date or datetime.utcnow().strftime("%Y-%m-%d")
        run_backfill(locations, args.start_date, end_date, args.chunk_days)
    else:
        run_incremental(locations, lookback_days=args.lookback_days)


if __name__ == "__main__":
    main()