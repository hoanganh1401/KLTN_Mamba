"""Shared local-time helpers for Vietnam air-quality data."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

LOCAL_TZ = "Asia/Ho_Chi_Minh"
LOCAL_ZONE = ZoneInfo(LOCAL_TZ)


def now_local() -> datetime:
    """Return the current Vietnam local time."""
    return datetime.now(LOCAL_ZONE)


def parse_time_local(values) -> pd.Series:
    """Parse datetime-like values as Vietnam local time.

    Naive timestamps are treated as local wall-clock time. Timezone-aware
    timestamps are converted to Vietnam time, so old +00:00 rows and new +07:00
    rows can still be read consistently.
    """
    raw = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(raw):
        parsed = pd.to_datetime(raw, errors="coerce")
        if getattr(parsed.dt, "tz", None) is None:
            return parsed.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
        return parsed.dt.tz_convert(LOCAL_TZ)

    def _one(value):
        if pd.isna(value):
            return pd.NaT
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is None:
            return ts.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
        return ts.tz_convert(LOCAL_TZ)

    return raw.map(_one)


def format_time_local_strings(values) -> pd.Series:
    """Format datetime-like values as YYYY-MM-DD HH:MM:SS+07:00 strings."""
    ts = parse_time_local(values)
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S+07:00")
