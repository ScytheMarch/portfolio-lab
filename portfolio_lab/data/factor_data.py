"""
Fama-French 5-factor data fetching and caching.

Primary source: Kenneth French's data library.
Falls back gracefully with warnings if unavailable.
"""
import io
import logging
import zipfile
from typing import Optional

import pandas as pd
import requests

from portfolio_lab.config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache
_ff5_daily_cache: Optional[pd.DataFrame] = None
_ff5_monthly_cache: Optional[pd.DataFrame] = None

FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def fetch_ff5_factors(frequency: str = "daily", force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5-factor data.

    Args:
        frequency: 'daily' or 'monthly'.
        force_refresh: Bypass cache.

    Returns:
        DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
        Index: DatetimeIndex
        Values: decimal returns (e.g., 0.01 = 1%)
        None if fetch fails.
    """
    global _ff5_daily_cache, _ff5_monthly_cache

    if frequency == "daily" and _ff5_daily_cache is not None and not force_refresh:
        return _ff5_daily_cache.copy()
    if frequency == "monthly" and _ff5_monthly_cache is not None and not force_refresh:
        return _ff5_monthly_cache.copy()

    settings = get_settings()
    url = settings.ff5_source_url if frequency == "daily" else settings.ff5_monthly_url

    try:
        logger.info(f"Downloading FF5 {frequency} factors from French data library...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("utf-8")

        df = _parse_ff5_csv(raw, frequency)
        if df is not None and not df.empty:
            if frequency == "daily":
                _ff5_daily_cache = df
            else:
                _ff5_monthly_cache = df
            logger.info(f"FF5 {frequency} data loaded: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
            return df.copy()

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download FF5 data: {e}")
    except Exception as e:
        logger.warning(f"Failed to parse FF5 data: {e}")

    logger.warning(
        "Fama-French 5-factor data unavailable. Factor analysis will be disabled. "
        "You can retry or provide data manually."
    )
    return None


def _parse_ff5_csv(raw: str, frequency: str) -> Optional[pd.DataFrame]:
    """
    Parse the raw CSV text from Kenneth French's zip file.

    The format has header rows, then data, then possibly annual data.
    Data values are in percentage points (e.g., 0.50 = 0.50%).
    """
    lines = raw.strip().split("\n")

    # Find the header row containing "Mkt-RF"
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:
            header_idx = i
            break

    if header_idx is None:
        logger.warning("Could not find FF5 header row.")
        return None

    # Find end of data (blank line or non-numeric first column)
    data_lines = []
    for line in lines[header_idx + 1:]:
        parts = line.strip().split(",")
        if len(parts) < 6:
            break
        # First column should be a date-like number
        try:
            int(parts[0].strip())
            data_lines.append(line)
        except ValueError:
            break

    if not data_lines:
        return None

    # Build DataFrame
    header = lines[header_idx].strip().split(",")
    header = [h.strip() for h in header]

    records = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            records.append(parts[:7])  # date + 5 factors + RF

    col_names = header[:7] if len(header) >= 7 else header
    # First column is the date
    df = pd.DataFrame(records, columns=col_names)

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    # Parse dates
    if frequency == "daily":
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m", errors="coerce")

    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

    # Convert to float and from percentage to decimal
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.dropna()
    return df


def get_factor_names() -> list[str]:
    """Return the standard FF5 factor names."""
    return FF5_FACTORS.copy()
