"""
Market data fetching via yfinance.

Handles adjusted prices, dividends, and basic validation.
Designed to be replaceable with other data providers.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_lab.config import get_settings

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when data fetching fails or returns insufficient data."""
    pass


def fetch_price_data(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_years: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (adjusted close).
        Missing tickers are dropped with a warning.
    """
    settings = get_settings()
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        years = lookback_years or settings.default_lookback_years
        start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    logger.info(f"Fetching price data for {tickers} from {start} to {end}")

    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        raise DataFetchError(f"yfinance download failed: {e}")

    if raw.empty:
        raise DataFetchError(f"No data returned for tickers: {tickers}")

    # Handle single vs multi-ticker yfinance output
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker returns flat columns
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Validate each ticker
    valid_tickers = []
    warnings = []
    for t in tickers:
        if t not in prices.columns:
            warnings.append(f"Ticker '{t}' not found in downloaded data.")
            continue
        col = prices[t].dropna()
        if len(col) < 60:  # minimum ~3 months of daily data
            warnings.append(
                f"Ticker '{t}' has only {len(col)} data points (need >=60). Dropping."
            )
            continue
        valid_tickers.append(t)

    for w in warnings:
        logger.warning(w)

    if not valid_tickers:
        raise DataFetchError(
            f"No valid tickers with sufficient data. Warnings: {warnings}"
        )

    prices = prices[valid_tickers].dropna(how="all")
    # Forward-fill small gaps (weekends already handled), then drop remaining NaN rows
    prices = prices.ffill(limit=5).dropna()

    return prices


def fetch_dividend_data(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_years: Optional[int] = None,
) -> dict[str, pd.Series]:
    """
    Fetch dividend history for each ticker.

    Returns:
        Dict mapping ticker -> Series of dividend payments (DatetimeIndex).
        Empty Series for tickers with no dividend history.
    """
    settings = get_settings()
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        years = lookback_years or settings.default_lookback_years
        start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    result = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            divs = ticker_obj.dividends
            if divs is not None and not divs.empty:
                # Filter to date range
                mask = (divs.index >= start) & (divs.index <= end)
                result[t] = divs.loc[mask]
            else:
                result[t] = pd.Series(dtype=float)
                logger.info(f"No dividend data for {t}")
        except Exception as e:
            logger.warning(f"Failed to fetch dividends for {t}: {e}")
            result[t] = pd.Series(dtype=float)

    return result


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute return series from price data.

    Args:
        prices: Adjusted close prices.
        method: 'simple' for arithmetic returns, 'log' for log returns.

    Returns:
        DataFrame of returns (same shape minus first row).
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()
