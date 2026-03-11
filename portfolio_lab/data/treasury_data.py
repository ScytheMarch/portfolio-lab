"""
Treasury data: risk-free rate from 91-day T-bill, Treasury proxies.

Uses ^IRX (13-week T-bill yield index) as primary source.
Falls back to BIL ETF trailing return or manual default.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_lab.config import get_settings

logger = logging.getLogger(__name__)


# Standard Treasury proxy tickers for portfolio construction
TREASURY_PROXIES = {
    "T-Bill (1-3 month)": ["BIL", "SGOV"],
    "Short-Term (1-3 year)": ["SHV", "SHY", "VGSH"],
    "Intermediate (3-10 year)": ["IEF", "VGIT", "GOVT"],
    "Long-Term (10-30 year)": ["TLT", "VGLT", "EDV"],
    "TIPS (inflation-protected)": ["TIP", "VTIP", "SCHP"],
}


def get_treasury_proxy_tickers() -> dict[str, list[str]]:
    """Return dict of maturity bucket -> list of proxy ETF tickers."""
    return TREASURY_PROXIES.copy()


def fetch_risk_free_rate(manual_override: Optional[float] = None) -> tuple[float, str]:
    """
    Fetch the current risk-free rate from the 91-day T-bill.

    Attempts in order:
    1. Manual override if provided.
    2. ^IRX (13-week T-bill discount yield index).
    3. BIL ETF trailing 30-day annualized return as proxy.
    4. Settings default.

    Returns:
        Tuple of (annualized_rate_decimal, source_description).

    The ^IRX index quotes in percentage points (e.g., 5.25 = 5.25%).
    We convert to decimal (0.0525).
    """
    settings = get_settings()

    if manual_override is not None:
        rate = float(manual_override)
        source = f"manual_override ({rate:.4f})"
        logger.info(f"Risk-free rate: {source}")
        return rate, source

    # Attempt 1: ^IRX
    rate, source = _try_irx()
    if rate is not None:
        return rate, source

    # Attempt 2: BIL ETF proxy
    rate, source = _try_bil_proxy()
    if rate is not None:
        return rate, source

    # Fallback
    rate = settings.risk_free_rate
    source = f"settings_default ({rate:.4f})"
    logger.warning(f"Using fallback risk-free rate: {source}")
    return rate, source


def _try_irx() -> tuple[Optional[float], str]:
    """Try fetching the 13-week T-bill yield from ^IRX."""
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if hist.empty:
            logger.warning("^IRX returned no data.")
            return None, ""
        # ^IRX quotes in percentage points
        latest = hist["Close"].dropna().iloc[-1]
        rate = float(latest) / 100.0  # convert to decimal
        if rate < 0 or rate > 0.30:  # sanity check
            logger.warning(f"^IRX returned suspicious value: {latest}")
            return None, ""
        source = f"^IRX 13-week T-bill ({rate:.4f})"
        logger.info(f"Risk-free rate from ^IRX: {source}")
        return rate, source
    except Exception as e:
        logger.warning(f"Failed to fetch ^IRX: {e}")
        return None, ""


def _try_bil_proxy() -> tuple[Optional[float], str]:
    """Estimate risk-free rate from BIL ETF trailing return."""
    try:
        bil = yf.Ticker("BIL")
        hist = bil.history(period="3mo")
        if hist.empty or len(hist) < 20:
            return None, ""
        # Annualize the trailing return
        prices = hist["Close"].dropna()
        total_return = prices.iloc[-1] / prices.iloc[0] - 1
        days = (prices.index[-1] - prices.index[0]).days
        if days <= 0:
            return None, ""
        annual_return = (1 + total_return) ** (365.0 / days) - 1
        rate = float(annual_return)
        if rate < -0.01 or rate > 0.20:
            logger.warning(f"BIL proxy returned suspicious rate: {rate}")
            return None, ""
        source = f"BIL ETF proxy ({rate:.4f})"
        logger.info(f"Risk-free rate from BIL: {source}")
        return rate, source
    except Exception as e:
        logger.warning(f"Failed to compute BIL proxy rate: {e}")
        return None, ""


def fetch_treasury_yields() -> dict[str, Optional[float]]:
    """
    Fetch approximate current yields for various Treasury maturities.

    Returns dict of label -> yield (decimal) or None.
    Uses yfinance Treasury yield tickers where available.
    """
    yield_tickers = {
        "3-Month": "^IRX",
        # Note: yfinance has no direct 2-Year ticker; omitted to avoid misleading data
        "5-Year": "^FVX",
        "10-Year": "^TNX",
        "30-Year": "^TYX",
    }

    results = {}
    for label, ticker in yield_tickers.items():
        try:
            obj = yf.Ticker(ticker)
            hist = obj.history(period="5d")
            if not hist.empty:
                val = hist["Close"].dropna().iloc[-1]
                results[label] = float(val) / 100.0
            else:
                results[label] = None
        except Exception:
            results[label] = None

    return results
