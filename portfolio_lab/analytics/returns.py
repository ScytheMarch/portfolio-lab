"""
Return calculations: arithmetic, geometric, CAGR, log returns.

All functions clearly distinguish between return types.
Conventions:
    - Returns are in decimal form (0.05 = 5%).
    - Annual returns assume 252 trading days unless specified.
    - Geometric mean and CAGR are preferred for multi-period realized performance.
    - Arithmetic mean is used for single-period expected return estimation.
"""
import numpy as np
import pandas as pd

from portfolio_lab.config import get_settings


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) period returns from prices."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute continuously compounded (log) returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def arithmetic_mean_return(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int | None = None,
) -> pd.Series | float:
    """
    Arithmetic mean of period returns, optionally annualized.

    This is the simple average. Appropriate for single-period expected return
    estimation but overstates multi-period compounded growth.

    Args:
        returns: Period returns (daily, monthly, etc.).
        annualize: If True, scale to annual.
        periods_per_year: Trading days or months per year. Auto-detected if None.
    """
    ppy = periods_per_year or _infer_periods_per_year(returns)
    mean = returns.mean()
    if annualize:
        return mean * ppy
    return mean


def geometric_mean_return(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int | None = None,
) -> pd.Series | float:
    """
    Geometric mean of period returns, optionally annualized.

    This is the compounded average growth rate per period. More accurate
    than arithmetic mean for realized multi-period performance.

    Geometric mean = (prod(1 + r_i))^(1/n) - 1

    Args:
        returns: Period returns (daily, monthly, etc.).
        annualize: If True, compound to annual.
        periods_per_year: Auto-detected if None.
    """
    ppy = periods_per_year or _infer_periods_per_year(returns)

    if isinstance(returns, pd.DataFrame):
        result = {}
        for col in returns.columns:
            result[col] = _geo_mean_series(returns[col], annualize, ppy)
        return pd.Series(result)
    else:
        return _geo_mean_series(returns, annualize, ppy)


def _geo_mean_series(r: pd.Series, annualize: bool, ppy: int) -> float:
    """Geometric mean for a single return series."""
    r_clean = r.dropna()
    n = len(r_clean)
    if n == 0:
        return 0.0

    # Use log-sum-exp for numerical stability
    cumulative = np.sum(np.log1p(r_clean.values))
    geo_per_period = np.exp(cumulative / n) - 1

    if annualize:
        return (1 + geo_per_period) ** ppy - 1
    return geo_per_period


def cagr(prices: pd.Series | pd.DataFrame) -> pd.Series | float:
    """
    Compound Annual Growth Rate from a price series.

    CAGR = (P_end / P_start)^(1/years) - 1

    This is the realized annualized return. Not a forward estimate.
    """
    if isinstance(prices, pd.DataFrame):
        return pd.Series({col: _cagr_series(prices[col]) for col in prices.columns})
    return _cagr_series(prices)


def _cagr_series(p: pd.Series) -> float:
    """CAGR for a single price series."""
    p_clean = p.dropna()
    if len(p_clean) < 2:
        return 0.0
    start = p_clean.iloc[0]
    end = p_clean.iloc[-1]
    if start <= 0:
        return 0.0

    # Calculate years from index
    days = (p_clean.index[-1] - p_clean.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25

    return (end / start) ** (1.0 / years) - 1


def annualize_return(
    period_return: float,
    periods_per_year: int = 252,
    method: str = "compound",
) -> float:
    """
    Annualize a per-period return.

    Args:
        period_return: Return per period (e.g., daily return).
        periods_per_year: Number of periods per year.
        method: 'compound' (geometric) or 'simple' (arithmetic scaling).
    """
    if method == "compound":
        return (1 + period_return) ** periods_per_year - 1
    else:
        return period_return * periods_per_year


def annualize_volatility(
    period_vol: float,
    periods_per_year: int = 252,
) -> float:
    """Annualize volatility using sqrt(T) rule."""
    return period_vol * np.sqrt(periods_per_year)


def exponentially_weighted_return(
    returns: pd.Series | pd.DataFrame,
    span: int = 60,
    annualize: bool = True,
    periods_per_year: int | None = None,
) -> pd.Series | float:
    """
    Exponentially weighted mean return (gives more weight to recent data).

    Args:
        returns: Period returns.
        span: EWM span parameter (number of periods for half-life).
        annualize: If True, scale to annual.
    """
    ppy = periods_per_year or _infer_periods_per_year(returns)
    ewm_mean = returns.ewm(span=span).mean().iloc[-1]
    if annualize:
        return ewm_mean * ppy
    return ewm_mean


def _infer_periods_per_year(returns: pd.Series | pd.DataFrame) -> int:
    """Infer whether data is daily or monthly from index frequency."""
    settings = get_settings()
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 10:
        median_gap = pd.Series(returns.index).diff().median()
        if median_gap is not None and hasattr(median_gap, "days"):
            if median_gap.days > 15:
                return settings.months_per_year
    return settings.trading_days_per_year
