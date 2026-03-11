"""
Performance ratios: Sharpe, Sortino, and related metrics.
"""
import numpy as np
import pandas as pd

from portfolio_lab.analytics.returns import arithmetic_mean_return, annualize_volatility
from portfolio_lab.analytics.risk import downside_deviation


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """
    Annualized Sharpe Ratio = (annualized_return - risk_free_rate) / annualized_vol.

    Args:
        returns: Period returns (daily, monthly).
        risk_free_rate: Annualized risk-free rate (decimal).
        periods_per_year: Periods per year for annualization.

    Uses arithmetic mean for expected return in the numerator,
    which is standard for Sharpe ratio computation.
    """
    ann_ret = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
    ann_vol = annualize_volatility(returns.std(), periods_per_year)

    if isinstance(ann_vol, pd.Series):
        # Replace zero vol to avoid division by zero
        ann_vol = ann_vol.replace(0, np.nan)
    elif ann_vol == 0:
        return 0.0 if isinstance(ann_ret, float) else pd.Series(0.0, index=returns.columns)

    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """
    Annualized Sortino Ratio = (annualized_return - risk_free_rate) / downside_deviation.

    Uses downside deviation instead of total volatility.
    Only penalizes returns below the target_return threshold.
    """
    ann_ret = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
    dd = downside_deviation(
        returns,
        threshold=target_return / periods_per_year,  # per-period threshold
        annualize=True,
        periods_per_year=periods_per_year,
    )

    if isinstance(dd, pd.Series):
        dd = dd.replace(0, np.nan)
    elif dd == 0:
        return 0.0 if isinstance(ann_ret, float) else pd.Series(0.0, index=returns.columns)

    return (ann_ret - risk_free_rate) / dd


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio = annualized_active_return / tracking_error.

    Args:
        returns: Portfolio period returns.
        benchmark_returns: Benchmark period returns (aligned).
    """
    # Align
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0

    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0

    active_ann = active.mean() * periods_per_year
    return active_ann / te


def calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar Ratio = annualized return / |max drawdown|.
    """
    from portfolio_lab.analytics.risk import max_drawdown as mdd_fn

    ann_ret = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
    mdd = abs(mdd_fn(prices))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd
