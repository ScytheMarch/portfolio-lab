"""
Income analytics: dividend yield, income projection, fee-adjusted income.
"""
import numpy as np
import pandas as pd
from typing import Optional

from portfolio_lab.data.fund_metadata import FundInfo


def portfolio_income_yield(
    fund_info: dict[str, FundInfo],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Compute weighted portfolio income yield.

    Args:
        fund_info: Dict of ticker -> FundInfo (must have dividend_yield).
        weights: Dict of ticker -> weight.

    Returns:
        Tuple of (portfolio_yield, per_asset_contribution).
        Missing yields are treated as 0 with a warning logged.
    """
    total_yield = 0.0
    contributions = {}
    for ticker, w in weights.items():
        info = fund_info.get(ticker)
        if info and info.dividend_yield is not None:
            asset_yield = info.dividend_yield
        else:
            asset_yield = 0.0
        contrib = w * asset_yield
        contributions[ticker] = contrib
        total_yield += contrib

    return total_yield, contributions


def net_income_yield(
    gross_yield: float,
    expense_ratio: float,
    advisory_fee: float = 0.0,
) -> float:
    """
    Net income yield after fees.

    net = gross - expense_ratio - advisory_fee

    Note: This is a simplification. In reality, expense ratios reduce NAV
    rather than being deducted directly from income. But for planning
    purposes, this approximation is standard.
    """
    return max(gross_yield - expense_ratio - advisory_fee, 0.0)


def weighted_expense_ratio(
    fund_info: dict[str, FundInfo],
    weights: dict[str, float],
) -> tuple[float, dict[str, Optional[float]]]:
    """
    Compute weighted average expense ratio.

    Returns:
        Tuple of (portfolio_er, per_asset_er).
        Missing ERs are excluded from weighting with warning.
    """
    total_er = 0.0
    per_asset = {}
    missing_tickers = []

    for ticker, w in weights.items():
        info = fund_info.get(ticker)
        if info and info.expense_ratio is not None:
            er = info.expense_ratio
            total_er += w * er
            per_asset[ticker] = er
        else:
            missing_tickers.append(ticker)
            per_asset[ticker] = None

    return total_er, per_asset


def project_income(
    initial_value: float,
    income_yield: float,
    growth_rate: float,
    horizon_years: int,
    inflation_rate: float = 0.0,
    reinvest: bool = False,
) -> pd.DataFrame:
    """
    Project income stream over a horizon.

    Args:
        initial_value: Starting portfolio value.
        income_yield: Annual income yield (decimal).
        growth_rate: Expected annual capital growth (decimal, net of fees).
        horizon_years: Number of years.
        inflation_rate: Annual inflation for real income calculation.
        reinvest: If True, income is reinvested; if False, withdrawn.

    Returns:
        DataFrame with columns: year, nominal_income, real_income,
        portfolio_value_nominal, portfolio_value_real.
    """
    rows = []
    value = initial_value
    cumulative_inflation = 1.0

    for year in range(1, horizon_years + 1):
        income_nominal = value * income_yield
        cumulative_inflation *= (1 + inflation_rate)
        income_real = income_nominal / cumulative_inflation

        rows.append({
            "year": year,
            "nominal_income": income_nominal,
            "real_income": income_real,
            "portfolio_value_nominal": value,
            "portfolio_value_real": value / cumulative_inflation,
        })

        if reinvest:
            value = value * (1 + growth_rate)
        else:
            value = (value - income_nominal) * (1 + growth_rate)
            value = max(value, 0)

    return pd.DataFrame(rows)
