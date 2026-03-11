"""
Fama-French 5-factor model: regression, factor exposures, factor-implied returns.

Runs OLS regressions of asset excess returns on FF5 factors.
Degrades gracefully when factor data is unavailable.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_lab.data.factor_data import fetch_ff5_factors, FF5_FACTORS

logger = logging.getLogger(__name__)


@dataclass
class FactorRegressionResult:
    """Results from a single-asset factor regression."""
    ticker: str
    alpha: float = 0.0  # annualized alpha
    alpha_t_stat: float = 0.0
    factor_loadings: dict[str, float] = field(default_factory=dict)
    factor_t_stats: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    residual_vol: float = 0.0
    n_obs: int = 0
    warnings: list[str] = field(default_factory=list)


def run_factor_regression(
    asset_returns: pd.Series,
    factor_data: Optional[pd.DataFrame] = None,
    frequency: str = "daily",
    annualization_factor: int = 252,
) -> FactorRegressionResult:
    """
    Run FF5 regression for a single asset.

    Model: R_i - RF = alpha + beta1*(Mkt-RF) + beta2*SMB + beta3*HML
                                + beta4*RMW + beta5*CMA + epsilon

    Args:
        asset_returns: Simple return series for the asset.
        factor_data: FF5 DataFrame. Fetched if None.
        frequency: 'daily' or 'monthly'.
        annualization_factor: 252 for daily, 12 for monthly.

    Returns:
        FactorRegressionResult.
    """
    ticker = asset_returns.name or "Unknown"
    result = FactorRegressionResult(ticker=ticker)

    if factor_data is None:
        factor_data = fetch_ff5_factors(frequency=frequency)

    if factor_data is None:
        result.warnings.append("Factor data unavailable. Regression skipped.")
        return result

    # Align dates
    combined = pd.concat([asset_returns, factor_data], axis=1, join="inner").dropna()

    if len(combined) < 60:
        result.warnings.append(
            f"Only {len(combined)} overlapping observations (need >=60). "
            "Regression may be unreliable."
        )
    if len(combined) < 30:
        result.warnings.append("Too few observations for meaningful regression.")
        return result

    # Dependent variable: excess return
    y = combined.iloc[:, 0] - combined["RF"]

    # Independent variables: 5 factors
    X_cols = [c for c in FF5_FACTORS if c in combined.columns]
    if not X_cols:
        result.warnings.append("No factor columns found in data.")
        return result

    X = combined[X_cols].values
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    y_arr = y.values

    # OLS via normal equation: (X'X)^-1 X'y
    try:
        XtX = X_with_const.T @ X_with_const
        Xty = X_with_const.T @ y_arr
        betas = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        result.warnings.append("Singular matrix in regression. Skipping.")
        return result

    # Residuals and statistics
    y_hat = X_with_const @ betas
    residuals = y_arr - y_hat
    n = len(y_arr)
    k = len(betas)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k) if n > k else 0.0

    # Standard errors
    mse = ss_res / (n - k) if n > k else 0.0
    try:
        var_betas = mse * np.linalg.inv(XtX)
        se_betas = np.sqrt(np.diag(var_betas))
        t_stats = betas / se_betas
    except np.linalg.LinAlgError:
        se_betas = np.zeros(k)
        t_stats = np.zeros(k)

    # Pack results
    result.alpha = float(betas[0]) * annualization_factor  # annualize daily alpha
    result.alpha_t_stat = float(t_stats[0])
    result.r_squared = float(r_squared)
    result.adj_r_squared = float(adj_r_squared)
    # Use sqrt(MSE) for residual vol — accounts for degrees of freedom (n - k)
    result.residual_vol = float(np.sqrt(mse)) * np.sqrt(annualization_factor)
    result.n_obs = n

    for i, factor_name in enumerate(X_cols):
        result.factor_loadings[factor_name] = float(betas[i + 1])
        result.factor_t_stats[factor_name] = float(t_stats[i + 1])

    return result


def run_portfolio_factor_regression(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    factor_data: Optional[pd.DataFrame] = None,
    frequency: str = "daily",
) -> FactorRegressionResult:
    """
    Run FF5 regression on the weighted portfolio return series.
    """
    w = np.asarray(weights)
    # Compute portfolio returns
    common_idx = asset_returns.dropna().index
    port_returns = (asset_returns.loc[common_idx] * w).sum(axis=1)
    port_returns.name = "Portfolio"
    ann = 252 if frequency == "daily" else 12
    return run_factor_regression(port_returns, factor_data, frequency, ann)


def estimate_factor_loadings(
    asset_returns: pd.DataFrame,
    factor_data: Optional[pd.DataFrame] = None,
    frequency: str = "daily",
) -> dict[str, FactorRegressionResult]:
    """
    Run factor regressions for all assets.

    Returns:
        Dict mapping ticker -> FactorRegressionResult.
    """
    if factor_data is None:
        factor_data = fetch_ff5_factors(frequency=frequency)

    results = {}
    for col in asset_returns.columns:
        ann = 252 if frequency == "daily" else 12
        results[col] = run_factor_regression(
            asset_returns[col], factor_data, frequency, ann
        )
    return results


def portfolio_factor_exposures(
    asset_loadings: dict[str, FactorRegressionResult],
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Compute weighted portfolio-level factor exposures.

    Args:
        asset_loadings: Per-asset factor regression results.
        weights: Asset weights.

    Returns:
        Dict of factor_name -> weighted loading.
    """
    exposures = {}
    for factor in FF5_FACTORS:
        exposure = 0.0
        for ticker, w in weights.items():
            if ticker in asset_loadings:
                exposure += w * asset_loadings[ticker].factor_loadings.get(factor, 0.0)
        exposures[factor] = exposure
    return exposures


def factor_implied_expected_return(
    factor_loadings: dict[str, float],
    factor_risk_premia: Optional[dict[str, float]] = None,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Estimate expected return from factor exposures and risk premia.

    E[R] = RF + sum(beta_i * premium_i)

    Args:
        factor_loadings: Asset or portfolio factor betas.
        factor_risk_premia: Annualized factor risk premiums. Uses historical
            averages if not provided.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Factor-implied expected return (annualized).
    """
    # Historical average risk premia (approximate long-term annualized)
    # These are rough estimates; users should override for forward-looking views.
    default_premia = {
        "Mkt-RF": 0.06,  # equity risk premium ~6%
        "SMB": 0.02,     # small cap premium ~2%
        "HML": 0.03,     # value premium ~3%
        "RMW": 0.03,     # profitability premium ~3%
        "CMA": 0.02,     # investment premium ~2%
    }

    premia = factor_risk_premia or default_premia
    implied = risk_free_rate
    for factor, loading in factor_loadings.items():
        if factor in premia:
            implied += loading * premia[factor]

    return implied
