"""
Efficient frontier computation.

Traces the frontier by solving min-volatility for a range of target returns.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_lab.optimization.solver import optimize_portfolio

logger = logging.getLogger(__name__)


@dataclass
class EfficientFrontierResult:
    """Result from efficient frontier computation."""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights_matrix: np.ndarray  # (n_points, n_assets)
    tickers: list[str]
    max_sharpe_idx: int
    min_vol_idx: int
    risk_free_rate: float


def compute_efficient_frontier(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.05,
    n_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    max_vol: Optional[float] = None,
) -> EfficientFrontierResult:
    """
    Compute the efficient frontier by tracing target returns.

    1. Find minimum volatility portfolio.
    2. Find maximum return portfolio.
    3. Solve min-vol for n_points target returns between them.

    Args:
        tickers: Ticker list.
        expected_returns: Annualized expected returns.
        cov_matrix: Annualized covariance matrix.
        risk_free_rate: For Sharpe calculation.
        n_points: Number of frontier points.
        min_weight/max_weight: Weight bounds.

    Returns:
        EfficientFrontierResult with frontier data.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(tickers)

    # Find min-vol portfolio
    min_vol_res = optimize_portfolio(
        tickers, mu, sigma, risk_free_rate,
        objective="min_volatility",
        min_weight=min_weight, max_weight=max_weight,
    )

    if not min_vol_res.success:
        logger.warning("Min volatility optimization failed for frontier.")
        # Fallback: use equal weight as min return bound
        min_ret = float(np.mean(mu))
    else:
        min_ret = min_vol_res.expected_return

    # Max achievable return is the highest single-asset return (given bounds)
    max_ret = float(np.max(mu))

    # Generate target returns
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_rets = []
    frontier_vols = []
    frontier_sharpes = []
    frontier_weights = []

    for target in target_returns:
        res = optimize_portfolio(
            tickers, mu, sigma, risk_free_rate,
            objective="target_return",
            target_return=target,
            min_weight=min_weight, max_weight=max_weight,
            max_vol=max_vol,
            n_restarts=5,
        )

        if res.success:
            frontier_rets.append(res.expected_return)
            frontier_vols.append(res.volatility)
            frontier_sharpes.append(res.sharpe_ratio)
            frontier_weights.append(res.raw_weights)

    if not frontier_rets:
        logger.error("Efficient frontier computation failed: no feasible points.")
        # Return a minimal result
        return EfficientFrontierResult(
            returns=np.array([]),
            volatilities=np.array([]),
            sharpe_ratios=np.array([]),
            weights_matrix=np.array([]),
            tickers=tickers,
            max_sharpe_idx=0,
            min_vol_idx=0,
            risk_free_rate=risk_free_rate,
        )

    rets = np.array(frontier_rets)
    vols = np.array(frontier_vols)
    sharpes = np.array(frontier_sharpes)
    weights_mat = np.array(frontier_weights)

    max_sharpe_idx = int(np.argmax(sharpes))
    min_vol_idx = int(np.argmin(vols))

    return EfficientFrontierResult(
        returns=rets,
        volatilities=vols,
        sharpe_ratios=sharpes,
        weights_matrix=weights_mat,
        tickers=tickers,
        max_sharpe_idx=max_sharpe_idx,
        min_vol_idx=min_vol_idx,
        risk_free_rate=risk_free_rate,
    )
