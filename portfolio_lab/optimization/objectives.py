"""
Objective functions for portfolio optimization.

Each function returns a scalar to be MINIMIZED by scipy.optimize.
For maximization objectives (e.g., Sharpe), we return the negative.
"""
import numpy as np


def negative_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Negative Sharpe ratio (minimize this to maximize Sharpe).

    Sharpe = (w'μ - rf) / sqrt(w'Σw)
    """
    port_return = weights @ expected_returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    if port_vol < 1e-10:
        return 0.0
    return -(port_return - risk_free_rate) / port_vol


def portfolio_volatility_obj(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Portfolio volatility (minimize)."""
    return np.sqrt(weights @ cov_matrix @ weights)


def negative_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """Negative expected return (minimize to maximize return)."""
    return -(weights @ expected_returns)


def negative_income_yield(
    weights: np.ndarray,
    yields: np.ndarray,
) -> float:
    """Negative weighted income yield (minimize to maximize yield)."""
    return -(weights @ yields)


def negative_net_income(
    weights: np.ndarray,
    yields: np.ndarray,
    expense_ratios: np.ndarray,
) -> float:
    """
    Negative net income yield after expenses.

    net_yield = sum(w_i * (yield_i - er_i))
    """
    net_yields = yields - expense_ratios
    return -(weights @ net_yields)


def negative_diversification_ratio(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Negative diversification ratio (minimize to maximize diversification)."""
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol = weights @ asset_vols
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    if port_vol < 1e-10:
        return 0.0
    return -(weighted_vol / port_vol)
