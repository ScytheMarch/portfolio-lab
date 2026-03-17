"""
Constraint builders for portfolio optimization.

Returns lists of scipy.optimize constraint dicts or bound tuples.
"""
import numpy as np
from typing import Optional


def full_investment_constraint() -> dict:
    """Weights must sum to 1."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def weight_bounds(
    n_assets: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    per_asset_bounds: Optional[dict[int, tuple[float, float]]] = None,
) -> list[tuple[float, float]]:
    """
    Build bounds for each asset weight.

    Args:
        n_assets: Number of assets.
        min_weight: Global minimum weight (0 = no shorting).
        max_weight: Global maximum weight.
        per_asset_bounds: Override bounds for specific assets by index.

    Returns:
        List of (min, max) tuples for scipy.
    """
    bounds = [(min_weight, max_weight)] * n_assets
    if per_asset_bounds:
        for idx, (lo, hi) in per_asset_bounds.items():
            if 0 <= idx < n_assets:
                bounds[idx] = (lo, hi)
    return bounds


def max_volatility_constraint(
    cov_matrix: np.ndarray,
    max_vol: float,
) -> dict:
    """Portfolio volatility must not exceed max_vol."""
    return {
        "type": "ineq",
        "fun": lambda w: max_vol - np.sqrt(w @ cov_matrix @ w),
    }


def min_return_constraint(
    expected_returns: np.ndarray,
    min_return: float,
) -> dict:
    """Portfolio expected return must be at least min_return."""
    return {
        "type": "ineq",
        "fun": lambda w: w @ expected_returns - min_return,
    }


def target_return_constraint(
    expected_returns: np.ndarray,
    target_return: float,
) -> dict:
    """Portfolio expected return must be at least target_return.

    Uses an inequality constraint (>=) instead of equality (==) because
    equality constraints are often infeasible when weight bounds and
    diversification constraints limit achievable returns.  The optimizer
    still minimizes volatility, so it naturally lands as close to the
    target as possible without overshooting unnecessarily.
    """
    return {
        "type": "ineq",
        "fun": lambda w: w @ expected_returns - target_return,
    }


def min_yield_constraint(
    yields: np.ndarray,
    min_yield: float,
) -> dict:
    """Portfolio income yield must be at least min_yield."""
    return {
        "type": "ineq",
        "fun": lambda w: w @ yields - min_yield,
    }


def max_expense_constraint(
    expense_ratios: np.ndarray,
    max_er: float,
) -> dict:
    """Weighted expense ratio must not exceed max_er."""
    return {
        "type": "ineq",
        "fun": lambda w: max_er - w @ expense_ratios,
    }


def min_diversification_constraint(
    cov_matrix: np.ndarray,
    min_div_ratio: float,
) -> dict:
    """Diversification ratio must be at least min_div_ratio."""
    asset_vols = np.sqrt(np.diag(cov_matrix))

    def div_constraint(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-10:
            return 0.0
        return (w @ asset_vols) / port_vol - min_div_ratio

    return {"type": "ineq", "fun": div_constraint}


def min_factor_exposure_constraint(
    factor_loadings_matrix: np.ndarray,
    factor_idx: int,
    min_exposure: float,
) -> dict:
    """Portfolio-level factor loading must be at least min_exposure.

    Args:
        factor_loadings_matrix: (n_assets x n_factors) matrix of per-asset betas.
        factor_idx: Column index of the target factor.
        min_exposure: Minimum weighted factor loading.
    """
    betas = factor_loadings_matrix[:, factor_idx]
    return {
        "type": "ineq",
        "fun": lambda w: w @ betas - min_exposure,
    }


def build_constraints(
    n_assets: int,
    cov_matrix: np.ndarray,
    expected_returns: Optional[np.ndarray] = None,
    yields: Optional[np.ndarray] = None,
    expense_ratios: Optional[np.ndarray] = None,
    target_return: Optional[float] = None,
    min_return: Optional[float] = None,
    max_vol: Optional[float] = None,
    min_yield: Optional[float] = None,
    max_er: Optional[float] = None,
    min_div_ratio: Optional[float] = None,
    factor_tilt_targets: Optional[dict[str, float]] = None,
    factor_loadings_matrix: Optional[np.ndarray] = None,
    factor_names: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build a complete constraint list from parameters.

    Always includes full investment. Adds optional constraints as specified.
    """
    constraints = [full_investment_constraint()]

    if target_return is not None and expected_returns is not None:
        constraints.append(target_return_constraint(expected_returns, target_return))
    elif min_return is not None and expected_returns is not None:
        constraints.append(min_return_constraint(expected_returns, min_return))

    if max_vol is not None:
        constraints.append(max_volatility_constraint(cov_matrix, max_vol))

    if min_yield is not None and yields is not None:
        constraints.append(min_yield_constraint(yields, min_yield))

    if max_er is not None and expense_ratios is not None:
        constraints.append(max_expense_constraint(expense_ratios, max_er))

    if min_div_ratio is not None:
        constraints.append(min_diversification_constraint(cov_matrix, min_div_ratio))

    if factor_tilt_targets and factor_loadings_matrix is not None and factor_names:
        for factor_name, min_exp in factor_tilt_targets.items():
            if factor_name in factor_names:
                idx = factor_names.index(factor_name)
                constraints.append(
                    min_factor_exposure_constraint(factor_loadings_matrix, idx, min_exp)
                )

    return constraints
