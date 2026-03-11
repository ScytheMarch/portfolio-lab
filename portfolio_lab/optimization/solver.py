"""
Portfolio optimization solver.

Wraps scipy.optimize.minimize with structured input/output.
Supports multiple optimization objectives with modular constraints.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from portfolio_lab.optimization.objectives import (
    negative_sharpe,
    portfolio_volatility_obj,
    negative_income_yield,
    negative_net_income,
)
from portfolio_lab.optimization.constraints import (
    build_constraints,
    weight_bounds,
)

logger = logging.getLogger(__name__)

SUPPORTED_OBJECTIVES = [
    "max_sharpe",
    "min_volatility",
    "target_return",
    "max_income",
    "max_net_income",
    "custom",
]


@dataclass
class OptimizationResult:
    """Structured result from portfolio optimization."""
    success: bool
    objective: str
    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    income_yield: float = 0.0
    expense_ratio: float = 0.0
    message: str = ""
    raw_weights: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)


def optimize_portfolio(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.05,
    objective: str = "max_sharpe",
    # Weight bounds
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    per_asset_bounds: Optional[dict[int, tuple[float, float]]] = None,
    # Optional constraint parameters
    target_return: Optional[float] = None,
    max_vol: Optional[float] = None,
    min_return: Optional[float] = None,
    min_yield: Optional[float] = None,
    max_er: Optional[float] = None,
    min_div_ratio: Optional[float] = None,
    # Income data (for income objectives)
    yields: Optional[np.ndarray] = None,
    expense_ratios: Optional[np.ndarray] = None,
    # Solver options
    n_restarts: int = 10,
) -> OptimizationResult:
    """
    Run portfolio optimization.

    Args:
        tickers: List of ticker symbols.
        expected_returns: 1D array of annualized expected returns.
        cov_matrix: Annualized covariance matrix (n x n).
        risk_free_rate: Annualized risk-free rate.
        objective: One of SUPPORTED_OBJECTIVES.
        min_weight/max_weight: Global weight bounds.
        per_asset_bounds: Per-asset bound overrides.
        target_return: Required for 'target_return' objective.
        max_vol: Optional max volatility constraint.
        min_return: Optional minimum return constraint.
        yields/expense_ratios: Required for income objectives.
        n_restarts: Number of random restarts to avoid local optima.

    Returns:
        OptimizationResult with optimal weights and metrics.
    """
    n = len(tickers)
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    if mu.shape[0] != n or sigma.shape != (n, n):
        return OptimizationResult(
            success=False,
            objective=objective,
            message=f"Dimension mismatch: {n} tickers, returns shape {mu.shape}, cov shape {sigma.shape}",
        )

    # Build bounds and constraints
    bounds = weight_bounds(n, min_weight, max_weight, per_asset_bounds)

    _yields = np.asarray(yields, dtype=float) if yields is not None else np.zeros(n)
    _ers = np.asarray(expense_ratios, dtype=float) if expense_ratios is not None else np.zeros(n)

    constraint_target = target_return if objective == "target_return" else None
    constraints = build_constraints(
        n_assets=n,
        cov_matrix=sigma,
        expected_returns=mu,
        yields=_yields,
        expense_ratios=_ers,
        target_return=constraint_target,
        min_return=min_return if objective != "target_return" else None,
        max_vol=max_vol,
        min_yield=min_yield,
        max_er=max_er,
        min_div_ratio=min_div_ratio,
    )

    # Select objective function
    if objective == "max_sharpe":
        def obj_fn(w):
            return negative_sharpe(w, mu, sigma, risk_free_rate)
    elif objective in ("min_volatility", "target_return"):
        def obj_fn(w):
            return portfolio_volatility_obj(w, sigma)
    elif objective == "max_income":
        if yields is None:
            return OptimizationResult(
                success=False, objective=objective,
                message="Income objective requires yields data.",
            )
        def obj_fn(w):
            return negative_income_yield(w, _yields)
    elif objective == "max_net_income":
        if yields is None:
            return OptimizationResult(
                success=False, objective=objective,
                message="Net income objective requires yields and expense ratios.",
            )
        def obj_fn(w):
            return negative_net_income(w, _yields, _ers)
    else:
        return OptimizationResult(
            success=False, objective=objective,
            message=f"Unknown objective: {objective}. Supported: {SUPPORTED_OBJECTIVES}",
        )

    # Multi-start optimization
    best_result = None
    best_value = np.inf
    rng = np.random.default_rng(42)

    for i in range(n_restarts):
        if i == 0:
            # Start with equal weights
            w0 = np.ones(n) / n
        else:
            # Random Dirichlet starting point
            w0 = rng.dirichlet(np.ones(n))

        try:
            res = minimize(
                obj_fn,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

            if res.success and res.fun < best_value:
                best_value = res.fun
                best_result = res
        except Exception as e:
            logger.debug(f"Optimization restart {i} failed: {e}")
            continue

    if best_result is None or not best_result.success:
        return OptimizationResult(
            success=False,
            objective=objective,
            message=(
                f"Optimizer failed after {n_restarts} restarts. "
                f"Last message: {best_result.message if best_result else 'No feasible solution found.'}"
            ),
        )

    # Extract results
    w_opt = best_result.x
    # Clean up tiny weights
    w_opt = np.where(np.abs(w_opt) < 1e-6, 0, w_opt)
    # Renormalize to sum to 1
    w_sum = w_opt.sum()
    if w_sum < 1e-10:
        return OptimizationResult(
            success=False,
            objective=objective,
            message="Optimizer returned near-zero weights. Problem may be infeasible.",
        )
    w_opt = w_opt / w_sum

    port_ret = float(w_opt @ mu)
    port_vol = float(np.sqrt(w_opt @ sigma @ w_opt))
    port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    port_yield = float(w_opt @ _yields)
    port_er = float(w_opt @ _ers)

    weights_dict = {tickers[i]: float(w_opt[i]) for i in range(n)}

    warnings = []
    # Check for near-boundary weights
    for i, t in enumerate(tickers):
        if w_opt[i] >= max_weight - 1e-4:
            warnings.append(f"{t} is at maximum weight ({max_weight*100:.0f}%)")
        if w_opt[i] <= min_weight + 1e-4 and min_weight > 0:
            warnings.append(f"{t} is at minimum weight ({min_weight*100:.0f}%)")

    return OptimizationResult(
        success=True,
        objective=objective,
        weights=weights_dict,
        expected_return=port_ret,
        volatility=port_vol,
        sharpe_ratio=port_sharpe,
        income_yield=port_yield,
        expense_ratio=port_er,
        message=best_result.message,
        raw_weights=w_opt,
        warnings=warnings,
    )
