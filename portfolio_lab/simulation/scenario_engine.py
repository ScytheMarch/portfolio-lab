"""
Scenario orchestration: ties together optimization, simulation, and reporting.
"""
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from portfolio_lab.analytics.returns import (
    simple_returns,
    arithmetic_mean_return,
    geometric_mean_return,
    exponentially_weighted_return,
)
from portfolio_lab.analytics.risk import (
    covariance_matrix,
    correlation_matrix,
    portfolio_volatility,
    component_contribution_to_risk,
    marginal_contribution_to_risk,
    risk_contribution_pct,
    diversification_ratio,
    diversification_benefit,
)
from portfolio_lab.analytics.performance import sharpe_ratio, sortino_ratio
from portfolio_lab.analytics.income import (
    portfolio_income_yield,
    weighted_expense_ratio,
    net_income_yield,
)
from portfolio_lab.analytics.factor_model import (
    estimate_factor_loadings,
    portfolio_factor_exposures,
    factor_implied_expected_return,
)
from portfolio_lab.analytics.reporting import generate_post_simulation_report
from portfolio_lab.data.fund_metadata import FundInfo
from portfolio_lab.optimization.solver import optimize_portfolio, OptimizationResult
from portfolio_lab.simulation.monte_carlo import MonteCarloEngine, MonteCarloResult

logger = logging.getLogger(__name__)

RETURN_METHODS = {
    "historical_arithmetic": "Historical Arithmetic Mean",
    "historical_geometric": "Historical Geometric Mean",
    "exponentially_weighted": "Exponentially Weighted Mean",
    "capm": "CAPM Estimate",
    "ff5_implied": "Fama-French 5-Factor Implied",
    "user_specified": "User-Specified Forward Returns",
    "blended": "Blended Estimate",
}


def estimate_expected_returns(
    returns: pd.DataFrame,
    method: str = "historical_arithmetic",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.05,
    factor_data: Optional[pd.DataFrame] = None,
    user_returns: Optional[dict[str, float]] = None,
    blend_weights: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], str]:
    """
    Estimate expected returns using the specified method.

    Returns:
        Tuple of (dict of ticker -> expected return, method_label).
    """
    tickers = returns.columns.tolist()
    label = RETURN_METHODS.get(method, method)

    if method == "historical_arithmetic":
        ann = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
        return dict(zip(tickers, ann.values if hasattr(ann, 'values') else [ann])), label

    elif method == "historical_geometric":
        geo = geometric_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
        return dict(zip(tickers, geo.values if hasattr(geo, 'values') else [geo])), label

    elif method == "exponentially_weighted":
        ewm = exponentially_weighted_return(
            returns, span=60, annualize=True, periods_per_year=periods_per_year
        )
        if isinstance(ewm, pd.Series):
            return dict(zip(tickers, ewm.values)), label
        return {tickers[0]: ewm}, label

    elif method == "capm":
        # CAPM: E[R_i] = RF + beta_i * (E[R_m] - RF)
        # Equity risk premium: use factor data if available, else default ~6%
        market_premium = 0.06
        if factor_data is not None and "Mkt-RF" in factor_data.columns:
            market_premium = float(factor_data["Mkt-RF"].mean()) * periods_per_year
        result = {}
        # Use equal-weighted portfolio as market proxy if no better option
        mkt_proxy = returns.mean(axis=1)
        var_mkt = mkt_proxy.var()
        for t in tickers:
            if var_mkt > 0:
                beta = returns[t].cov(mkt_proxy) / var_mkt
            else:
                beta = 1.0
            result[t] = risk_free_rate + beta * market_premium
        return result, label

    elif method == "ff5_implied":
        loadings = estimate_factor_loadings(returns, factor_data)
        result = {}
        for t in tickers:
            if loadings[t].factor_loadings:
                result[t] = factor_implied_expected_return(
                    loadings[t].factor_loadings,
                    risk_free_rate=risk_free_rate,
                )
            else:
                # Fallback to arithmetic mean
                fallback = arithmetic_mean_return(returns[t], annualize=True, periods_per_year=periods_per_year)
                result[t] = float(fallback)
                logger.warning(f"Factor regression failed for {t}; using arithmetic mean as fallback.")
        return result, label

    elif method == "user_specified" and user_returns:
        return user_returns, label

    elif method == "blended":
        # Blend multiple methods
        bw = blend_weights or {"historical_arithmetic": 0.5, "historical_geometric": 0.5}
        blended = {t: 0.0 for t in tickers}
        for sub_method, weight in bw.items():
            sub_rets, _ = estimate_expected_returns(
                returns, sub_method, periods_per_year, risk_free_rate, factor_data
            )
            for t in tickers:
                blended[t] += weight * sub_rets.get(t, 0)
        return blended, f"Blended ({', '.join(f'{m}:{w:.0%}' for m, w in bw.items())})"

    else:
        # Default fallback
        ann = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
        return dict(zip(tickers, ann.values if hasattr(ann, 'values') else [ann])), "Historical Arithmetic Mean (default)"


def run_full_scenario(
    prices: pd.DataFrame,
    fund_info: dict[str, FundInfo],
    # Optimization config
    objective: str = "max_sharpe",
    return_method: str = "historical_arithmetic",
    risk_free_rate: float = 0.05,
    risk_free_source: str = "manual",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: Optional[float] = None,
    max_vol: Optional[float] = None,
    min_yield: Optional[float] = None,
    # Monte Carlo config
    initial_investment: float = 100_000,
    horizon_years: int = 10,
    n_simulations: int = 5000,
    inflation_rate: float = 0.03,
    stochastic_inflation: bool = False,
    rebalance_frequency: str = "annual",
    annual_contribution: float = 0.0,
    annual_withdrawal: float = 0.0,
    # Goals
    target_value: Optional[float] = None,
    target_return_goal: Optional[float] = None,
    # Optional
    advisory_fee: float = 0.0,
    user_returns: Optional[dict[str, float]] = None,
    factor_data: Optional[pd.DataFrame] = None,
    mc_seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the complete scenario: estimate returns -> optimize -> simulate -> report.

    Returns:
        Dict with keys: optimization_result, mc_result, report, analytics.
    """
    tickers = prices.columns.tolist()
    n = len(tickers)

    # 1. Compute returns
    returns = simple_returns(prices)

    # 2. Estimate expected returns
    exp_returns_dict, return_label = estimate_expected_returns(
        returns, return_method, risk_free_rate=risk_free_rate,
        factor_data=factor_data, user_returns=user_returns,
    )
    exp_returns_arr = np.array([exp_returns_dict[t] for t in tickers])

    # 3. Covariance matrix
    cov = covariance_matrix(returns, annualize=True)
    cov_arr = cov.values

    # 4. Income and expense data
    yields_arr = np.array([
        (fund_info[t].dividend_yield or 0.0) if t in fund_info else 0.0
        for t in tickers
    ])
    er_arr = np.array([
        (fund_info[t].expense_ratio or 0.0) if t in fund_info else 0.0
        for t in tickers
    ])

    # 5. Optimize
    opt_result = optimize_portfolio(
        tickers=tickers,
        expected_returns=exp_returns_arr,
        cov_matrix=cov_arr,
        risk_free_rate=risk_free_rate,
        objective=objective,
        min_weight=min_weight,
        max_weight=max_weight,
        target_return=target_return,
        max_vol=max_vol,
        min_yield=min_yield,
        yields=yields_arr,
        expense_ratios=er_arr,
    )

    if not opt_result.success:
        return {
            "optimization_result": opt_result,
            "mc_result": None,
            "report": None,
            "error": opt_result.message,
        }

    weights = opt_result.raw_weights
    weights_dict = opt_result.weights

    # 6. Portfolio analytics
    port_vol = float(portfolio_volatility(weights, cov_arr))
    port_gross_ret = float(weights @ exp_returns_arr)
    port_er = float(weights @ er_arr)
    # NOTE: yfinance adjusted prices already reflect fund expense ratios (NAV is net
    # of ER). So historical return estimates from price data are already net of ER.
    # We only subtract the advisory fee here to avoid double-counting.
    # port_er is still tracked for reporting transparency.
    port_net_ret = port_gross_ret - advisory_fee

    # Income
    income_yd, income_contribs = portfolio_income_yield(fund_info, weights_dict)

    # Risk contributions
    ccr = component_contribution_to_risk(weights, cov_arr)
    mcr = marginal_contribution_to_risk(weights, cov_arr)
    rpct = risk_contribution_pct(weights, cov_arr)

    ccr_dict = {tickers[i]: float(ccr[i]) for i in range(n)}
    mcr_dict = {tickers[i]: float(mcr[i]) for i in range(n)}
    rpct_dict = {tickers[i]: float(rpct[i]) for i in range(n)}

    # Diversification
    div_ratio = diversification_ratio(weights, cov_arr)
    div_benefit_val = diversification_benefit(weights, cov_arr)

    # Correlation observations
    corr = correlation_matrix(returns)
    corr_obs = _extract_correlation_observations(corr, tickers)
    corr_flat = _flatten_correlation_matrix(corr)

    # Factor exposures (best effort)
    factor_exps = None
    try:
        loadings = estimate_factor_loadings(returns, factor_data)
        factor_exps = portfolio_factor_exposures(loadings, weights_dict)
    except Exception as e:
        logger.warning(f"Factor analysis skipped: {e}")

    # Sharpe and Sortino on historical returns for context
    port_hist_returns = (returns * weights).sum(axis=1)
    hist_sharpe = float(sharpe_ratio(port_hist_returns, risk_free_rate))
    hist_sortino = float(sortino_ratio(port_hist_returns, risk_free_rate))

    # 7. Monte Carlo
    mc = MonteCarloEngine(
        expected_return=port_net_ret,
        volatility=port_vol,
        initial_investment=initial_investment,
        horizon_years=horizon_years,
        n_simulations=n_simulations,
        inflation_rate=inflation_rate,
        stochastic_inflation=stochastic_inflation,
        rebalance_frequency=rebalance_frequency,
        annual_contribution=annual_contribution,
        annual_withdrawal=annual_withdrawal,
        seed=mc_seed,
    )
    mc_result = mc.run()
    mc_result.compute_stats(target_value=target_value)

    # 8. Build goals dict
    goals = {}
    if target_value is not None:
        goals["target_value"] = target_value
    if target_return_goal is not None:
        goals["target_return"] = target_return_goal

    # 9. Generate report
    report = generate_post_simulation_report(
        tickers=tickers,
        weights=weights_dict,
        objective=objective,
        return_method=return_label,
        expected_gross_return=port_gross_ret,
        expected_net_return=port_net_ret,
        expected_volatility=port_vol,
        sharpe=hist_sharpe,
        sortino=hist_sortino,
        weighted_er=port_er,
        advisory_fee=advisory_fee,
        income_yield=income_yd,
        diversification_score=div_ratio,
        diversification_benefit_val=div_benefit_val,
        factor_exposures=factor_exps,
        risk_free_rate=risk_free_rate,
        risk_free_source=risk_free_source,
        expected_returns_per_asset=exp_returns_dict,
        income_contributions=income_contribs,
        component_risk=ccr_dict,
        marginal_risk=mcr_dict,
        risk_pct=rpct_dict,
        correlation_observations=corr_obs,
        correlation_matrix_flat=corr_flat,
        terminal_values=mc_result.nominal_terminal,
        real_terminal_values=mc_result.real_terminal,
        initial_investment=initial_investment,
        inflation_rate=inflation_rate,
        horizon_years=horizon_years,
        goals=goals,
    )

    return {
        "optimization_result": opt_result,
        "mc_result": mc_result,
        "report": report,
        "analytics": {
            "returns": returns,
            "cov_matrix": cov,
            "correlation_matrix": corr,
            "expected_returns": exp_returns_dict,
            "return_method": return_label,
            "factor_loadings": factor_exps,
        },
    }


def _extract_correlation_observations(
    corr: pd.DataFrame, tickers: list[str], threshold: float = 0.7
) -> list[str]:
    """Extract noteworthy correlation observations."""
    obs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            val = corr.iloc[i, j]
            if abs(val) > threshold:
                direction = "positively" if val > 0 else "negatively"
                obs.append(
                    f"{tickers[i]} and {tickers[j]} are {direction} correlated ({val:.2f})"
                )
    return obs


def _flatten_correlation_matrix(corr: pd.DataFrame) -> dict:
    """Flatten correlation matrix to dict of 'A_B' -> correlation."""
    flat = {}
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            flat[f"{tickers[i]}_{tickers[j]}"] = float(corr.iloc[i, j])
    return flat
