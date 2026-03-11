"""
Deterministic post-simulation portfolio breakdown report.

This is NOT an AI/LLM feature. All commentary is generated from
rule-based logic applied to portfolio analytics and simulation results.

Architecture:
    Each section is built by an independent function.
    generate_post_simulation_report() orchestrates all sections.
    The UI renders the structured dict output.
"""
import numpy as np
from typing import Any, Optional

from portfolio_lab.analytics.diagnostics import (
    generate_stress_flags,
    generate_key_takeaways,
)


def build_portfolio_summary(
    tickers: list[str],
    weights: dict[str, float],
    objective: str,
    expected_gross_return: float,
    expected_net_return: float,
    expected_volatility: float,
    sharpe: float,
    sortino: float,
    weighted_er: float,
    income_yield: float,
    diversification_score: float,
    factor_exposures: Optional[dict[str, float]],
    risk_free_rate: float,
    risk_free_source: str,
    return_method: str,
) -> dict[str, Any]:
    """Section 1: Portfolio Summary."""
    return {
        "section": "Portfolio Summary",
        "tickers": tickers,
        "weights": weights,
        "objective": objective,
        "return_estimation_method": return_method,
        "expected_gross_return": expected_gross_return,
        "expected_net_return": expected_net_return,
        "expected_volatility": expected_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "weighted_expense_ratio": weighted_er,
        "income_yield": income_yield,
        "diversification_score": diversification_score,
        "factor_exposures": factor_exposures,
        "risk_free_rate": risk_free_rate,
        "risk_free_source": risk_free_source,
    }


def build_simulation_summary(
    terminal_values: np.ndarray,
    real_terminal_values: np.ndarray,
    initial_investment: float,
    inflation_rate: float,
    target_value: Optional[float] = None,
    target_return: Optional[float] = None,
    horizon_years: int = 10,
) -> dict[str, Any]:
    """
    Section 2: Monte Carlo Outcome Summary.

    terminal_values: 1D array of nominal terminal portfolio values.
    real_terminal_values: 1D array of inflation-adjusted terminal values.
    """
    n = len(terminal_values)

    # Nominal statistics
    nom_mean = float(np.mean(terminal_values))
    nom_median = float(np.median(terminal_values))
    nom_p10 = float(np.percentile(terminal_values, 10))
    nom_p25 = float(np.percentile(terminal_values, 25))
    nom_p75 = float(np.percentile(terminal_values, 75))
    nom_p90 = float(np.percentile(terminal_values, 90))

    # Real statistics
    real_mean = float(np.mean(real_terminal_values))
    real_median = float(np.median(real_terminal_values))
    real_p10 = float(np.percentile(real_terminal_values, 10))
    real_p25 = float(np.percentile(real_terminal_values, 25))
    real_p75 = float(np.percentile(real_terminal_values, 75))
    real_p90 = float(np.percentile(real_terminal_values, 90))

    # Loss probabilities
    prob_nominal_loss = float(np.mean(terminal_values < initial_investment))
    prob_real_loss = float(np.mean(real_terminal_values < initial_investment))

    # Goal analysis
    goal_hit_rate = None
    goal_miss_rate = None
    if target_value is not None:
        goal_hit_rate = float(np.mean(terminal_values >= target_value))
        goal_miss_rate = 1.0 - goal_hit_rate

    # Target return analysis
    return_goal_hit_rate = None
    return_goal_miss_rate = None
    if target_return is not None:
        required_terminal = initial_investment * (1 + target_return) ** horizon_years
        return_goal_hit_rate = float(np.mean(terminal_values >= required_terminal))
        return_goal_miss_rate = 1.0 - return_goal_hit_rate

    return {
        "section": "Monte Carlo Outcome Summary",
        "n_simulations": n,
        "horizon_years": horizon_years,
        "initial_investment": initial_investment,
        "inflation_rate": inflation_rate,
        "nominal": {
            "mean": nom_mean,
            "median": nom_median,
            "p10": nom_p10,
            "p25": nom_p25,
            "p75": nom_p75,
            "p90": nom_p90,
        },
        "real": {
            "mean": real_mean,
            "median": real_median,
            "p10": real_p10,
            "p25": real_p25,
            "p75": real_p75,
            "p90": real_p90,
        },
        "prob_nominal_loss": prob_nominal_loss,
        "prob_real_loss": prob_real_loss,
        "target_value": target_value,
        "goal_hit_rate": goal_hit_rate,
        "goal_miss_rate": goal_miss_rate,
        "target_return": target_return,
        "return_goal_hit_rate": return_goal_hit_rate,
        "return_goal_miss_rate": return_goal_miss_rate,
    }


def build_return_driver_breakdown(
    tickers: list[str],
    weights: dict[str, float],
    expected_returns: dict[str, float],
    income_contributions: dict[str, float],
    factor_contributions: Optional[dict[str, float]],
    gross_return: float,
    net_return: float,
    expense_ratio: float,
    advisory_fee: float,
) -> dict[str, Any]:
    """Section 3: Return Driver Breakdown."""
    # Return contribution = weight * expected return
    return_contributions = {}
    for t in tickers:
        return_contributions[t] = weights.get(t, 0) * expected_returns.get(t, 0)

    # Sort by contribution
    sorted_contribs = sorted(return_contributions.items(), key=lambda x: x[1], reverse=True)
    biggest_drivers = sorted_contribs[:3]
    biggest_drags = sorted_contribs[-3:]

    fee_drag = gross_return - net_return

    return {
        "section": "Return Driver Breakdown",
        "return_contributions": return_contributions,
        "income_contributions": income_contributions,
        "factor_contributions": factor_contributions,
        "gross_return": gross_return,
        "net_return": net_return,
        "expense_ratio_drag": expense_ratio,
        "advisory_fee_drag": advisory_fee,
        "total_fee_drag": fee_drag,
        "biggest_upside_drivers": biggest_drivers,
        "biggest_return_drags": biggest_drags,
    }


def build_risk_driver_breakdown(
    tickers: list[str],
    weights: dict[str, float],
    component_risk: dict[str, float],
    marginal_risk: dict[str, float],
    risk_pct: dict[str, float],
    portfolio_volatility: float,
    correlation_observations: list[str],
) -> dict[str, Any]:
    """Section 4: Risk Driver Breakdown."""
    # Top 3 risk contributors
    sorted_risk = sorted(risk_pct.items(), key=lambda x: x[1], reverse=True)
    top3_risk = sorted_risk[:3]

    # Concentration warnings
    concentration_warnings = []
    for ticker, pct in sorted_risk:
        if pct > 0.35:
            concentration_warnings.append(
                f"{ticker} contributes {pct*100:.1f}% of portfolio risk"
            )

    return {
        "section": "Risk Driver Breakdown",
        "component_risk_contributions": component_risk,
        "marginal_risk_contributions": marginal_risk,
        "risk_percentage_contributions": risk_pct,
        "portfolio_volatility": portfolio_volatility,
        "top_3_risk_contributors": top3_risk,
        "concentration_warnings": concentration_warnings,
        "correlation_observations": correlation_observations,
    }


def build_diversification_summary(
    div_ratio: float,
    div_benefit: float,
    tickers: list[str],
    weights: dict[str, float],
    risk_pct: dict[str, float],
    correlation_matrix: dict,
) -> dict[str, Any]:
    """Section 5: Diversification Breakdown."""
    # Interpretation band
    if div_ratio < 1.2:
        quality = "Poor"
        interpretation = "The portfolio has minimal diversification benefit."
    elif div_ratio < 1.5:
        quality = "Moderate"
        interpretation = "The portfolio has some diversification benefit but could improve."
    elif div_ratio < 2.0:
        quality = "Good"
        interpretation = "The portfolio is well-diversified."
    else:
        quality = "Excellent"
        interpretation = "The portfolio achieves strong diversification."

    # Most diversifying: lowest risk contribution relative to weight
    diversifying_assets = []
    weak_diversifiers = []
    for t in tickers:
        w = weights.get(t, 0)
        rp = risk_pct.get(t, 0)
        if w > 0.01:
            ratio = rp / w
            if ratio < 0.8:
                diversifying_assets.append((t, ratio))
            elif ratio > 1.3:
                weak_diversifiers.append((t, ratio))

    diversifying_assets.sort(key=lambda x: x[1])
    weak_diversifiers.sort(key=lambda x: x[1], reverse=True)

    # Highly correlated pairs
    high_corr_pairs = []
    if isinstance(correlation_matrix, dict):
        for key, val in correlation_matrix.items():
            if isinstance(val, (int, float)) and val > 0.8:
                high_corr_pairs.append((key, val))

    return {
        "section": "Diversification Breakdown",
        "diversification_ratio": div_ratio,
        "diversification_benefit": div_benefit,
        "quality_band": quality,
        "interpretation": interpretation,
        "most_diversifying_assets": diversifying_assets[:5],
        "weakest_diversifiers": weak_diversifiers[:5],
        "high_correlation_pairs": high_corr_pairs,
    }


def build_inflation_summary(
    nominal_median_terminal: float,
    real_median_terminal: float,
    nominal_cagr: float,
    real_cagr: float,
    inflation_rate: float,
    horizon_years: int,
    initial_investment: float,
) -> dict[str, Any]:
    """Section 6: Inflation Impact Breakdown."""
    inflation_drag = nominal_median_terminal - real_median_terminal
    pp_reduction = 1.0 - (1.0 / (1 + inflation_rate) ** horizon_years)

    return {
        "section": "Inflation Impact Breakdown",
        "nominal_median_terminal": nominal_median_terminal,
        "real_median_terminal": real_median_terminal,
        "nominal_cagr": nominal_cagr,
        "real_cagr": real_cagr,
        "inflation_rate": inflation_rate,
        "horizon_years": horizon_years,
        "inflation_drag_dollars": inflation_drag,
        "purchasing_power_reduction_pct": pp_reduction,
    }


def build_goal_attainment_summary(
    terminal_values: np.ndarray,
    real_terminal_values: np.ndarray,
    initial_investment: float,
    goals: dict[str, float],
    horizon_years: int,
) -> dict[str, Any]:
    """
    Section 7: Goal Attainment and Probability Analysis.

    goals dict can include:
        'target_value': desired nominal terminal value
        'target_real_value': desired real terminal value
        'target_return': desired annualized return (used to compute terminal)
        'max_volatility': already checked at optimization time
        'stretch_target': aspirational target (e.g., 1.5x of target)
    """
    n = len(terminal_values)
    results = {}

    # Probability below starting principal
    results["prob_below_principal_nominal"] = float(
        np.mean(terminal_values < initial_investment)
    )
    results["prob_below_principal_real"] = float(
        np.mean(real_terminal_values < initial_investment)
    )

    # Target value goal
    if "target_value" in goals:
        tv = goals["target_value"]
        results["target_value"] = tv
        results["target_value_hit_rate"] = float(np.mean(terminal_values >= tv))
        results["target_value_miss_rate"] = 1.0 - results["target_value_hit_rate"]

    # Target real value
    if "target_real_value" in goals:
        trv = goals["target_real_value"]
        results["target_real_value"] = trv
        results["target_real_value_hit_rate"] = float(
            np.mean(real_terminal_values >= trv)
        )
        results["target_real_value_miss_rate"] = 1.0 - results["target_real_value_hit_rate"]

    # Target return
    if "target_return" in goals:
        tr = goals["target_return"]
        required = initial_investment * (1 + tr) ** horizon_years
        results["target_return"] = tr
        results["required_terminal_for_target_return"] = required
        results["target_return_hit_rate"] = float(np.mean(terminal_values >= required))
        results["target_return_miss_rate"] = 1.0 - results["target_return_hit_rate"]

    # Stretch target
    if "stretch_target" in goals:
        st = goals["stretch_target"]
        results["stretch_target"] = st
        results["stretch_target_hit_rate"] = float(np.mean(terminal_values >= st))

    return {
        "section": "Goal Attainment and Probability Analysis",
        "goals": goals,
        "initial_investment": initial_investment,
        "horizon_years": horizon_years,
        **results,
    }


def generate_post_simulation_report(
    # Portfolio inputs
    tickers: list[str],
    weights: dict[str, float],
    objective: str,
    return_method: str,
    # Analytics results
    expected_gross_return: float,
    expected_net_return: float,
    expected_volatility: float,
    sharpe: float,
    sortino: float,
    weighted_er: float,
    advisory_fee: float,
    income_yield: float,
    diversification_score: float,
    diversification_benefit_val: float,
    factor_exposures: Optional[dict[str, float]],
    risk_free_rate: float,
    risk_free_source: str,
    # Per-asset analytics
    expected_returns_per_asset: dict[str, float],
    income_contributions: dict[str, float],
    component_risk: dict[str, float],
    marginal_risk: dict[str, float],
    risk_pct: dict[str, float],
    correlation_observations: list[str],
    correlation_matrix_flat: dict,
    # Monte Carlo results
    terminal_values: np.ndarray,
    real_terminal_values: np.ndarray,
    initial_investment: float,
    inflation_rate: float,
    horizon_years: int,
    # Goals
    goals: dict[str, float],
) -> dict[str, Any]:
    """
    Master function: generate the complete post-simulation report.

    Returns a structured dict with all 9 report sections.
    """
    report = {}

    # 1. Portfolio Summary
    report["portfolio_summary"] = build_portfolio_summary(
        tickers=tickers,
        weights=weights,
        objective=objective,
        expected_gross_return=expected_gross_return,
        expected_net_return=expected_net_return,
        expected_volatility=expected_volatility,
        sharpe=sharpe,
        sortino=sortino,
        weighted_er=weighted_er,
        income_yield=income_yield,
        diversification_score=diversification_score,
        factor_exposures=factor_exposures,
        risk_free_rate=risk_free_rate,
        risk_free_source=risk_free_source,
        return_method=return_method,
    )

    # 2. Monte Carlo Summary
    target_value = goals.get("target_value")
    target_return = goals.get("target_return")
    report["simulation_summary"] = build_simulation_summary(
        terminal_values=terminal_values,
        real_terminal_values=real_terminal_values,
        initial_investment=initial_investment,
        inflation_rate=inflation_rate,
        target_value=target_value,
        target_return=target_return,
        horizon_years=horizon_years,
    )

    # 3. Return Drivers
    report["return_drivers"] = build_return_driver_breakdown(
        tickers=tickers,
        weights=weights,
        expected_returns=expected_returns_per_asset,
        income_contributions=income_contributions,
        factor_contributions=factor_exposures,
        gross_return=expected_gross_return,
        net_return=expected_net_return,
        expense_ratio=weighted_er,
        advisory_fee=advisory_fee,
    )

    # 4. Risk Drivers
    report["risk_drivers"] = build_risk_driver_breakdown(
        tickers=tickers,
        weights=weights,
        component_risk=component_risk,
        marginal_risk=marginal_risk,
        risk_pct=risk_pct,
        portfolio_volatility=expected_volatility,
        correlation_observations=correlation_observations,
    )

    # 5. Diversification
    report["diversification"] = build_diversification_summary(
        div_ratio=diversification_score,
        div_benefit=diversification_benefit_val,
        tickers=tickers,
        weights=weights,
        risk_pct=risk_pct,
        correlation_matrix=correlation_matrix_flat,
    )

    # 6. Inflation Impact
    nom_median = float(np.median(terminal_values))
    real_median = float(np.median(real_terminal_values))
    nom_cagr = (nom_median / initial_investment) ** (1.0 / horizon_years) - 1 if horizon_years > 0 else 0
    real_cagr = (real_median / initial_investment) ** (1.0 / horizon_years) - 1 if horizon_years > 0 else 0

    report["inflation_impact"] = build_inflation_summary(
        nominal_median_terminal=nom_median,
        real_median_terminal=real_median,
        nominal_cagr=nom_cagr,
        real_cagr=real_cagr,
        inflation_rate=inflation_rate,
        horizon_years=horizon_years,
        initial_investment=initial_investment,
    )

    # 7. Goal Attainment
    report["goal_attainment"] = build_goal_attainment_summary(
        terminal_values=terminal_values,
        real_terminal_values=real_terminal_values,
        initial_investment=initial_investment,
        goals=goals,
        horizon_years=horizon_years,
    )

    # 8. Stress Flags
    report["stress_flags"] = generate_stress_flags(report)

    # 9. Key Takeaways
    report["key_takeaways"] = generate_key_takeaways(report)

    return report
