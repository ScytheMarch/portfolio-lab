"""
Deterministic diagnostics: stress flags, warnings, and key takeaways.

All logic is rule-based. No LLM. No AI commentary.
Each rule has a clear threshold and a fixed message template.
"""
from typing import Any


def generate_stress_flags(report: dict[str, Any]) -> dict[str, Any]:
    """
    Section 8: Generate deterministic stress/warning flags.

    Examines the report sections and flags issues using fixed thresholds.
    """
    flags = []

    # --- Goal attainment flags ---
    sim = report.get("simulation_summary", {})
    goal_hit = sim.get("goal_hit_rate")
    if goal_hit is not None and goal_hit < 0.50:
        flags.append({
            "severity": "high",
            "category": "goal_attainment",
            "message": (
                f"Low goal hit rate: only {goal_hit*100:.1f}% of simulations "
                f"reach the target value. The goal may be unrealistic."
            ),
        })
    elif goal_hit is not None and goal_hit < 0.65:
        flags.append({
            "severity": "medium",
            "category": "goal_attainment",
            "message": (
                f"Moderate goal hit rate: {goal_hit*100:.1f}% of simulations "
                f"reach the target. Consider lowering target or increasing risk budget."
            ),
        })

    return_hit = sim.get("return_goal_hit_rate")
    if return_hit is not None and return_hit < 0.50:
        flags.append({
            "severity": "high",
            "category": "goal_attainment",
            "message": (
                f"Target return goal hit rate is only {return_hit*100:.1f}%. "
                f"The target return may be too aggressive for this portfolio."
            ),
        })

    # --- Inflation flags ---
    inflation = report.get("inflation_impact", {})
    nom_cagr = inflation.get("nominal_cagr", 0)
    real_cagr = inflation.get("real_cagr", 0)
    if nom_cagr > 0 and (nom_cagr - real_cagr) > 0.025:
        flags.append({
            "severity": "medium",
            "category": "inflation",
            "message": (
                f"Inflation materially erodes results: nominal CAGR {nom_cagr*100:.1f}% "
                f"vs real CAGR {real_cagr*100:.1f}% "
                f"(gap: {(nom_cagr - real_cagr)*100:.1f}% per year)."
            ),
        })

    pp_reduction = inflation.get("purchasing_power_reduction_pct", 0)
    if pp_reduction > 0.30:
        flags.append({
            "severity": "high",
            "category": "inflation",
            "message": (
                f"Purchasing power reduction of {pp_reduction*100:.0f}% over the horizon. "
                f"Real outcomes are significantly worse than nominal."
            ),
        })

    # --- Nominal/real loss probabilities ---
    prob_nom_loss = sim.get("prob_nominal_loss", 0)
    if prob_nom_loss > 0.20:
        flags.append({
            "severity": "high",
            "category": "loss_risk",
            "message": (
                f"{prob_nom_loss*100:.1f}% probability of nominal loss "
                f"(ending below initial investment)."
            ),
        })

    prob_real_loss = sim.get("prob_real_loss", 0)
    if prob_real_loss > 0.30:
        flags.append({
            "severity": "medium",
            "category": "loss_risk",
            "message": (
                f"{prob_real_loss*100:.1f}% probability of real loss "
                f"(ending below inflation-adjusted principal)."
            ),
        })

    # --- Concentration flags ---
    risk = report.get("risk_drivers", {})
    for warning in risk.get("concentration_warnings", []):
        flags.append({
            "severity": "medium",
            "category": "concentration",
            "message": warning,
        })

    # --- Diversification flags ---
    div = report.get("diversification", {})
    if div.get("quality_band") == "Poor":
        flags.append({
            "severity": "medium",
            "category": "diversification",
            "message": (
                f"Poor diversification (ratio: {div.get('diversification_ratio', 0):.2f}). "
                f"Consider adding uncorrelated assets."
            ),
        })

    # --- Fee flags ---
    portfolio = report.get("portfolio_summary", {})
    weighted_er = portfolio.get("weighted_expense_ratio", 0)
    if weighted_er > 0.005:  # > 50bps
        flags.append({
            "severity": "medium",
            "category": "fees",
            "message": (
                f"Weighted expense ratio of {weighted_er*100:.2f}% is above 50bps. "
                f"Fee drag may be material over the investment horizon."
            ),
        })

    ret = report.get("return_drivers", {})
    fee_drag = ret.get("total_fee_drag", 0)
    gross = ret.get("gross_return", 0)
    if gross > 0 and fee_drag / gross > 0.15:
        flags.append({
            "severity": "medium",
            "category": "fees",
            "message": (
                f"Fees consume {fee_drag/gross*100:.1f}% of gross expected return."
            ),
        })

    return {
        "section": "Stress Flags / Weaknesses",
        "flags": flags,
        "n_flags": len(flags),
        "high_severity_count": sum(1 for f in flags if f["severity"] == "high"),
        "medium_severity_count": sum(1 for f in flags if f["severity"] == "medium"),
    }


def generate_key_takeaways(report: dict[str, Any]) -> dict[str, Any]:
    """
    Section 9: Deterministic key takeaways.

    Rule-based analyst-style summary bullets.
    """
    takeaways = []

    # Diversification takeaway
    div = report.get("diversification", {})
    dr = div.get("diversification_ratio", 1.0)
    if dr >= 1.5:
        takeaways.append(
            f"Portfolio diversification is {div.get('quality_band', 'good').lower()} "
            f"(diversification ratio: {dr:.2f}). "
            f"The portfolio benefits from meaningful volatility reduction."
        )
    elif dr < 1.2:
        takeaways.append(
            f"Portfolio diversification is poor (ratio: {dr:.2f}). "
            f"Adding uncorrelated assets could meaningfully reduce risk."
        )

    # Concentration takeaway
    risk = report.get("risk_drivers", {})
    top3 = risk.get("top_3_risk_contributors", [])
    if top3 and len(top3) > 0:
        top_ticker, top_pct = top3[0]
        if top_pct > 0.35:
            takeaways.append(
                f"Risk is concentrated: {top_ticker} contributes "
                f"{top_pct*100:.1f}% of portfolio risk."
            )

    # Goal takeaway
    sim = report.get("simulation_summary", {})
    goal_hit = sim.get("goal_hit_rate")
    if goal_hit is not None:
        if goal_hit >= 0.80:
            takeaways.append(
                f"Goal attainment is strong: {goal_hit*100:.1f}% of simulations "
                f"reach the target."
            )
        elif goal_hit >= 0.60:
            takeaways.append(
                f"Goal attainment is moderate: {goal_hit*100:.1f}% of simulations "
                f"reach the target. There is meaningful miss risk."
            )
        else:
            takeaways.append(
                f"Goal attainment is weak: only {goal_hit*100:.1f}% of simulations "
                f"reach the target. The objective is not reliably met."
            )

    # Inflation takeaway
    inflation = report.get("inflation_impact", {})
    drag = inflation.get("inflation_drag_dollars", 0)
    initial = sim.get("initial_investment", 100000)
    if drag > initial * 0.10:
        takeaways.append(
            f"Inflation erodes ${drag:,.0f} of terminal wealth in median scenarios. "
            f"Real purchasing power is notably reduced."
        )

    # Fee takeaway
    ret = report.get("return_drivers", {})
    fee_drag = ret.get("total_fee_drag", 0)
    if fee_drag > 0.005:
        takeaways.append(
            f"Total fee drag of {fee_drag*100:.2f}% per year reduces compounded "
            f"growth over the investment horizon."
        )

    # Return method note
    portfolio = report.get("portfolio_summary", {})
    method = portfolio.get("return_estimation_method", "")
    if method:
        takeaways.append(
            f"Expected returns estimated using: {method}. "
            f"Net return after fees: {portfolio.get('expected_net_return', 0)*100:.2f}%."
        )

    return {
        "section": "Key Takeaways",
        "takeaways": takeaways,
    }
