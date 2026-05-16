"""
Streamlit page layouts for portfolio_lab.

Renders structured report data from the analytics/reporting engine.
No analytics logic here — pages only display.
"""
import streamlit as st
import numpy as np
import pandas as pd
from typing import Any, Optional

from portfolio_lab.ui import charts


def render_report_section(title: str, data: dict, key_prefix: str = ""):
    """Generic renderer for a report section."""
    st.markdown(f"### {title}")


def render_portfolio_summary(summary: dict[str, Any]):
    """Render Section 1: Portfolio Summary."""
    st.markdown("### 1. Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Gross Return", f"{summary['expected_gross_return']*100:.2f}%")
        st.metric("Expected Net Return", f"{summary['expected_net_return']*100:.2f}%")
    with col2:
        st.metric("Expected Volatility", f"{summary['expected_volatility']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    with col3:
        st.metric("Sortino Ratio", f"{summary['sortino_ratio']:.2f}")
        st.metric("Income Yield", f"{summary['income_yield']*100:.2f}%")
    with col4:
        st.metric("Expense Ratio", f"{summary['weighted_expense_ratio']*100:.3f}%")
        st.metric("Diversification Score", f"{summary['diversification_score']:.2f}")

    st.caption(
        f"Objective: **{summary['objective']}** | "
        f"Return method: **{summary['return_estimation_method']}** | "
        f"Risk-free: {summary['risk_free_rate']*100:.2f}% ({summary['risk_free_source']})"
    )

    # Weights table
    weights = summary["weights"]
    weights_df = pd.DataFrame({
        "Ticker": list(weights.keys()),
        "Weight": [f"{v*100:.2f}%" for v in weights.values()],
    })
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    # Factor exposures
    if summary.get("factor_exposures"):
        st.markdown("**Factor Exposures:**")
        fe = summary["factor_exposures"]
        fe_df = pd.DataFrame({
            "Factor": list(fe.keys()),
            "Loading": [f"{v:.3f}" for v in fe.values()],
        })
        st.dataframe(fe_df, hide_index=True, use_container_width=True)


def render_simulation_summary(sim: dict[str, Any]):
    """Render Section 2: Monte Carlo Outcome Summary."""
    st.markdown("### 2. Monte Carlo Outcome Summary")
    st.caption(
        f"{sim['n_simulations']:,} simulations | "
        f"{sim['horizon_years']}-year horizon | "
        f"Initial: ${sim['initial_investment']:,.0f} | "
        f"Inflation: {sim['inflation_rate']*100:.1f}%"
    )

    # Goal hit rate — prominent display
    if sim.get("goal_hit_rate") is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Goal Hit Rate: {sim['goal_hit_rate']*100:.1f}%** of simulations")
        with col2:
            st.error(f"**Goal Miss Rate: {sim['goal_miss_rate']*100:.1f}%** of simulations")

    if sim.get("return_goal_hit_rate") is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Return Goal Hit: {sim['return_goal_hit_rate']*100:.1f}%**")
        with col2:
            st.error(f"**Return Goal Miss: {sim['return_goal_miss_rate']*100:.1f}%**")

    # Terminal value table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nominal Terminal Values**")
        nom = sim["nominal"]
        _render_percentile_table(nom, "$")

    with col2:
        st.markdown("**Real (Inflation-Adjusted) Terminal Values**")
        real = sim["real"]
        _render_percentile_table(real, "$")

    # Loss probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prob. of Nominal Loss", f"{sim['prob_nominal_loss']*100:.1f}%")
    with col2:
        st.metric("Prob. of Real Loss", f"{sim['prob_real_loss']*100:.1f}%")


def _render_percentile_table(stats: dict, prefix: str = "$"):
    """Render a percentile stats table."""
    rows = [
        ("Mean", stats["mean"]),
        ("Median", stats["median"]),
        ("10th Percentile", stats["p10"]),
        ("25th Percentile", stats["p25"]),
        ("75th Percentile", stats["p75"]),
        ("90th Percentile", stats["p90"]),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df["Value"] = df["Value"].apply(lambda x: f"{prefix}{x:,.0f}")
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_return_drivers(drivers: dict[str, Any]):
    """Render Section 3: Return Driver Breakdown."""
    st.markdown("### 3. Return Driver Breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gross Return", f"{drivers['gross_return']*100:.2f}%")
    with col2:
        st.metric("Net Return", f"{drivers['net_return']*100:.2f}%")
    with col3:
        st.metric("Fee Drag", f"{drivers['total_fee_drag']*100:.2f}%")

    # Contribution table
    contribs = drivers["return_contributions"]
    df = pd.DataFrame({
        "Ticker": list(contribs.keys()),
        "Return Contribution": [f"{v*100:.2f}%" for v in contribs.values()],
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Top drivers
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Upside Drivers:**")
        for ticker, val in drivers["biggest_upside_drivers"]:
            st.write(f"- {ticker}: {val*100:.2f}%")
    with col2:
        st.markdown("**Biggest Return Drags:**")
        for ticker, val in drivers["biggest_return_drags"]:
            st.write(f"- {ticker}: {val*100:.2f}%")


def render_risk_drivers(risk: dict[str, Any]):
    """Render Section 4: Risk Driver Breakdown."""
    st.markdown("### 4. Risk Driver Breakdown")
    st.metric("Portfolio Volatility", f"{risk['portfolio_volatility']*100:.2f}%")

    # Risk contribution table
    rpct = risk["risk_percentage_contributions"]
    df = pd.DataFrame({
        "Ticker": list(rpct.keys()),
        "Risk Contribution (%)": [f"{v*100:.1f}%" for v in rpct.values()],
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    if risk["concentration_warnings"]:
        for w in risk["concentration_warnings"]:
            st.warning(w)

    if risk["correlation_observations"]:
        n_obs = len(risk["correlation_observations"])
        with st.expander(f"Correlation Notes ({n_obs} entries)"):
            for obs in risk["correlation_observations"]:
                st.write(f"- {obs}")


def render_diversification(div: dict[str, Any]):
    """Render Section 5: Diversification Breakdown."""
    st.markdown("### 5. Diversification Breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diversification Ratio", f"{div['diversification_ratio']:.2f}")
    with col2:
        st.metric("Quality", div["quality_band"])
    with col3:
        st.metric("Benefit", f"{div['diversification_benefit']*100:.1f}%")

    st.info(div["interpretation"])

    col1, col2 = st.columns(2)
    with col1:
        if div["most_diversifying_assets"]:
            st.markdown("**Most Diversifying Assets:**")
            for ticker, ratio in div["most_diversifying_assets"]:
                st.write(f"- {ticker} (risk/weight ratio: {ratio:.2f})")
    with col2:
        if div["weakest_diversifiers"]:
            st.markdown("**Weakest Diversifiers:**")
            for ticker, ratio in div["weakest_diversifiers"]:
                st.write(f"- {ticker} (risk/weight ratio: {ratio:.2f})")


def render_inflation_impact(infl: dict[str, Any]):
    """Render Section 6: Inflation Impact."""
    st.markdown("### 6. Inflation Impact Breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nominal Median Terminal", f"${infl['nominal_median_terminal']:,.0f}")
        st.metric("Nominal CAGR", f"{infl['nominal_cagr']*100:.2f}%")
    with col2:
        st.metric("Real Median Terminal", f"${infl['real_median_terminal']:,.0f}")
        st.metric("Real CAGR", f"{infl['real_cagr']*100:.2f}%")
    with col3:
        st.metric("Inflation Drag ($)", f"${infl['inflation_drag_dollars']:,.0f}")
        st.metric("Purchasing Power Loss", f"{infl['purchasing_power_reduction_pct']*100:.1f}%")


def render_goal_attainment(goals: dict[str, Any]):
    """Render Section 7: Goal Attainment."""
    st.markdown("### 7. Goal Attainment & Probability Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Prob. Below Starting Principal (Nominal)",
            f"{goals.get('prob_below_principal_nominal', 0)*100:.1f}%",
        )
    with col2:
        st.metric(
            "Prob. Below Starting Principal (Real)",
            f"{goals.get('prob_below_principal_real', 0)*100:.1f}%",
        )

    if "target_value_hit_rate" in goals:
        st.success(
            f"**Target Value (${goals['target_value']:,.0f}):** "
            f"Hit rate = {goals['target_value_hit_rate']*100:.1f}% | "
            f"Miss rate = {goals['target_value_miss_rate']*100:.1f}%"
        )

    if "target_return_hit_rate" in goals:
        st.success(
            f"**Target Return ({goals['target_return']*100:.1f}%):** "
            f"Hit rate = {goals['target_return_hit_rate']*100:.1f}% | "
            f"Miss rate = {goals['target_return_miss_rate']*100:.1f}%"
        )

    if "stretch_target_hit_rate" in goals:
        st.info(
            f"**Stretch Target (${goals['stretch_target']:,.0f}):** "
            f"Hit rate = {goals['stretch_target_hit_rate']*100:.1f}%"
        )


def render_stress_flags(flags_data: dict[str, Any]):
    """Render Section 8: Stress Flags."""
    st.markdown("### 8. Stress Flags / Weaknesses")

    flags = flags_data.get("flags", [])
    if not flags:
        st.success("No stress flags detected.")
        return

    st.warning(
        f"**{flags_data['n_flags']} flags:** "
        f"{flags_data['high_severity_count']} high severity, "
        f"{flags_data['medium_severity_count']} medium severity"
    )

    for f in flags:
        if f["severity"] == "high":
            st.error(f"**[{f['category'].upper()}]** {f['message']}")
        else:
            st.warning(f"**[{f['category'].upper()}]** {f['message']}")


def render_key_takeaways(takeaways_data: dict[str, Any]):
    """Render Section 9: Key Takeaways."""
    st.markdown("### 9. Key Takeaways")
    for t in takeaways_data.get("takeaways", []):
        st.markdown(f"- {t}")


def render_full_report(report: dict[str, Any]):
    """Render the complete post-simulation report."""
    st.header("Post-Simulation Portfolio Breakdown Report")
    st.markdown("---")

    render_portfolio_summary(report["portfolio_summary"])
    st.markdown("---")

    render_simulation_summary(report["simulation_summary"])
    st.markdown("---")

    render_return_drivers(report["return_drivers"])
    st.markdown("---")

    render_risk_drivers(report["risk_drivers"])
    st.markdown("---")

    render_diversification(report["diversification"])
    st.markdown("---")

    render_inflation_impact(report["inflation_impact"])
    st.markdown("---")

    render_goal_attainment(report["goal_attainment"])
    st.markdown("---")

    render_stress_flags(report["stress_flags"])
    st.markdown("---")

    render_key_takeaways(report["key_takeaways"])
