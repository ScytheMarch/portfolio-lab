"""Analysis page — runs the pipeline and displays results as an integrated dashboard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from portfolio_lab.ui.styles import (
    gradient_header, glass_card, section_header, metric_badge,
    ticker_chips_html, quality_color, GREEN, YELLOW, RED, TEXT_SECONDARY,
)
from portfolio_lab.ui.definitions import explain, why_it_matters

logger = logging.getLogger(__name__)


def render() -> None:
    st.markdown(gradient_header("Analysis Dashboard", "📊"), unsafe_allow_html=True)

    tickers = st.session_state.get("tickers", [])
    if not tickers:
        st.info(
            "No portfolio selected. Go to **Portfolio Builder** to choose your assets, "
            "then click **Run Analysis**."
        )
        return

    # Check if analysis should run
    trigger = st.session_state.pop("trigger_analysis", False)
    has_results = "analysis_result" in st.session_state

    if not trigger and not has_results:
        st.info(
            f"Portfolio has **{len(tickers)}** assets ready. "
            f"Click **Run Analysis** in the top bar to start."
        )
        st.markdown(ticker_chips_html(tickers), unsafe_allow_html=True)
        return

    if trigger:
        # Clear stale results before running new analysis
        st.session_state.pop("analysis_result", None)
        _run_analysis_pipeline(tickers)

    if "analysis_result" not in st.session_state:
        return

    result = st.session_state["analysis_result"]
    opt_res = result["optimization_result"]
    mc_res = result["mc_result"]
    report = result["report"]
    analytics = result["analytics"]

    # Show optimizer warnings (e.g., fallback messages)
    if opt_res.warnings:
        for w in opt_res.warnings:
            st.warning(w)

    # ── Row 1: Portfolio Summary Metrics ───────────────────────────────
    _render_summary_metrics(report["portfolio_summary"])

    st.markdown("---")

    # ── Row 2: Allocation + Monte Carlo ───────────────────────────────
    col_alloc, col_mc = st.columns(2)

    with col_alloc:
        st.markdown(section_header("Asset Allocation"), unsafe_allow_html=True)
        from portfolio_lab.ui.charts import allocation_pie_chart
        st.plotly_chart(allocation_pie_chart(opt_res.weights), use_container_width=True)

    with col_mc:
        st.markdown(section_header("Monte Carlo Projection"), unsafe_allow_html=True)
        from portfolio_lab.ui.charts import monte_carlo_fan_chart
        mc_settings = st.session_state.get("settings_mc", {})
        st.plotly_chart(
            monte_carlo_fan_chart(
                mc_res.nominal_paths,
                mc_settings.get("horizon_years", 10),
                mc_settings.get("initial_investment", 100_000),
            ),
            use_container_width=True,
        )
        # Loss / goal badges
        sim = report["simulation_summary"]
        badges = metric_badge("Prob. of Loss", f"{sim['prob_nominal_loss']*100:.1f}%",
                              RED if sim['prob_nominal_loss'] > 0.1 else GREEN)
        if sim.get("goal_hit_rate") is not None:
            hit_color = GREEN if sim["goal_hit_rate"] > 0.7 else YELLOW if sim["goal_hit_rate"] > 0.4 else RED
            badges += metric_badge("Goal Hit Rate", f"{sim['goal_hit_rate']*100:.0f}%", hit_color)
        st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 3: Key Takeaways ──────────────────────────────────────────
    st.markdown(section_header("Key Takeaways"), unsafe_allow_html=True)
    takeaways = report.get("key_takeaways", {}).get("takeaways", [])
    if takeaways:
        for t in takeaways:
            st.markdown(f"- {t}")
    else:
        st.caption("No takeaways generated.")

    st.markdown("---")

    # ── Row 4: Return Drivers ─────────────────────────────────────────
    _render_return_drivers(report["return_drivers"])

    st.markdown("---")

    # ── Row 5: Risk + Diversification ─────────────────────────────────
    col_risk, col_div = st.columns(2)

    with col_risk:
        _render_risk_drivers(report["risk_drivers"])

    with col_div:
        _render_diversification(report["diversification"])

    st.markdown("---")

    # ── Row 6: Terminal Distribution + Inflation ──────────────────────
    col_term, col_infl = st.columns(2)

    with col_term:
        st.markdown(section_header("Terminal Value Distribution"), unsafe_allow_html=True)
        from portfolio_lab.ui.charts import terminal_value_distribution_chart
        goal_settings = st.session_state.get("settings_goals", {})
        st.plotly_chart(
            terminal_value_distribution_chart(
                mc_res.nominal_terminal,
                mc_settings.get("initial_investment", 100_000),
                target_value=goal_settings.get("target_value"),
            ),
            use_container_width=True,
        )

    with col_infl:
        _render_inflation_impact(report["inflation_impact"])

    st.markdown("---")

    # ── Row 7: Factor Analysis (tabbed) ───────────────────────────────
    if "factor_analysis" in report:
        _render_factor_analysis(report["factor_analysis"], analytics, opt_res)

        st.markdown("---")

    # ── Row 8: Stress Flags ───────────────────────────────────────────
    _render_stress_flags(report["stress_flags"])

    # ── Row 9: Additional Charts ──────────────────────────────────────
    with st.expander("More Charts", expanded=False):
        _render_additional_charts(analytics, opt_res, report, tickers)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_analysis_pipeline(tickers: list[str]):
    """Execute the full analysis pipeline and cache results."""
    from portfolio_lab.data.market_data import fetch_price_data, DataFetchError
    from portfolio_lab.data.fund_metadata import fetch_fund_metadata
    from portfolio_lab.data.treasury_data import fetch_risk_free_rate
    from portfolio_lab.data.factor_data import fetch_ff5_factors
    from portfolio_lab.simulation.scenario_engine import run_full_scenario

    data_settings = st.session_state.get("settings_data", {"lookback_years": 5, "rf_override": None})
    opt_settings = st.session_state.get("settings_opt", {
        "objective": "max_sharpe", "return_method": "historical_geometric",
        "min_weight": 0.01, "max_weight": 0.15,
    })
    mc_settings = st.session_state.get("settings_mc", {
        "initial_investment": 100_000, "n_simulations": 5_000, "horizon_years": 10,
        "rebalance_frequency": "annual", "inflation_rate": 0.03, "stochastic_inflation": False,
        "annual_contribution": 0, "annual_withdrawal": 0,
    })
    goal_settings = st.session_state.get("settings_goals", {})
    fee_settings = st.session_state.get("settings_fees", {"advisory_fee": 0})
    factor_tilts = st.session_state.get("settings_factor_tilts", {})
    scenario_settings = st.session_state.get("settings_scenarios", {})

    # Fetch data
    with st.spinner("Fetching market data..."):
        try:
            prices = fetch_price_data(tickers, lookback_years=data_settings["lookback_years"])
            tickers = prices.columns.tolist()
            st.session_state["tickers"] = tickers
        except DataFetchError as e:
            st.error(f"Data fetch failed: {e}")
            return

    with st.spinner("Fetching fund metadata..."):
        fund_info = fetch_fund_metadata(tickers)
        # Condense warnings
        missing_er = []
        missing_yield = []
        for t, info in fund_info.items():
            for w in info.warnings:
                if "Expense ratio not available" in w:
                    missing_er.append(t)
                elif "Dividend yield not available" in w:
                    missing_yield.append(t)
        if missing_er:
            with st.expander(f"Expense ratio unavailable for {len(missing_er)} ticker(s)"):
                st.write(", ".join(missing_er))
                st.caption("Manual input recommended. Do NOT assume zero for funds/ETFs.")
        if missing_yield:
            with st.expander(f"Dividend yield unavailable for {len(missing_yield)} ticker(s)"):
                st.write(", ".join(missing_yield))
                st.caption("Treated as 0% yield in income calculations.")

    with st.spinner("Fetching risk-free rate..."):
        rf_rate, rf_source = fetch_risk_free_rate(data_settings.get("rf_override"))

    factor_data = None
    with st.spinner("Fetching Fama-French factor data..."):
        factor_data = fetch_ff5_factors(frequency="daily")
        if factor_data is None:
            st.warning("Factor data unavailable. Factor analysis features will be limited.")
            if opt_settings.get("return_method") == "ff5_implied":
                opt_settings["return_method"] = "historical_geometric"

    with st.spinner("Running optimization and simulation..."):
        result = run_full_scenario(
            prices=prices,
            fund_info=fund_info,
            objective=opt_settings["objective"],
            return_method=opt_settings.get("return_method", "historical_geometric"),
            risk_free_rate=rf_rate,
            risk_free_source=rf_source,
            min_weight=opt_settings.get("min_weight", 0.01),
            max_weight=opt_settings.get("max_weight", 0.15),
            target_return=opt_settings.get("target_return"),
            max_vol=opt_settings.get("max_vol"),
            min_yield=opt_settings.get("min_yield"),
            initial_investment=mc_settings["initial_investment"],
            horizon_years=mc_settings["horizon_years"],
            n_simulations=mc_settings["n_simulations"],
            inflation_rate=mc_settings["inflation_rate"],
            stochastic_inflation=mc_settings["stochastic_inflation"],
            rebalance_frequency=mc_settings["rebalance_frequency"],
            annual_contribution=mc_settings["annual_contribution"],
            annual_withdrawal=mc_settings["annual_withdrawal"],
            target_value=goal_settings.get("target_value"),
            target_return_goal=goal_settings.get("target_return_goal"),
            advisory_fee=fee_settings.get("advisory_fee", 0),
            factor_data=factor_data,
            mc_seed=42,
            factor_tilt_targets=factor_tilts if factor_tilts else None,
            scenario_premia=scenario_settings.get("scenario_premia"),
        )

    if result.get("error"):
        st.error(f"Optimization failed: {result['error']}")
        opt = result.get("optimization_result")
        if opt and opt.warnings:
            for w in opt.warnings:
                st.warning(w)
        return

    st.session_state["analysis_result"] = result
    st.session_state["analysis_rf_rate"] = rf_rate
    st.session_state["analysis_tickers"] = tickers
    st.success(f"Analysis complete — {len(tickers)} assets, {mc_settings['n_simulations']:,} simulations.")


# ═══════════════════════════════════════════════════════════════════════════
# Section renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_summary_metrics(summary: dict):
    st.markdown(section_header("Portfolio Summary"), unsafe_allow_html=True)
    st.caption(
        f"Objective: **{summary['objective']}** · "
        f"Return method: **{summary['return_estimation_method']}** · "
        f"Risk-free: {summary['risk_free_rate']*100:.2f}% ({summary['risk_free_source']})"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = summary["expected_gross_return"]
        st.metric("Expected Return", f"{val*100:.2f}%")
        st.caption(f"Net of fees: {summary['expected_net_return']*100:.2f}%")
    with c2:
        st.metric("Volatility", f"{summary['expected_volatility']*100:.2f}%")
        st.caption(explain("volatility"))
    with c3:
        sr = summary["sharpe_ratio"]
        st.metric("Sharpe Ratio", f"{sr:.2f}")
        label = "Excellent" if sr > 2 else "Good" if sr > 1 else "Adequate" if sr > 0.5 else "Below Average"
        color = GREEN if sr > 1 else YELLOW if sr > 0.5 else RED
        st.markdown(metric_badge(label, f"{sr:.2f}", color), unsafe_allow_html=True)
    with c4:
        st.metric("Income Yield", f"{summary['income_yield']*100:.2f}%")
        st.caption(f"Expense ratio: {summary['weighted_expense_ratio']*100:.3f}%")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sortino Ratio", f"{summary['sortino_ratio']:.2f}")
    with c2:
        dr = summary["diversification_score"]
        st.metric("Diversification", f"{dr:.2f}")
        label = "Strong" if dr > 1.5 else "Moderate" if dr > 1.2 else "Low"
        color = GREEN if dr > 1.5 else YELLOW if dr > 1.2 else RED
        st.markdown(metric_badge(label, "", color), unsafe_allow_html=True)
    with c3:
        st.metric("Assets", f"{len(summary['weights'])}")
    with c4:
        # Factor exposures summary
        fe = summary.get("factor_exposures", {})
        if fe:
            mkt = fe.get("Mkt-RF", 0)
            st.metric("Market Beta", f"{mkt:.2f}")


def _render_return_drivers(drivers: dict):
    st.markdown(section_header("Return Drivers"), unsafe_allow_html=True)
    st.caption(
        "Where your portfolio's expected return comes from. "
        "Return contribution = weight x expected return for each asset."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Gross Return", f"{drivers['gross_return']*100:.2f}%")
    with c2:
        st.metric("Net Return", f"{drivers['net_return']*100:.2f}%")
    with c3:
        st.metric("Fee Drag", f"{drivers['total_fee_drag']*100:.2f}%")

    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("**Top Upside Drivers**")
        for ticker, val in drivers["biggest_upside_drivers"]:
            color = GREEN
            st.markdown(
                f'<span style="color:{color};font-weight:600">{ticker}</span> '
                f'<span style="color:{TEXT_SECONDARY}">+{val*100:.2f}%</span>',
                unsafe_allow_html=True,
            )
    with col_down:
        st.markdown("**Biggest Drags**")
        for ticker, val in drivers["biggest_return_drags"]:
            st.markdown(
                f'<span style="color:{YELLOW};font-weight:600">{ticker}</span> '
                f'<span style="color:{TEXT_SECONDARY}">{val*100:.2f}%</span>',
                unsafe_allow_html=True,
            )


def _render_risk_drivers(risk: dict):
    st.markdown(section_header("Risk Analysis"), unsafe_allow_html=True)
    st.metric("Portfolio Volatility", f"{risk['portfolio_volatility']*100:.2f}%")
    st.caption(explain("volatility"))

    from portfolio_lab.ui.charts import risk_contribution_chart
    st.plotly_chart(risk_contribution_chart(risk["risk_percentage_contributions"]), use_container_width=True)

    if risk["concentration_warnings"]:
        for w in risk["concentration_warnings"]:
            st.warning(w)

    if risk["correlation_observations"]:
        n = len(risk["correlation_observations"])
        with st.expander(f"Correlation Notes ({n} entries)"):
            for obs in risk["correlation_observations"]:
                st.write(f"- {obs}")


def _render_diversification(div: dict):
    st.markdown(section_header("Diversification"), unsafe_allow_html=True)
    st.caption(explain("diversification_ratio"))

    c1, c2 = st.columns(2)
    with c1:
        dr = div["diversification_ratio"]
        st.metric("Diversification Ratio", f"{dr:.2f}")
        color = GREEN if dr > 1.5 else YELLOW if dr > 1.2 else RED
        st.markdown(metric_badge(div["quality_band"], f"{div['diversification_benefit']*100:.0f}% benefit", color),
                    unsafe_allow_html=True)
    with c2:
        st.info(div["interpretation"])


def _render_inflation_impact(infl: dict):
    st.markdown(section_header("Inflation Impact"), unsafe_allow_html=True)
    st.caption("How inflation erodes your portfolio's purchasing power over the projection horizon.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Nominal Median Terminal", f"${infl['nominal_median_terminal']:,.0f}")
        st.metric("Real Median Terminal", f"${infl['real_median_terminal']:,.0f}")
    with c2:
        st.metric("Inflation Drag", f"${infl['inflation_drag_dollars']:,.0f}")
        st.metric("Purchasing Power Loss", f"{infl['purchasing_power_reduction_pct']*100:.1f}%")


def _render_factor_analysis(factor_data: dict, analytics: dict, opt_res):
    st.markdown(section_header("Factor Analysis"), unsafe_allow_html=True)
    st.caption(
        "Fama-French 5-Factor decomposition of your portfolio's returns. " +
        explain("fama_french")
    )

    tab_reg, tab_attr, tab_scenario, tab_history = st.tabs([
        "Regression", "Attribution", "Scenarios", "Factor History"
    ])

    with tab_reg:
        _render_factor_regression(factor_data)

    with tab_attr:
        attrib = factor_data.get("factor_attribution")
        if attrib:
            st.caption("How much of the portfolio's historical return came from each factor.")
            from portfolio_lab.ui.charts import factor_attribution_chart
            st.plotly_chart(factor_attribution_chart(attrib), use_container_width=True)

            rows = [{"Source": f, "Contribution": f"{v*100:+.2f}%"} for f, v in attrib.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Factor attribution not available.")

    with tab_scenario:
        scenarios = factor_data.get("scenario_results")
        if scenarios:
            st.caption("How your portfolio would perform under different factor environments.")
            from portfolio_lab.ui.charts import scenario_comparison_chart
            st.plotly_chart(
                scenario_comparison_chart(scenarios, base_return=opt_res.expected_return),
                use_container_width=True,
            )
            rows = [{"Scenario": n, "Implied Return": f"{r*100:.2f}%"} for n, r in scenarios.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Configure factor scenarios in the Builder to see results here.")

    with tab_history:
        factor_cum = analytics.get("factor_cumulative_returns")
        factor_corr = analytics.get("factor_correlation")

        if factor_cum is not None:
            st.caption(
                "Cumulative growth of $1 invested in each factor. "
                "Shows which factors have been rewarded historically."
            )
            from portfolio_lab.ui.charts import factor_premium_history_chart
            st.plotly_chart(factor_premium_history_chart(factor_cum), use_container_width=True)

        if factor_corr is not None:
            st.caption(
                "Correlations between the 5 factors. They're designed to be orthogonal "
                "but real-world correlations are never perfectly zero."
            )
            from portfolio_lab.ui.charts import factor_correlation_heatmap
            st.plotly_chart(factor_correlation_heatmap(factor_corr), use_container_width=True)


def _render_factor_regression(factor_data: dict):
    port_reg = factor_data.get("portfolio_regression")
    if not port_reg:
        st.info("Factor regression not available.")
        return

    st.caption(
        "OLS regression of portfolio returns against the 5 Fama-French factors. "
        "A significant alpha (|t| > 2.0) suggests returns beyond what factor exposures explain."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        alpha_str = f"{port_reg['alpha']*100:.2f}%"
        if port_reg["alpha_significant"]:
            st.success(f"Alpha: {alpha_str} (t={port_reg['alpha_t_stat']:.2f}) — Significant")
        else:
            st.info(f"Alpha: {alpha_str} (t={port_reg['alpha_t_stat']:.2f}) — Not significant")
    with c2:
        st.metric("R-squared", f"{port_reg['r_squared']:.3f}")
        st.caption(explain("r_squared"))
    with c3:
        st.metric("Observations", f"{port_reg['n_obs']:,}")

    # Factor loadings table
    loadings = port_reg["factor_loadings"]
    t_stats = port_reg["factor_t_stats"]
    rows = []
    for f in sorted(loadings.keys(), key=lambda x: abs(t_stats.get(x, 0)), reverse=True):
        beta = loadings.get(f, 0)
        t = t_stats.get(f, 0)
        rows.append({
            "Factor": f,
            "Beta": f"{beta:.4f}",
            "t-stat": f"{t:.2f}",
            "Significant (95%)": "Yes" if abs(t) > 2.0 else "No",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Factor exposure chart
    from portfolio_lab.ui.charts import factor_regression_summary_chart
    st.plotly_chart(
        factor_regression_summary_chart(loadings, t_stats),
        use_container_width=True,
    )

    # Per-asset regressions
    asset_regs = factor_data.get("asset_regressions", [])
    if asset_regs:
        with st.expander(f"Per-Asset Factor Regressions ({len(asset_regs)} assets)"):
            rows = []
            for reg in asset_regs:
                rows.append({
                    "Ticker": reg["ticker"],
                    "Alpha (ann.)": f"{reg['alpha']*100:.2f}%",
                    "Alpha t": f"{reg['alpha_t_stat']:.1f}",
                    "Mkt-RF": f"{reg.get('Mkt-RF_beta', 0):.3f}",
                    "SMB": f"{reg.get('SMB_beta', 0):.3f}",
                    "HML": f"{reg.get('HML_beta', 0):.3f}",
                    "RMW": f"{reg.get('RMW_beta', 0):.3f}",
                    "CMA": f"{reg.get('CMA_beta', 0):.3f}",
                    "R²": f"{reg['r_squared']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_stress_flags(flags_data: dict):
    st.markdown(section_header("Stress Flags"), unsafe_allow_html=True)
    st.caption("Potential weaknesses or risks detected in your portfolio.")

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


def _render_additional_charts(analytics: dict, opt_res, report: dict, tickers: list[str]):
    """Extra charts in a collapsible section."""
    from portfolio_lab.ui.charts import (
        nominal_vs_real_chart, drawdown_chart, rolling_performance_chart,
        efficient_frontier_chart, income_projection_chart,
        correlation_heatmap, factor_exposure_chart,
    )
    from portfolio_lab.optimization.efficient_frontier import compute_efficient_frontier
    from portfolio_lab.analytics.income import project_income

    mc_settings = st.session_state.get("settings_mc", {})
    mc_res = st.session_state["analysis_result"]["mc_result"]
    rf_rate = st.session_state.get("analysis_rf_rate", 0.04)
    opt_settings = st.session_state.get("settings_opt", {})

    # Nominal vs Real
    st.plotly_chart(
        nominal_vs_real_chart(
            mc_res.nominal_paths, mc_res.real_paths,
            mc_settings.get("horizon_years", 10),
        ),
        use_container_width=True,
    )

    # Correlation heatmap
    st.plotly_chart(correlation_heatmap(analytics["correlation_matrix"]), use_container_width=True)

    # Factor exposures
    if analytics.get("factor_loadings"):
        st.plotly_chart(factor_exposure_chart(analytics["factor_loadings"]), use_container_width=True)

    # Efficient frontier
    with st.spinner("Computing efficient frontier..."):
        exp_rets = analytics["expected_returns"]
        mu_arr = np.array([exp_rets[t] for t in tickers])
        ef = compute_efficient_frontier(
            tickers, mu_arr, analytics["cov_matrix"].values, rf_rate,
            n_points=30,
            min_weight=opt_settings.get("min_weight", 0.01),
            max_weight=opt_settings.get("max_weight", 0.15),
        )
        if len(ef.returns) > 0:
            st.plotly_chart(
                efficient_frontier_chart(
                    ef.volatilities, ef.returns,
                    ef.max_sharpe_idx, ef.min_vol_idx,
                    current_vol=opt_res.volatility,
                    current_ret=opt_res.expected_return,
                    risk_free_rate=rf_rate,
                ),
                use_container_width=True,
            )

    # Drawdown
    returns_df = analytics["returns"]
    port_prices = (1 + (returns_df * opt_res.raw_weights).sum(axis=1)).cumprod()
    st.plotly_chart(drawdown_chart(port_prices, "Historical Portfolio Drawdown"), use_container_width=True)

    # Rolling performance
    st.plotly_chart(rolling_performance_chart(returns_df), use_container_width=True)

    # Income projection
    summary = report["portfolio_summary"]
    if summary["income_yield"] > 0:
        income_proj = project_income(
            mc_settings.get("initial_investment", 100_000),
            summary["income_yield"],
            summary["expected_net_return"],
            mc_settings.get("horizon_years", 10),
            mc_settings.get("inflation_rate", 0.03),
        )
        st.plotly_chart(
            income_projection_chart(
                income_proj["year"].tolist(),
                income_proj["nominal_income"].tolist(),
                income_proj["real_income"].tolist(),
            ),
            use_container_width=True,
        )
