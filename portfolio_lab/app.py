"""
portfolio_lab — Portfolio Construction, Optimization & Simulation Platform

Entry point for the Streamlit application.
Run: streamlit run portfolio_lab/app.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure package is importable when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_lab.config import get_settings
from portfolio_lab.data.market_data import fetch_price_data, DataFetchError
from portfolio_lab.data.fund_metadata import fetch_fund_metadata
from portfolio_lab.data.treasury_data import fetch_risk_free_rate
from portfolio_lab.data.factor_data import fetch_ff5_factors
from portfolio_lab.analytics.returns import simple_returns, cagr, geometric_mean_return
from portfolio_lab.analytics.risk import (
    covariance_matrix,
    correlation_matrix,
    max_drawdown,
    drawdown_series,
)
from portfolio_lab.analytics.income import project_income
from portfolio_lab.optimization.efficient_frontier import compute_efficient_frontier
from portfolio_lab.simulation.scenario_engine import run_full_scenario
from portfolio_lab.ui.components import (
    render_ticker_input,
    render_optimization_settings,
    render_monte_carlo_settings,
    render_goal_settings,
    render_fee_settings,
    render_data_settings,
    render_factor_tilt_settings,
    render_factor_scenario_settings,
)
from portfolio_lab.ui.charts import (
    efficient_frontier_chart,
    correlation_heatmap,
    allocation_pie_chart,
    monte_carlo_fan_chart,
    nominal_vs_real_chart,
    drawdown_chart,
    risk_contribution_chart,
    factor_exposure_chart,
    goal_hit_rate_gauge,
    income_projection_chart,
    rolling_performance_chart,
    terminal_value_distribution_chart,
    factor_attribution_chart,
    factor_premium_history_chart,
    factor_correlation_heatmap,
    factor_regression_summary_chart,
    scenario_comparison_chart,
)
from portfolio_lab.ui.pages import render_full_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page config ---
st.set_page_config(
    page_title="Portfolio Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Portfolio Lab")
st.caption(
    "Portfolio construction, optimization, simulation & diagnostics. "
    "Not investment advice. All results are model outputs based on historical data and assumptions."
)


# =====================================================
# SIDEBAR: All inputs
# =====================================================
with st.sidebar:
    st.header("Configuration")

    # Tickers
    tickers = render_ticker_input()

    # Data settings
    data_settings = render_data_settings()

    # Optimization
    opt_settings = render_optimization_settings()

    # Monte Carlo
    mc_settings = render_monte_carlo_settings()

    # Goals
    goal_settings = render_goal_settings()

    # Factor tilts
    factor_tilt_targets = render_factor_tilt_settings()

    # Factor scenarios
    factor_scenario_settings = render_factor_scenario_settings()

    # Fees
    fee_settings = render_fee_settings()

    # Run button
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

# =====================================================
# MAIN: Analysis execution
# =====================================================
if run_analysis and tickers:
    with st.spinner("Fetching market data..."):
        try:
            prices = fetch_price_data(
                tickers,
                lookback_years=data_settings["lookback_years"],
            )
            st.success(f"Loaded {len(prices)} days of data for {len(prices.columns)} tickers.")
            # Update tickers to only those that were successfully fetched
            tickers = prices.columns.tolist()
        except DataFetchError as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    with st.spinner("Fetching fund metadata..."):
        fund_info = fetch_fund_metadata(tickers)

        # Condense metadata warnings into grouped summaries
        missing_er = []
        missing_yield = []
        other_warnings = []
        for t, info in fund_info.items():
            for w in info.warnings:
                if "Expense ratio not available" in w:
                    missing_er.append(t)
                elif "Dividend yield not available" in w:
                    missing_yield.append(t)
                else:
                    other_warnings.append(w)

        if missing_er:
            with st.expander(f"Expense ratio unavailable for {len(missing_er)} ticker(s)"):
                st.write(", ".join(missing_er))
                st.caption("Manual input recommended. Do NOT assume zero for funds/ETFs.")
        if missing_yield:
            with st.expander(f"Dividend yield unavailable for {len(missing_yield)} ticker(s)"):
                st.write(", ".join(missing_yield))
                st.caption("These will be treated as 0% yield in income calculations.")
        for w in other_warnings:
            st.warning(w)

    with st.spinner("Fetching risk-free rate..."):
        rf_rate, rf_source = fetch_risk_free_rate(data_settings["rf_override"])
        st.info(f"Risk-free rate: {rf_rate*100:.2f}% (source: {rf_source})")

    # Always fetch factor data (needed for regression analysis, tilting, scenarios)
    factor_data = None
    with st.spinner("Fetching Fama-French factor data..."):
        factor_data = fetch_ff5_factors(frequency="daily")
        if factor_data is None:
            st.warning("Factor data unavailable. Factor analysis features will be limited.")
            if opt_settings["return_method"] == "ff5_implied":
                st.warning("Falling back to arithmetic mean returns.")
                opt_settings["return_method"] = "historical_arithmetic"

    # =====================================================
    # RUN SCENARIO
    # =====================================================
    with st.spinner("Running optimization and simulation..."):
        result = run_full_scenario(
            prices=prices,
            fund_info=fund_info,
            objective=opt_settings["objective"],
            return_method=opt_settings["return_method"],
            risk_free_rate=rf_rate,
            risk_free_source=rf_source,
            min_weight=opt_settings["min_weight"],
            max_weight=opt_settings["max_weight"],
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
            target_value=goal_settings["target_value"],
            target_return_goal=goal_settings["target_return_goal"],
            advisory_fee=fee_settings["advisory_fee"],
            factor_data=factor_data,
            mc_seed=42,
            factor_tilt_targets=factor_tilt_targets if factor_tilt_targets else None,
            scenario_premia=factor_scenario_settings.get("scenario_premia"),
        )

    if result.get("error"):
        st.error(f"Optimization failed: {result['error']}")
        opt_res = result["optimization_result"]
        if opt_res and opt_res.warnings:
            for w in opt_res.warnings:
                st.warning(w)
        st.stop()

    opt_res = result["optimization_result"]
    mc_res = result["mc_result"]
    report = result["report"]
    analytics = result["analytics"]

    # =====================================================
    # CHARTS TAB
    # =====================================================
    tab_report, tab_charts, tab_data = st.tabs(["Report", "Charts", "Raw Data"])

    with tab_report:
        render_full_report(report)

    with tab_charts:
        st.header("Visualizations")

        # Allocation
        st.plotly_chart(
            allocation_pie_chart(opt_res.weights),
            use_container_width=True,
        )

        # Monte Carlo fan chart
        st.plotly_chart(
            monte_carlo_fan_chart(
                mc_res.nominal_paths,
                mc_settings["horizon_years"],
                mc_settings["initial_investment"],
            ),
            use_container_width=True,
        )

        # Terminal value distribution (confidence intervals)
        st.plotly_chart(
            terminal_value_distribution_chart(
                mc_res.nominal_terminal,
                mc_settings["initial_investment"],
                target_value=goal_settings.get("target_value"),
            ),
            use_container_width=True,
        )

        # Nominal vs Real
        st.plotly_chart(
            nominal_vs_real_chart(
                mc_res.nominal_paths,
                mc_res.real_paths,
                mc_settings["horizon_years"],
            ),
            use_container_width=True,
        )

        # Goal hit rate gauge
        sim_summary = report["simulation_summary"]
        if sim_summary.get("goal_hit_rate") is not None:
            st.plotly_chart(
                goal_hit_rate_gauge(sim_summary["goal_hit_rate"], "Goal Hit Rate"),
                use_container_width=True,
            )

        # Risk contributions
        risk_pct = report["risk_drivers"]["risk_percentage_contributions"]
        st.plotly_chart(
            risk_contribution_chart(risk_pct),
            use_container_width=True,
        )

        # Correlation heatmap
        corr = analytics["correlation_matrix"]
        st.plotly_chart(
            correlation_heatmap(corr),
            use_container_width=True,
        )

        # Factor exposures
        if analytics.get("factor_loadings"):
            st.plotly_chart(
                factor_exposure_chart(analytics["factor_loadings"]),
                use_container_width=True,
            )

        # Factor regression with significance
        port_reg = analytics.get("portfolio_factor_regression")
        if port_reg and port_reg.factor_loadings:
            st.plotly_chart(
                factor_regression_summary_chart(
                    port_reg.factor_loadings,
                    port_reg.factor_t_stats,
                ),
                use_container_width=True,
            )

        # Factor return attribution
        attrib = analytics.get("factor_attribution")
        if attrib:
            st.plotly_chart(
                factor_attribution_chart(attrib),
                use_container_width=True,
            )

        # Scenario comparison
        scenario_res = analytics.get("scenario_results")
        if scenario_res and analytics.get("factor_loadings"):
            st.plotly_chart(
                scenario_comparison_chart(
                    scenario_res,
                    base_return=opt_res.expected_return,
                ),
                use_container_width=True,
            )

        # Factor premium history
        factor_cum = analytics.get("factor_cumulative_returns")
        if factor_cum is not None:
            st.plotly_chart(
                factor_premium_history_chart(factor_cum),
                use_container_width=True,
            )

        # Factor correlation matrix
        factor_corr = analytics.get("factor_correlation")
        if factor_corr is not None:
            st.plotly_chart(
                factor_correlation_heatmap(factor_corr),
                use_container_width=True,
            )

        # Efficient frontier
        with st.spinner("Computing efficient frontier..."):
            returns_df = analytics["returns"]
            cov = analytics["cov_matrix"]
            exp_rets = analytics["expected_returns"]
            mu_arr = np.array([exp_rets[t] for t in tickers])

            ef = compute_efficient_frontier(
                tickers, mu_arr, cov.values, rf_rate,
                n_points=30,
                min_weight=opt_settings["min_weight"],
                max_weight=opt_settings["max_weight"],
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

        # Historical drawdown
        port_prices = (1 + (returns_df * opt_res.raw_weights).sum(axis=1)).cumprod()
        st.plotly_chart(
            drawdown_chart(port_prices, "Historical Portfolio Drawdown"),
            use_container_width=True,
        )

        # Rolling performance
        st.plotly_chart(
            rolling_performance_chart(returns_df),
            use_container_width=True,
        )

        # Income projection
        if report["portfolio_summary"]["income_yield"] > 0:
            income_proj = project_income(
                mc_settings["initial_investment"],
                report["portfolio_summary"]["income_yield"],
                report["portfolio_summary"]["expected_net_return"],
                mc_settings["horizon_years"],
                mc_settings["inflation_rate"],
            )
            st.plotly_chart(
                income_projection_chart(
                    income_proj["year"].tolist(),
                    income_proj["nominal_income"].tolist(),
                    income_proj["real_income"].tolist(),
                ),
                use_container_width=True,
            )

    with tab_data:
        st.header("Raw Data")

        st.subheader("Price Data (last 10 rows)")
        st.dataframe(prices.tail(10), use_container_width=True)

        st.subheader("Daily Returns (last 10 rows)")
        st.dataframe(analytics["returns"].tail(10), use_container_width=True)

        st.subheader("Covariance Matrix (Annualized)")
        st.dataframe(analytics["cov_matrix"], use_container_width=True)

        st.subheader("Correlation Matrix")
        st.dataframe(analytics["correlation_matrix"], use_container_width=True)

        st.subheader("Expected Returns")
        er = analytics["expected_returns"]
        er_df = pd.DataFrame({
            "Ticker": list(er.keys()),
            "Expected Return": [f"{v*100:.2f}%" for v in er.values()],
        })
        st.dataframe(er_df, hide_index=True, use_container_width=True)

        st.subheader("Fund Metadata")
        meta_rows = []
        for t, info in fund_info.items():
            meta_rows.append({
                "Ticker": t,
                "Name": info.name,
                "Category": info.category,
                "Asset Class": info.asset_class,
                "Expense Ratio": info.expense_ratio_display,
                "Yield": f"{info.dividend_yield*100:.2f}%" if info.dividend_yield else "N/A",
                "Type": info.quote_type,
            })
        st.dataframe(pd.DataFrame(meta_rows), hide_index=True, use_container_width=True)

        st.subheader("Monte Carlo Statistics")
        if mc_res.stats:
            stats_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:,.4f}" if isinstance(v, float) else str(v)}
                for k, v in mc_res.stats.items()
            ])
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

elif not tickers:
    st.info("Enter tickers in the sidebar and click **Run Analysis** to begin.")
else:
    st.info("Configure your portfolio settings in the sidebar and click **Run Analysis**.")
