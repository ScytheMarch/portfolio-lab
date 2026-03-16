"""Diagnostics page — raw data tables for power users."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio_lab.ui.styles import gradient_header, section_header


def render() -> None:
    st.markdown(gradient_header("Diagnostics", "🔬"), unsafe_allow_html=True)
    st.caption(
        "Raw data tables and detailed outputs from the analysis pipeline. "
        "For power users who want to inspect the numbers directly."
    )

    if "analysis_result" not in st.session_state:
        st.info("Run an analysis first to see diagnostics data here.")
        return

    result = st.session_state["analysis_result"]
    analytics = result["analytics"]
    mc_res = result["mc_result"]
    opt_res = result["optimization_result"]

    tab_prices, tab_returns, tab_matrices, tab_meta, tab_mc, tab_factors = st.tabs([
        "Prices", "Returns", "Matrices", "Fund Metadata", "Monte Carlo", "Factor Detail"
    ])

    with tab_prices:
        st.markdown(section_header("Price Data (Last 20 Rows)"), unsafe_allow_html=True)
        st.caption("Adjusted close prices from Yahoo Finance.")
        # Retrieve prices from analytics if available
        if "prices" in analytics:
            st.dataframe(analytics["prices"].tail(20), use_container_width=True)
        else:
            st.info("Price data not stored in analytics. Re-run analysis to populate.")

    with tab_returns:
        st.markdown(section_header("Daily Returns (Last 20 Rows)"), unsafe_allow_html=True)
        st.caption("Simple daily returns: (P_t / P_{t-1}) - 1")
        st.dataframe(analytics["returns"].tail(20), use_container_width=True)

        st.markdown(section_header("Expected Returns"), unsafe_allow_html=True)
        er = analytics["expected_returns"]
        er_df = pd.DataFrame({
            "Ticker": list(er.keys()),
            "Expected Return": [f"{v*100:.2f}%" for v in er.values()],
        }).sort_values("Expected Return", ascending=False)
        st.dataframe(er_df, hide_index=True, use_container_width=True)

    with tab_matrices:
        st.markdown(section_header("Covariance Matrix (Annualized)"), unsafe_allow_html=True)
        st.caption("Annualized covariance of daily returns (x 252).")
        st.dataframe(analytics["cov_matrix"], use_container_width=True)

        st.markdown(section_header("Correlation Matrix"), unsafe_allow_html=True)
        st.caption("Pairwise Pearson correlation of daily returns.")
        st.dataframe(analytics["correlation_matrix"], use_container_width=True)

    with tab_meta:
        st.markdown(section_header("Fund Metadata"), unsafe_allow_html=True)
        st.caption("Metadata from Yahoo Finance — name, category, expense ratio, dividend yield.")
        fund_info = analytics.get("fund_info")
        if fund_info:
            rows = []
            for t, info in fund_info.items():
                rows.append({
                    "Ticker": t,
                    "Name": info.name,
                    "Category": info.category,
                    "Asset Class": info.asset_class,
                    "Expense Ratio": info.expense_ratio_display,
                    "Yield": f"{info.dividend_yield*100:.2f}%" if info.dividend_yield else "N/A",
                    "Type": info.quote_type,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Fund metadata not available in analytics output.")

    with tab_mc:
        st.markdown(section_header("Monte Carlo Statistics"), unsafe_allow_html=True)
        st.caption("Summary statistics across all simulation paths.")
        if mc_res.stats:
            stats_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:,.4f}" if isinstance(v, float) else str(v)}
                for k, v in mc_res.stats.items()
            ])
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        st.markdown(section_header("Optimization Weights"), unsafe_allow_html=True)
        weights = opt_res.weights
        w_df = pd.DataFrame({
            "Ticker": list(weights.keys()),
            "Weight": [f"{v*100:.2f}%" for v in weights.values()],
        }).sort_values("Weight", ascending=False)
        st.dataframe(w_df, hide_index=True, use_container_width=True)

    with tab_factors:
        st.markdown(section_header("Per-Asset Factor Regressions"), unsafe_allow_html=True)
        st.caption(
            "OLS regression of each asset's excess returns against the 5 Fama-French factors. "
            "Alpha is annualized. t-stat > 2.0 indicates statistical significance at 95%."
        )
        report = result["report"]
        factor_analysis = report.get("factor_analysis", {})
        asset_regs = factor_analysis.get("asset_regressions", [])

        if asset_regs:
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
                    "Obs": reg.get("n_obs", ""),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Factor regression data not available. Ensure factor data was fetched successfully.")
