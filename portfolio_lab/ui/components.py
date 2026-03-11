"""
Streamlit input components for portfolio_lab.

All user-facing controls are defined here.
The UI only renders; it does not compute analytics.
"""
import streamlit as st
from portfolio_lab.config import get_settings


def render_ticker_input() -> list[str]:
    """Render ticker input widget. Returns list of valid tickers."""
    settings = get_settings()

    st.subheader("Asset Selection")

    input_mode = st.radio(
        "How would you like to select assets?",
        ["Enter tickers manually", "Use demo portfolio"],
        horizontal=True,
    )

    if input_mode == "Use demo portfolio":
        tickers = settings.demo_tickers
        st.info(f"Demo tickers: {', '.join(tickers)}")
    else:
        raw = st.text_input(
            "Enter ticker symbols (comma-separated)",
            value="VTI, VXUS, BND, VNQ, GLD, TLT, BIL",
            help="E.g., VTI, VXUS, BND, GLD",
        )
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker.")

    return tickers


def render_optimization_settings() -> dict:
    """Render optimization objective and constraint controls."""
    st.subheader("Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        objective = st.selectbox(
            "Optimization Objective",
            [
                "max_sharpe",
                "min_volatility",
                "target_return",
                "max_income",
                "max_net_income",
            ],
            format_func=lambda x: {
                "max_sharpe": "Maximum Sharpe Ratio",
                "min_volatility": "Minimum Volatility",
                "target_return": "Target Return",
                "max_income": "Maximum Income",
                "max_net_income": "Maximum Net Income",
            }.get(x, x),
        )

        return_method = st.selectbox(
            "Expected Return Method",
            [
                "historical_arithmetic",
                "historical_geometric",
                "exponentially_weighted",
                "ff5_implied",
                "blended",
            ],
            format_func=lambda x: {
                "historical_arithmetic": "Historical Arithmetic Mean",
                "historical_geometric": "Historical Geometric Mean",
                "exponentially_weighted": "Exponentially Weighted",
                "ff5_implied": "Fama-French 5-Factor Implied",
                "blended": "Blended (50% Arith / 50% Geo)",
            }.get(x, x),
        )

    with col2:
        min_weight = st.number_input("Min Weight per Asset", 0.0, 1.0, 0.0, 0.01)
        max_weight = st.number_input("Max Weight per Asset", 0.0, 1.0, 1.0, 0.01)

    # Optional constraints
    with st.expander("Advanced Constraints"):
        target_return = None
        if objective == "target_return":
            target_return = st.number_input(
                "Target Annual Return (%)", 0.0, 50.0, 8.0, 0.5
            ) / 100

        max_vol = st.number_input(
            "Max Volatility (%, 0=unconstrained)", 0.0, 100.0, 0.0, 0.5
        )
        max_vol = max_vol / 100 if max_vol > 0 else None

        min_yield = st.number_input(
            "Min Income Yield (%, 0=unconstrained)", 0.0, 20.0, 0.0, 0.1
        )
        min_yield = min_yield / 100 if min_yield > 0 else None

    return {
        "objective": objective,
        "return_method": return_method,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "target_return": target_return,
        "max_vol": max_vol,
        "min_yield": min_yield,
    }


def render_monte_carlo_settings() -> dict:
    """Render Monte Carlo simulation controls."""
    st.subheader("Monte Carlo Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_investment = st.number_input(
            "Initial Investment ($)", 1_000, 100_000_000, 100_000, 10_000
        )
        n_simulations = st.number_input(
            "Number of Simulations", 100, 50_000, 5_000, 500
        )

    with col2:
        horizon_years = st.number_input("Horizon (years)", 1, 50, 10, 1)
        rebalance_frequency = st.selectbox(
            "Rebalance Frequency",
            ["annual", "semi-annual", "quarterly", "monthly"],
        )

    with col3:
        inflation_rate = st.number_input(
            "Inflation Rate (%)", 0.0, 15.0, 3.0, 0.25
        ) / 100
        stochastic_inflation = st.checkbox("Stochastic Inflation", value=False)

    # Cash flows
    with st.expander("Contributions & Withdrawals"):
        annual_contribution = st.number_input(
            "Annual Contribution ($)", 0, 1_000_000, 0, 1_000
        )
        annual_withdrawal = st.number_input(
            "Annual Withdrawal ($)", 0, 1_000_000, 0, 1_000
        )

    return {
        "initial_investment": float(initial_investment),
        "n_simulations": int(n_simulations),
        "horizon_years": int(horizon_years),
        "rebalance_frequency": rebalance_frequency,
        "inflation_rate": inflation_rate,
        "stochastic_inflation": stochastic_inflation,
        "annual_contribution": float(annual_contribution),
        "annual_withdrawal": float(annual_withdrawal),
    }


def render_goal_settings() -> dict:
    """Render goal / target settings."""
    st.subheader("Goal Settings")

    col1, col2 = st.columns(2)

    with col1:
        target_value = st.number_input(
            "Target Ending Value ($, 0=none)", 0, 100_000_000, 0, 10_000
        )
        target_value = float(target_value) if target_value > 0 else None

    with col2:
        target_return_goal = st.number_input(
            "Target Annualized Return (%, 0=none)", 0.0, 50.0, 0.0, 0.5
        )
        target_return_goal = target_return_goal / 100 if target_return_goal > 0 else None

    return {
        "target_value": target_value,
        "target_return_goal": target_return_goal,
    }


def render_fee_settings() -> dict:
    """Render fee / expense settings."""
    st.subheader("Fee Settings")
    advisory_fee = st.number_input(
        "Advisory Fee (% per year)", 0.0, 3.0, 0.0, 0.05
    ) / 100
    return {"advisory_fee": advisory_fee}


def render_data_settings() -> dict:
    """Render data lookback and risk-free rate settings."""
    with st.expander("Data & Risk-Free Rate Settings"):
        lookback_years = st.number_input("Lookback Period (years)", 1, 30, 5, 1)
        rf_override = st.number_input(
            "Risk-Free Rate Override (%, 0=auto-fetch)", 0.0, 20.0, 0.0, 0.1
        )
        rf_override = rf_override / 100 if rf_override > 0 else None

    return {
        "lookback_years": int(lookback_years),
        "rf_override": rf_override,
    }
