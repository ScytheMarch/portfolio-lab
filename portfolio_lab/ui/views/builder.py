"""Portfolio Builder page — ticker selection, settings configuration, and random basket."""

from __future__ import annotations

import streamlit as st

from portfolio_lab.ui.styles import (
    gradient_header, glass_card, section_header, ticker_chips_html,
    ACCENT_INDIGO, TEXT_SECONDARY,
)
from portfolio_lab.ui.definitions import explain
from portfolio_lab.utils.random_basket import DEFAULT_UNIVERSE, generate_random_portfolio
from portfolio_lab.config import get_settings


def render() -> None:
    st.markdown(gradient_header("Portfolio Builder", "🏗️"), unsafe_allow_html=True)
    st.caption(
        "Build your portfolio, configure analysis settings, then hit **Run Analysis** "
        "in the top bar. Your selections persist across pages."
    )

    # ── Section 1: Portfolio Source ────────────────────────────────────────
    st.markdown(section_header("Step 1 · Choose Your Portfolio"), unsafe_allow_html=True)

    source_mode = st.radio(
        "Portfolio source",
        ["Manual Entry", "Demo Portfolio", "Random Basket"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_mode == "Manual Entry":
        _render_manual_entry()
    elif source_mode == "Demo Portfolio":
        _render_demo_portfolio()
    else:
        _render_random_basket()

    # Show current portfolio
    tickers = st.session_state.get("tickers", [])
    if tickers:
        st.markdown(
            glass_card(
                f'<div style="color:#64748b;font-size:0.75em;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px">'
                f'CURRENT PORTFOLIO — {len(tickers)} ASSETS</div>'
                + ticker_chips_html(tickers)
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section 2: Analysis Settings ──────────────────────────────────────
    st.markdown(section_header("Step 2 · Configure Analysis Settings"), unsafe_allow_html=True)
    st.caption(
        "Sensible defaults are pre-filled. Adjust as needed — hover over any input "
        "for a plain-English explanation."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        _render_data_settings()
        _render_optimization_settings()
        _render_factor_tilt_settings()

    with col_right:
        _render_monte_carlo_settings()
        _render_goals_fees_settings()
        _render_factor_scenario_settings()

    # ── Config status ─────────────────────────────────────────────────────
    tickers = st.session_state.get("tickers", [])
    if tickers:
        st.success(
            f"Ready to analyze **{len(tickers)}** assets. "
            f"Click **Run Analysis** in the top bar to start."
        )
    else:
        st.warning("Select at least one ticker above to begin.")


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio source renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_manual_entry():
    raw = st.text_input(
        "Enter ticker symbols (comma-separated)",
        value=st.session_state.get("manual_tickers_raw", "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, XOM, PG, UNH, BND, GLD, VNQ, TLT"),
        help="Type stock/ETF tickers separated by commas. e.g. VTI, VXUS, BND, GLD",
        key="manual_tickers_raw",
    )
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    st.session_state["tickers"] = tickers


def _render_demo_portfolio():
    settings = get_settings()
    tickers = settings.demo_tickers
    st.session_state["tickers"] = tickers
    st.info(f"Loaded demo portfolio with {len(tickers)} assets.")


def _render_random_basket():
    st.caption(
        "Randomly select 30-60 tickers from a master universe of ~60 major US stocks. "
        "Lock tickers to always include them. Exclude tickers to temporarily remove them."
    )

    # Editable universe
    with st.expander("Edit Master Universe"):
        universe_raw = st.text_area(
            "Master basket (one ticker per line or comma-separated)",
            value=", ".join(st.session_state.get("basket_universe", DEFAULT_UNIVERSE)),
            height=120,
            help="Edit the pool of stocks to randomly sample from.",
        )
        parsed = [t.strip().upper() for t in universe_raw.replace("\n", ",").split(",") if t.strip()]
        st.session_state["basket_universe"] = parsed
        st.caption(f"{len(parsed)} tickers in universe")

    universe = st.session_state.get("basket_universe", DEFAULT_UNIVERSE)

    col1, col2 = st.columns(2)
    with col1:
        min_picks = st.slider("Min picks", 5, len(universe), 30, key="basket_min")
    with col2:
        max_picks = st.slider("Max picks", min_picks, len(universe), min(60, len(universe)), key="basket_max")

    col1, col2 = st.columns(2)
    with col1:
        locked = st.multiselect(
            "Locked tickers (always included)",
            options=universe,
            default=st.session_state.get("basket_locked", []),
            key="basket_locked",
            help="These tickers will always be in the generated portfolio.",
        )
    with col2:
        excluded = st.multiselect(
            "Excluded tickers (temporarily removed)",
            options=[t for t in universe if t not in locked],
            default=st.session_state.get("basket_excluded", []),
            key="basket_excluded",
            help="These tickers won't be included in the random selection.",
        )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("Generate Random Portfolio", type="primary", use_container_width=True, icon="🎲"):
            _do_random_generation(universe, min_picks, max_picks, locked, excluded)
    with col2:
        if st.button("Re-randomize", use_container_width=True, icon="🔄"):
            # Increment seed for new random selection
            st.session_state["basket_seed"] = st.session_state.get("basket_seed", 42) + 1
            _do_random_generation(universe, min_picks, max_picks, locked, excluded)
    with col3:
        seed = st.number_input("Seed", value=st.session_state.get("basket_seed", 42), key="basket_seed_input",
                               help="Set a specific seed for reproducible results.")
        st.session_state["basket_seed"] = int(seed)


def _do_random_generation(universe, min_picks, max_picks, locked, excluded):
    try:
        tickers = generate_random_portfolio(
            universe=universe,
            min_picks=min_picks,
            max_picks=max_picks,
            locked=locked,
            excluded=excluded,
            seed=st.session_state.get("basket_seed", 42),
        )
        st.session_state["tickers"] = tickers
        st.toast(f"Generated portfolio with {len(tickers)} tickers!")
    except ValueError as e:
        st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Settings renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_data_settings():
    with st.expander("Data & Assumptions", expanded=True):
        st.caption("How historical data is sourced and processed.")

        lookback_years = st.number_input(
            "Lookback Period (years)", 1, 30,
            value=st.session_state.get("lookback_years", 5), step=1,
            help=explain("expected_return") + " Longer lookbacks smooth out short-term noise but may not reflect current conditions.",
            key="lookback_years",
        )

        return_method = st.selectbox(
            "Expected Return Method",
            ["historical_arithmetic", "historical_geometric", "exponentially_weighted",
             "capm", "ff5_implied", "blended"],
            index=1,
            format_func=lambda x: {
                "historical_arithmetic": "Arithmetic Mean — simple average (tends to overstate)",
                "historical_geometric": "Geometric Mean — compound growth rate (most realistic)",
                "exponentially_weighted": "Exponentially Weighted — recent data weighted more",
                "capm": "CAPM — forward-looking, based on market beta",
                "ff5_implied": "Fama-French 5-Factor — based on factor exposures",
                "blended": "Blended — 50% arithmetic / 50% geometric",
            }.get(x, x),
            help=(
                "How expected returns are estimated. Arithmetic mean is the simplest but overstates "
                "compound growth. Geometric mean reflects actual compounding. CAPM and FF5 are "
                "forward-looking factor models."
            ),
            key="return_method",
        )

        rf_override = st.number_input(
            "Risk-Free Rate Override (%, 0 = auto-fetch from Treasury)",
            0.0, 20.0,
            value=st.session_state.get("rf_override_pct", 0.0), step=0.1,
            help="Set to 0 to automatically fetch the current 10-year Treasury yield.",
            key="rf_override_pct",
        )

        st.session_state["settings_data"] = {
            "lookback_years": int(lookback_years),
            "rf_override": rf_override / 100 if rf_override > 0 else None,
        }
        st.session_state["settings_return_method"] = return_method


def _render_optimization_settings():
    with st.expander("Optimization", expanded=True):
        st.caption("What the optimizer tries to achieve and how constrained it is.")

        objective = st.selectbox(
            "Objective",
            ["max_sharpe", "min_volatility", "target_return", "max_income", "max_net_income"],
            format_func=lambda x: {
                "max_sharpe": "Maximum Sharpe Ratio — best risk-adjusted return",
                "min_volatility": "Minimum Volatility — lowest possible risk",
                "target_return": "Target Return — hit a specific return level",
                "max_income": "Maximum Income — highest dividend yield",
                "max_net_income": "Maximum Net Income — yield minus expenses",
            }.get(x, x),
            help=explain("max_sharpe"),
            key="opt_objective",
        )

        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.number_input(
                "Min weight per asset", 0.0, 1.0,
                value=st.session_state.get("opt_min_weight", 0.01), step=0.01,
                help="Minimum allocation to any single asset. Set > 0 to prevent zero-weight assets.",
                key="opt_min_weight",
            )
        with col2:
            max_weight = st.number_input(
                "Max weight per asset", 0.01, 1.0,
                value=st.session_state.get("opt_max_weight", 0.15), step=0.01,
                help="Maximum allocation to any single asset. Lower = more diversified. 0.10-0.15 recommended for 30+ tickers.",
                key="opt_max_weight",
            )

        # Advanced constraints
        target_return = None
        max_vol = None
        min_yield = None

        if objective == "target_return":
            target_return = st.number_input(
                "Target Annual Return (%)", 0.0, 50.0, 8.0, 0.5,
                help="The exact return level the optimizer will try to hit.",
            ) / 100

        with st.expander("Advanced Constraints"):
            max_vol_input = st.number_input(
                "Max Volatility (%, 0 = unconstrained)", 0.0, 100.0, 0.0, 0.5,
                help=explain("volatility") + " Set a cap to limit portfolio risk.",
            )
            max_vol = max_vol_input / 100 if max_vol_input > 0 else None

            min_yield_input = st.number_input(
                "Min Income Yield (%, 0 = unconstrained)", 0.0, 20.0, 0.0, 0.1,
                help=explain("income_yield") + " Set a floor for dividend income.",
            )
            min_yield = min_yield_input / 100 if min_yield_input > 0 else None

        st.session_state["settings_opt"] = {
            "objective": objective,
            "return_method": st.session_state.get("settings_return_method", "historical_geometric"),
            "min_weight": min_weight,
            "max_weight": max_weight,
            "target_return": target_return,
            "max_vol": max_vol,
            "min_yield": min_yield,
        }


def _render_monte_carlo_settings():
    with st.expander("Monte Carlo Simulation", expanded=True):
        st.caption(
            "Simulate thousands of possible futures to understand the range of outcomes. "
            + explain("monte_carlo")
        )

        col1, col2 = st.columns(2)
        with col1:
            initial_investment = st.number_input(
                "Initial Investment ($)", 1_000, 100_000_000,
                value=st.session_state.get("mc_initial", 100_000), step=10_000,
                help="Starting portfolio value for the simulation.",
                key="mc_initial",
            )
            n_simulations = st.number_input(
                "Simulations", 500, 50_000,
                value=st.session_state.get("mc_nsims", 5_000), step=500,
                help="More simulations = smoother probability estimates. 5,000 is a good default.",
                key="mc_nsims",
            )

        with col2:
            horizon_years = st.number_input(
                "Horizon (years)", 1, 50,
                value=st.session_state.get("mc_horizon", 10), step=1,
                help="How far into the future to project.",
                key="mc_horizon",
            )
            rebalance_frequency = st.selectbox(
                "Rebalance Frequency",
                ["annual", "semi-annual", "quarterly", "monthly"],
                help="How often the portfolio returns to target weights.",
                key="mc_rebalance",
            )

        col1, col2 = st.columns(2)
        with col1:
            inflation_rate = st.number_input(
                "Inflation Rate (%)", 0.0, 15.0,
                value=st.session_state.get("mc_inflation_pct", 3.0), step=0.25,
                help="Expected average annual inflation. Used to compute real (purchasing-power-adjusted) returns.",
                key="mc_inflation_pct",
            ) / 100
        with col2:
            stochastic_inflation = st.checkbox(
                "Stochastic Inflation",
                value=st.session_state.get("mc_stochastic", False),
                help="If checked, inflation itself is randomized each year around the base rate.",
                key="mc_stochastic",
            )

        with st.expander("Contributions & Withdrawals"):
            col1, col2 = st.columns(2)
            with col1:
                annual_contribution = st.number_input(
                    "Annual Contribution ($)", 0, 1_000_000,
                    value=st.session_state.get("mc_contrib", 0), step=1_000,
                    help="Amount added to the portfolio each year.",
                    key="mc_contrib",
                )
            with col2:
                annual_withdrawal = st.number_input(
                    "Annual Withdrawal ($)", 0, 1_000_000,
                    value=st.session_state.get("mc_withdraw", 0), step=1_000,
                    help="Amount withdrawn from the portfolio each year.",
                    key="mc_withdraw",
                )

        st.session_state["settings_mc"] = {
            "initial_investment": float(initial_investment),
            "n_simulations": int(n_simulations),
            "horizon_years": int(horizon_years),
            "rebalance_frequency": rebalance_frequency,
            "inflation_rate": inflation_rate,
            "stochastic_inflation": stochastic_inflation,
            "annual_contribution": float(annual_contribution),
            "annual_withdrawal": float(annual_withdrawal),
        }


def _render_goals_fees_settings():
    with st.expander("Goals & Fees"):
        st.caption("Set target outcomes and advisory fees.")

        col1, col2 = st.columns(2)
        with col1:
            target_value = st.number_input(
                "Target Ending Value ($, 0 = none)", 0, 100_000_000,
                value=st.session_state.get("goal_target_value", 0), step=10_000,
                help="The dollar amount you're trying to reach. The simulation will report what % of scenarios hit this target.",
                key="goal_target_value",
            )
        with col2:
            target_return_goal = st.number_input(
                "Target Annualized Return (%, 0 = none)", 0.0, 50.0,
                value=st.session_state.get("goal_target_return_pct", 0.0), step=0.5,
                help="The annualized return you want to achieve.",
                key="goal_target_return_pct",
            )

        advisory_fee = st.number_input(
            "Advisory Fee (% per year)", 0.0, 3.0,
            value=st.session_state.get("fee_advisory_pct", 0.0), step=0.05,
            help="Annual advisory/management fee deducted from returns. Typical range: 0.25-1.0%.",
            key="fee_advisory_pct",
        )

        st.session_state["settings_goals"] = {
            "target_value": float(target_value) if target_value > 0 else None,
            "target_return_goal": target_return_goal / 100 if target_return_goal > 0 else None,
        }
        st.session_state["settings_fees"] = {
            "advisory_fee": advisory_fee / 100,
        }


def _render_factor_tilt_settings():
    with st.expander("Factor Tilts"):
        st.caption(
            "Constrain the optimizer to maintain minimum exposure to specific Fama-French risk factors. "
            "Leave at 0 for no constraint. "
            + explain("fama_french")
        )
        factor_labels = {
            "Mkt-RF": ("Market Beta", explain("mkt_rf")),
            "SMB": ("Small Cap Tilt", explain("smb")),
            "HML": ("Value Tilt", explain("hml")),
            "RMW": ("Profitability Tilt", explain("rmw")),
            "CMA": ("Conservative Investment Tilt", explain("cma")),
        }
        tilts = {}
        for factor, (label, tooltip) in factor_labels.items():
            val = st.number_input(
                f"Min {label} ({factor})",
                min_value=-1.0, max_value=2.0, value=0.0, step=0.05,
                help=tooltip,
                key=f"tilt_{factor}",
            )
            if val != 0.0:
                tilts[factor] = val

        st.session_state["settings_factor_tilts"] = tilts


def _render_factor_scenario_settings():
    from portfolio_lab.analytics.factor_model import SCENARIO_PRESETS

    with st.expander("Factor Scenarios"):
        st.caption(
            "Test how your portfolio would perform under different factor environments. "
            "Select a preset or enter custom factor return assumptions."
        )

        preset = st.selectbox(
            "Scenario Preset",
            ["Custom"] + list(SCENARIO_PRESETS.keys()),
            help="Pre-configured factor return scenarios. 'Bull Market' assumes strong equity returns; 'Value Rally' assumes value stocks outperform.",
            key="scenario_preset",
        )

        if preset != "Custom":
            defaults = SCENARIO_PRESETS[preset]
        else:
            defaults = {"Mkt-RF": 0.06, "SMB": 0.02, "HML": 0.03, "RMW": 0.03, "CMA": 0.02}

        scenario = {}
        col1, col2 = st.columns(2)
        factors = list(defaults.keys())
        for i, factor in enumerate(factors):
            with col1 if i % 2 == 0 else col2:
                scenario[factor] = st.number_input(
                    f"{factor} Return (%)",
                    min_value=-50.0, max_value=50.0,
                    value=defaults[factor] * 100, step=0.5,
                    key=f"scenario_{factor}",
                ) / 100

        st.session_state["settings_scenarios"] = {
            "preset": preset,
            "scenario_premia": scenario,
        }
