"""
Global configuration and defaults for portfolio_lab.

All assumptions are explicit and documented.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Central configuration for the portfolio lab."""

    # --- Risk-free rate ---
    # Default: 91-day T-bill. Updated at runtime from treasury_data module.
    risk_free_rate: float = 0.05  # annualized decimal; overridden by live fetch
    risk_free_source: str = "manual_default"  # tracks provenance

    # --- Inflation ---
    default_inflation_rate: float = 0.03  # 3% fixed assumption
    stochastic_inflation_mean: float = 0.03
    stochastic_inflation_std: float = 0.01

    # --- Monte Carlo defaults ---
    mc_num_simulations: int = 5000
    mc_horizon_years: int = 10
    mc_rebalance_frequency: str = "annual"  # annual, semi-annual, quarterly, monthly
    mc_initial_investment: float = 100_000.0

    # --- Optimization defaults ---
    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_shorting: bool = False

    # --- Fee defaults ---
    advisory_fee: float = 0.0  # annual advisory fee in decimal

    # --- Data defaults ---
    default_lookback_years: int = 5
    trading_days_per_year: int = 252
    months_per_year: int = 12

    # --- Treasury proxy tickers ---
    tbill_proxy_ticker: str = "^IRX"  # 13-week T-bill yield index
    tbill_etf_proxy: str = "BIL"  # SPDR 1-3 month T-bill ETF
    short_treasury_etf: str = "SHV"  # iShares Short Treasury Bond ETF
    intermediate_treasury_etf: str = "IEF"  # 7-10 year
    long_treasury_etf: str = "TLT"  # 20+ year
    tips_etf: str = "TIP"  # TIPS ETF

    # --- Factor data ---
    ff5_source_url: str = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    ff5_monthly_url: str = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )

    # --- Demo tickers ---
    demo_tickers: list = field(default_factory=lambda: [
        "VTI",   # Total US stock market
        "VXUS",  # International developed + emerging
        "BND",   # Total US bond market
        "VNQ",   # US REITs
        "GLD",   # Gold
        "TLT",   # Long-term Treasuries
        "BIL",   # T-bills
    ])

    demo_portfolio_weights: dict = field(default_factory=lambda: {
        "VTI": 0.40,
        "VXUS": 0.15,
        "BND": 0.20,
        "VNQ": 0.05,
        "GLD": 0.05,
        "TLT": 0.10,
        "BIL": 0.05,
    })


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
