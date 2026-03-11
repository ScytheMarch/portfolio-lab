"""
Risk analytics: covariance, correlation, volatility, drawdown, risk contributions.

Portfolio variance uses w' Σ w (quadratic form).
All risk measures clearly documented for interpretation.
"""
import numpy as np
import pandas as pd

from portfolio_lab.config import get_settings


def covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int | None = None,
    method: str = "sample",
    ewm_span: int | None = None,
) -> pd.DataFrame:
    """
    Compute the covariance matrix of asset returns.

    Args:
        returns: DataFrame of period returns.
        annualize: If True, scale to annual.
        periods_per_year: Auto-detected if None.
        method: 'sample' for standard, 'ewm' for exponentially weighted.
        ewm_span: Required if method='ewm'.

    Returns:
        Annualized covariance matrix.
    """
    ppy = periods_per_year or _infer_ppy(returns)

    if method == "ewm" and ewm_span is not None:
        cov = returns.ewm(span=ewm_span).cov().iloc[-len(returns.columns):]
        # Reshape from multi-index to square matrix
        tickers = returns.columns.tolist()
        cov = cov.droplevel(0)
        cov = cov.loc[tickers, tickers]
    else:
        cov = returns.cov()

    if annualize:
        cov = cov * ppy

    return cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix of asset returns."""
    return returns.corr()


def rolling_correlation(
    returns: pd.DataFrame,
    window: int = 63,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations.

    Args:
        returns: Asset returns.
        window: Rolling window in periods.
        pairs: Specific pairs to compute. If None, computes all pairs.

    Returns:
        DataFrame with columns like "A_vs_B" for each pair.
    """
    tickers = returns.columns.tolist()
    if pairs is None:
        pairs = [(tickers[i], tickers[j])
                 for i in range(len(tickers))
                 for j in range(i + 1, len(tickers))]

    result = {}
    for a, b in pairs:
        if a in returns.columns and b in returns.columns:
            result[f"{a}_vs_{b}"] = returns[a].rolling(window).corr(returns[b])

    return pd.DataFrame(result)


def portfolio_variance(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    Portfolio variance = w' Σ w.

    Args:
        weights: 1D array of portfolio weights.
        cov_matrix: Annualized covariance matrix.

    Returns:
        Portfolio variance (annualized if cov_matrix is annualized).
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    return float(w @ sigma @ w)


def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """Portfolio volatility = sqrt(w' Σ w)."""
    return np.sqrt(portfolio_variance(weights, cov_matrix))


def downside_deviation(
    returns: pd.Series | pd.DataFrame,
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: int | None = None,
) -> pd.Series | float:
    """
    Downside deviation (semi-deviation below threshold).

    Only considers returns below the threshold.

    Args:
        returns: Period returns.
        threshold: Minimum acceptable return per period (default 0).
        annualize: If True, annualize.
    """
    ppy = periods_per_year or _infer_ppy(returns)
    diff = returns - threshold
    downside = diff.clip(upper=0)
    dd = np.sqrt((downside ** 2).mean())
    if annualize:
        dd = dd * np.sqrt(ppy)
    return dd


def max_drawdown(prices: pd.Series | pd.DataFrame) -> pd.Series | float:
    """
    Maximum drawdown from peak.

    Returns negative value (e.g., -0.35 = 35% drawdown).
    """
    if isinstance(prices, pd.DataFrame):
        return pd.Series({col: _max_dd_series(prices[col]) for col in prices.columns})
    return _max_dd_series(prices)


def _max_dd_series(p: pd.Series) -> float:
    """Max drawdown for a single series."""
    p_clean = p.dropna()
    if len(p_clean) < 2:
        return 0.0
    cummax = p_clean.cummax()
    drawdowns = (p_clean - cummax) / cummax
    return float(drawdowns.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Return the full drawdown series (negative values)."""
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def marginal_contribution_to_risk(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Marginal Contribution to Risk (MCR) for each asset.

    MCR_i = (Σ w)_i / σ_p

    Args:
        weights: Portfolio weights.
        cov_matrix: Covariance matrix (annualized).

    Returns:
        1D array of marginal risk contributions.
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol == 0:
        return np.zeros_like(w)
    return (sigma @ w) / port_vol


def component_contribution_to_risk(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Component Contribution to Risk (CCR) for each asset.

    CCR_i = w_i * MCR_i
    Sum of CCR = portfolio volatility.

    Returns:
        1D array; sum equals portfolio volatility.
    """
    w = np.asarray(weights, dtype=float)
    mcr = marginal_contribution_to_risk(w, cov_matrix)
    return w * mcr


def risk_contribution_pct(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Percentage risk contribution of each asset.

    Returns:
        1D array summing to 1.0.
    """
    ccr = component_contribution_to_risk(weights, cov_matrix)
    total = ccr.sum()
    if total == 0:
        return np.zeros_like(ccr)
    return ccr / total


def diversification_ratio(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    Diversification Ratio = weighted average asset vol / portfolio vol.

    DR > 1 means diversification is providing a volatility reduction.
    Higher is better.

    Interpretation:
        < 1.2 = poor diversification
        1.2 - 1.5 = moderate diversification
        1.5 - 2.0 = good diversification
        > 2.0 = excellent diversification
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    # Individual asset volatilities (sqrt of diagonal)
    asset_vols = np.sqrt(np.diag(sigma))
    weighted_avg_vol = np.dot(w, asset_vols)

    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol == 0:
        return 1.0

    return weighted_avg_vol / port_vol


def diversification_benefit(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    Diversification benefit = 1 - (portfolio_vol / weighted_avg_vol).

    Returns a value between 0 and 1. Higher = more diversification benefit.
    """
    dr = diversification_ratio(weights, cov_matrix)
    if dr == 0:
        return 0.0
    return 1.0 - (1.0 / dr)


def _infer_ppy(returns: pd.Series | pd.DataFrame) -> int:
    """Infer periods per year."""
    settings = get_settings()
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 10:
        median_gap = pd.Series(returns.index).diff().median()
        if median_gap is not None and hasattr(median_gap, "days"):
            if median_gap.days > 15:
                return settings.months_per_year
    return settings.trading_days_per_year
