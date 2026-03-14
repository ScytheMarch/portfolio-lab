"""
Plotly chart builders for portfolio_lab.

Each function takes structured data and returns a Plotly figure.
No analytics logic here — charts only render.
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def efficient_frontier_chart(
    ef_vols: np.ndarray,
    ef_rets: np.ndarray,
    max_sharpe_idx: int,
    min_vol_idx: int,
    current_vol: Optional[float] = None,
    current_ret: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> go.Figure:
    """Plot efficient frontier with highlighted portfolios."""
    fig = go.Figure()

    # Frontier line
    fig.add_trace(go.Scatter(
        x=ef_vols * 100, y=ef_rets * 100,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#1f77b4", width=3),
    ))

    # Max Sharpe point
    fig.add_trace(go.Scatter(
        x=[ef_vols[max_sharpe_idx] * 100],
        y=[ef_rets[max_sharpe_idx] * 100],
        mode="markers",
        name="Max Sharpe",
        marker=dict(color="red", size=14, symbol="star"),
    ))

    # Min Vol point
    fig.add_trace(go.Scatter(
        x=[ef_vols[min_vol_idx] * 100],
        y=[ef_rets[min_vol_idx] * 100],
        mode="markers",
        name="Min Volatility",
        marker=dict(color="green", size=14, symbol="diamond"),
    ))

    # Current portfolio
    if current_vol is not None and current_ret is not None:
        fig.add_trace(go.Scatter(
            x=[current_vol * 100], y=[current_ret * 100],
            mode="markers",
            name="Current Portfolio",
            marker=dict(color="orange", size=14, symbol="circle"),
        ))

    # Capital Market Line
    if risk_free_rate > 0 and len(ef_vols) > 0:
        max_sharpe_vol = ef_vols[max_sharpe_idx]
        max_sharpe_ret = ef_rets[max_sharpe_idx]
        if max_sharpe_vol > 0:
            sharpe = (max_sharpe_ret - risk_free_rate) / max_sharpe_vol
            cml_vols = np.linspace(0, max(ef_vols) * 1.1, 50)
            cml_rets = risk_free_rate + sharpe * cml_vols
            fig.add_trace(go.Scatter(
                x=cml_vols * 100, y=cml_rets * 100,
                mode="lines",
                name="Capital Market Line",
                line=dict(color="gray", dash="dash", width=1),
            ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_white",
        height=500,
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plot correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_white",
        height=500,
    )
    return fig


def allocation_pie_chart(weights: dict[str, float]) -> go.Figure:
    """Pie chart of portfolio weights."""
    # Filter out zero weights
    filtered = {k: v for k, v in weights.items() if v > 0.001}
    fig = go.Figure(data=[go.Pie(
        labels=list(filtered.keys()),
        values=list(filtered.values()),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
    )])
    fig.update_layout(
        title="Portfolio Allocation",
        template="plotly_white",
        height=400,
    )
    return fig


def monte_carlo_fan_chart(
    nominal_paths: np.ndarray,
    horizon_years: int,
    initial_investment: float,
    percentiles: list[int] = [10, 25, 50, 75, 90],
    max_paths_to_plot: int = 200,
) -> go.Figure:
    """Monte Carlo fan chart showing percentile bands."""
    n_periods = nominal_paths.shape[1]
    x = np.linspace(0, horizon_years, n_periods)

    fig = go.Figure()

    # Percentile bands
    colors = ["rgba(31,119,180,0.1)", "rgba(31,119,180,0.2)",
              "rgba(31,119,180,0.3)", "rgba(31,119,180,0.2)", "rgba(31,119,180,0.1)"]

    pctls = {}
    for p in percentiles:
        pctls[p] = np.percentile(nominal_paths, p, axis=0)

    # Fill between bands
    pairs = [(10, 90), (25, 75)]
    fill_colors = ["rgba(31,119,180,0.15)", "rgba(31,119,180,0.3)"]

    for (lo, hi), color in zip(pairs, fill_colors):
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([pctls[hi], pctls[lo][::-1]]),
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{lo}th-{hi}th Percentile",
            showlegend=True,
        ))

    # Median line
    fig.add_trace(go.Scatter(
        x=x, y=pctls[50],
        mode="lines",
        name="Median",
        line=dict(color="#1f77b4", width=3),
    ))

    # Initial investment reference
    fig.add_hline(
        y=initial_investment,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Investment",
    )

    fig.update_layout(
        title="Monte Carlo Simulation - Nominal Outcomes",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500,
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def nominal_vs_real_chart(
    nominal_paths: np.ndarray,
    real_paths: np.ndarray,
    horizon_years: int,
) -> go.Figure:
    """Compare nominal vs real (inflation-adjusted) median paths."""
    n_periods = nominal_paths.shape[1]
    x = np.linspace(0, horizon_years, n_periods)

    nom_median = np.median(nominal_paths, axis=0)
    real_median = np.median(real_paths, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=nom_median, mode="lines",
        name="Nominal (Median)", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=real_median, mode="lines",
        name="Real / Inflation-Adjusted (Median)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Nominal vs Real Outcomes (Median)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400,
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def drawdown_chart(prices: pd.Series, title: str = "Portfolio Drawdown") -> go.Figure:
    """Plot drawdown series."""
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="red", width=1),
        fillcolor="rgba(255,0,0,0.2)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=350,
    )
    return fig


def risk_contribution_chart(risk_pct: dict[str, float]) -> go.Figure:
    """Bar chart of risk contribution percentages."""
    sorted_risk = sorted(risk_pct.items(), key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in sorted_risk]
    values = [v * 100 for _, v in sorted_risk]

    fig = go.Figure(data=[go.Bar(
        x=tickers, y=values,
        marker_color=["#ff4444" if v > 35 else "#1f77b4" for v in values],
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    )])
    fig.update_layout(
        title="Risk Contribution by Asset (%)",
        xaxis_title="Asset",
        yaxis_title="% of Portfolio Risk",
        template="plotly_white",
        height=400,
    )
    return fig


def factor_exposure_chart(factor_exposures: dict[str, float]) -> go.Figure:
    """Bar chart of factor exposures/betas."""
    factors = list(factor_exposures.keys())
    values = list(factor_exposures.values())

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    fig = go.Figure(data=[go.Bar(
        x=factors, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="auto",
    )])
    fig.update_layout(
        title="Fama-French 5-Factor Exposures",
        xaxis_title="Factor",
        yaxis_title="Beta Loading",
        template="plotly_white",
        height=400,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
    return fig


def goal_hit_rate_gauge(hit_rate: float, label: str = "Goal Hit Rate") -> go.Figure:
    """Gauge chart for goal hit rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hit_rate * 100,
        title={"text": label},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 65], "color": "#ffffcc"},
                {"range": [65, 80], "color": "#ccffcc"},
                {"range": [80, 100], "color": "#99ff99"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=300, template="plotly_white")
    return fig


def income_projection_chart(
    years: list[int],
    nominal_income: list[float],
    real_income: list[float],
) -> go.Figure:
    """Line chart of projected income."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=nominal_income, mode="lines+markers",
        name="Nominal Income", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=real_income, mode="lines+markers",
        name="Real Income", line=dict(color="#ff7f0e", dash="dash"),
    ))
    fig.update_layout(
        title="Projected Annual Income",
        xaxis_title="Year",
        yaxis_title="Income ($)",
        template="plotly_white",
        height=400,
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def terminal_value_distribution_chart(
    terminal_values: np.ndarray,
    initial_investment: float,
    target_value: Optional[float] = None,
) -> go.Figure:
    """
    Histogram of Monte Carlo terminal values with std deviation bands,
    percentile markers, and loss/goal miss probabilities.
    """
    n = len(terminal_values)
    mean = float(np.mean(terminal_values))
    std = float(np.std(terminal_values))
    median = float(np.median(terminal_values))
    p10 = float(np.percentile(terminal_values, 10))
    p90 = float(np.percentile(terminal_values, 90))

    # Loss and goal probabilities
    prob_loss = float(np.sum(terminal_values < initial_investment) / n) * 100
    goal_miss_pct = None
    if target_value is not None and target_value > 0:
        goal_miss_pct = float(np.sum(terminal_values < target_value) / n) * 100

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=80,
        marker_color="rgba(31,119,180,0.6)",
        name="Terminal Values",
        hovertemplate="$%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Mean line
    fig.add_vline(x=mean, line_dash="solid", line_color="#ff7f0e", line_width=2,
                  annotation_text=f"Mean: ${mean:,.0f}", annotation_position="top right")

    # Median line
    fig.add_vline(x=median, line_dash="dash", line_color="#2ca02c", line_width=2,
                  annotation_text=f"Median: ${median:,.0f}", annotation_position="top left")

    # +/- 1 std bands
    fig.add_vrect(x0=mean - std, x1=mean + std,
                  fillcolor="rgba(255,127,14,0.08)", line_width=0,
                  annotation_text="1 Std Dev", annotation_position="top left")

    # +/- 2 std bands
    lo_2 = max(mean - 2 * std, min(terminal_values))
    fig.add_vrect(x0=lo_2, x1=mean + 2 * std,
                  fillcolor="rgba(255,127,14,0.04)", line_width=0)

    # 10th / 90th percentile markers
    fig.add_vline(x=p10, line_dash="dot", line_color="#d62728", line_width=1,
                  annotation_text=f"10th: ${p10:,.0f}", annotation_position="bottom left")
    fig.add_vline(x=p90, line_dash="dot", line_color="#d62728", line_width=1,
                  annotation_text=f"90th: ${p90:,.0f}", annotation_position="bottom right")

    # Initial investment reference with loss probability
    loss_label = f"Initial: ${initial_investment:,.0f} | {prob_loss:.1f}% lost money"
    fig.add_vline(x=initial_investment, line_dash="dashdot", line_color="gray", line_width=2,
                  annotation_text=loss_label)

    # Target value with miss rate
    if target_value is not None and target_value > 0 and goal_miss_pct is not None:
        target_label = (
            f"Target: ${target_value:,.0f} | "
            f"{goal_miss_pct:.1f}% missed goal"
        )
        fig.add_vline(x=target_value, line_dash="solid", line_color="#9467bd", line_width=2,
                      annotation_text=target_label,
                      annotation_position="top right")

    # Build subtitle with key stats
    subtitle_parts = [f"Prob. of Loss: {prob_loss:.1f}%"]
    if goal_miss_pct is not None:
        subtitle_parts.append(f"Goal Miss Rate: {goal_miss_pct:.1f}%")
    subtitle = " | ".join(subtitle_parts)

    fig.update_layout(
        title=f"Terminal Value Distribution (Confidence Intervals)<br><sup>{subtitle}</sup>",
        xaxis_title="Terminal Portfolio Value ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
        xaxis=dict(tickformat="$,.0f"),
        showlegend=False,
    )
    return fig


def factor_attribution_chart(attribution: dict[str, float]) -> go.Figure:
    """Stacked bar chart decomposing portfolio return into factor contributions."""
    factors = list(attribution.keys())
    values = [v * 100 for v in attribution.values()]

    colors = {
        "Alpha": "#2ca02c",
        "Mkt-RF": "#1f77b4",
        "SMB": "#ff7f0e",
        "HML": "#d62728",
        "RMW": "#9467bd",
        "CMA": "#8c564b",
        "Risk-Free": "#7f7f7f",
        "Residual": "#bcbd22",
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=factors,
        y=values,
        marker_color=[colors.get(f, "#17becf") for f in factors],
        text=[f"{v:+.2f}%" for v in values],
        textposition="auto",
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

    total = sum(values)
    fig.update_layout(
        title=f"Factor Return Attribution (Total: {total:.2f}%)",
        xaxis_title="Factor",
        yaxis_title="Annualized Contribution (%)",
        template="plotly_white",
        height=450,
        showlegend=False,
    )
    return fig


def factor_premium_history_chart(cumulative_returns: pd.DataFrame) -> go.Figure:
    """Line chart of cumulative factor returns over time."""
    colors = {
        "Mkt-RF": "#1f77b4",
        "SMB": "#ff7f0e",
        "HML": "#d62728",
        "RMW": "#9467bd",
        "CMA": "#8c564b",
    }

    fig = go.Figure()
    for col in cumulative_returns.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col].values,
            mode="lines",
            name=col,
            line=dict(color=colors.get(col, "#17becf"), width=2),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", line_width=1,
                  annotation_text="$1 Starting Value")

    fig.update_layout(
        title="Cumulative Factor Returns (Growth of $1)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return ($)",
        template="plotly_white",
        height=500,
        yaxis=dict(tickformat="$.2f"),
    )
    return fig


def factor_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of correlations between FF5 factors."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 13},
    ))
    fig.update_layout(
        title="Factor Correlation Matrix (FF5)",
        template="plotly_white",
        height=450,
        width=550,
    )
    return fig


def factor_regression_summary_chart(
    portfolio_loadings: dict[str, float],
    portfolio_t_stats: dict[str, float],
) -> go.Figure:
    """Bar chart of portfolio factor loadings with significance markers."""
    factors = list(portfolio_loadings.keys())
    betas = list(portfolio_loadings.values())
    t_stats = [portfolio_t_stats.get(f, 0) for f in factors]

    colors = ["#2ca02c" if abs(t) > 2.0 else "#aaaaaa" for t in t_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=factors,
        y=betas,
        marker_color=colors,
        text=[f"{b:.3f}\n(t={t:.1f})" for b, t in zip(betas, t_stats)],
        textposition="auto",
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

    fig.update_layout(
        title="Portfolio Factor Loadings (green = significant at 95%)",
        xaxis_title="Factor",
        yaxis_title="Beta Loading",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def scenario_comparison_chart(
    scenario_results: dict[str, float],
    base_return: float,
) -> go.Figure:
    """Bar chart comparing portfolio return across factor scenarios."""
    names = list(scenario_results.keys())
    returns = [v * 100 for v in scenario_results.values()]

    colors = ["#2ca02c" if r > base_return * 100 else "#d62728" for r in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=returns,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in returns],
        textposition="auto",
    ))

    fig.add_hline(y=base_return * 100, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text=f"Base Case: {base_return*100:.1f}%")

    fig.update_layout(
        title="Scenario Analysis: Implied Portfolio Returns",
        xaxis_title="Scenario",
        yaxis_title="Implied Annual Return (%)",
        template="plotly_white",
        height=450,
        showlegend=False,
    )
    return fig


def rolling_performance_chart(
    returns: pd.DataFrame,
    window: int = 252,
) -> go.Figure:
    """Rolling annualized return chart."""
    rolling_ann = returns.rolling(window).mean() * 252
    fig = go.Figure()
    for col in rolling_ann.columns:
        fig.add_trace(go.Scatter(
            x=rolling_ann.index, y=rolling_ann[col].values * 100,
            mode="lines", name=col,
        ))
    fig.update_layout(
        title=f"Rolling {window}-Day Annualized Return",
        xaxis_title="Date",
        yaxis_title="Annualized Return (%)",
        template="plotly_white",
        height=400,
    )
    return fig
