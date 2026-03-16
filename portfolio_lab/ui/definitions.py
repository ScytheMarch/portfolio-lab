"""Educational definitions for every financial concept in Portfolio Lab.

Each entry provides a short tooltip, a longer explanation, an optional formula,
and interpretation thresholds so the UI can render colored badges and helper text.
"""

from portfolio_lab.ui.styles import glass_card, GREEN, YELLOW, RED, TEXT_SECONDARY

DEFINITIONS: dict[str, dict] = {
    # ── Portfolio Theory ──────────────────────────────────────────────────
    "expected_return": {
        "category": "Portfolio Theory",
        "short": "The annualized return the portfolio is projected to earn.",
        "long": (
            "Expected return is the weighted average of each asset's projected annual return. "
            "It represents the central tendency of outcomes — not a guarantee. Different estimation "
            "methods (arithmetic mean, geometric mean, CAPM, Fama-French) can produce very different "
            "numbers. Arithmetic mean tends to overstate future returns; geometric mean accounts for "
            "compounding drag; CAPM and FF5 are forward-looking factor-based estimates."
        ),
        "formula": "E[Rp] = sum(w_i * E[R_i])",
        "thresholds": {"excellent": 0.12, "good": 0.08, "adequate": 0.04},
    },
    "volatility": {
        "category": "Risk Metrics",
        "short": "Annualized standard deviation of returns — a measure of total risk.",
        "long": (
            "Volatility quantifies how much a portfolio's returns fluctuate around their average. "
            "Higher volatility means wider swings — both up and down. A portfolio with 15% volatility "
            "could reasonably gain or lose 15% in a year (one standard deviation). Diversification "
            "typically reduces portfolio volatility below the weighted average of individual asset vols, "
            "because assets don't move in perfect lockstep."
        ),
        "formula": "sigma_p = sqrt(w' * Cov * w)",
        "thresholds_inverted": True,
        "thresholds": {"low": 0.10, "moderate": 0.18, "high": 0.25},
    },
    "sharpe_ratio": {
        "category": "Risk Metrics",
        "short": "Excess return per unit of risk — the classic risk-adjusted performance measure.",
        "long": (
            "The Sharpe Ratio tells you how much extra return you're earning above the risk-free rate "
            "for each unit of volatility you're taking on. A Sharpe of 1.0 means you earn 1% of excess "
            "return for every 1% of volatility. Higher is better. Most diversified portfolios land "
            "between 0.3 and 1.0. Anything above 1.5 is exceptional and may reflect a short lookback "
            "period or survivorship bias."
        ),
        "formula": "SR = (E[Rp] - Rf) / sigma_p",
        "interpretation": {
            "< 0": ("Poor", RED),
            "0 - 0.5": ("Below Average", RED),
            "0.5 - 1.0": ("Adequate", YELLOW),
            "1.0 - 2.0": ("Good", GREEN),
            "> 2.0": ("Excellent", GREEN),
        },
        "thresholds": {"excellent": 2.0, "good": 1.0, "adequate": 0.5},
    },
    "sortino_ratio": {
        "category": "Risk Metrics",
        "short": "Like Sharpe, but only penalizes downside volatility — ignores upside swings.",
        "long": (
            "The Sortino Ratio improves on Sharpe by recognizing that investors don't mind upside "
            "volatility — only downside hurts. It divides excess return by downside deviation (the "
            "volatility of negative returns only). A portfolio that swings up a lot but rarely drops "
            "will have a much better Sortino than Sharpe. Generally, Sortino > 2.0 is strong."
        ),
        "formula": "Sortino = (E[Rp] - Rf) / Downside_Deviation",
        "thresholds": {"excellent": 3.0, "good": 2.0, "adequate": 1.0},
    },
    "max_drawdown": {
        "category": "Risk Metrics",
        "short": "The largest peak-to-trough decline in portfolio value.",
        "long": (
            "Max drawdown measures the worst-case loss from a peak to the subsequent trough. "
            "A -30% drawdown means at some point your portfolio lost 30% of its peak value before "
            "recovering. This is critical for understanding tail risk — even if average returns are "
            "good, a deep drawdown can wipe out years of gains and cause panic selling."
        ),
        "formula": "MDD = min(V_t / max(V_s for s <= t) - 1)",
        "thresholds_inverted": True,
    },
    "diversification_ratio": {
        "category": "Portfolio Theory",
        "short": "How much diversification benefit your portfolio captures.",
        "long": (
            "The diversification ratio compares the weighted average of individual asset volatilities "
            "to the portfolio's actual volatility. If it's 1.0, there's no diversification benefit — "
            "assets move in perfect lockstep. Above 1.0 means correlations are working in your favor. "
            "A ratio of 1.5 means your portfolio is 33% less volatile than a naive combination would "
            "suggest. Higher is better."
        ),
        "formula": "DR = sum(w_i * sigma_i) / sigma_p",
        "interpretation": {
            "< 1.0": ("No Benefit", RED),
            "1.0 - 1.2": ("Low", YELLOW),
            "1.2 - 1.5": ("Moderate", GREEN),
            "> 1.5": ("Strong", GREEN),
        },
        "thresholds": {"excellent": 1.5, "good": 1.2, "adequate": 1.0},
    },
    "income_yield": {
        "category": "Income",
        "short": "Weighted average dividend yield of the portfolio.",
        "long": (
            "Income yield is the percentage of your portfolio value that is paid out as dividends "
            "or distributions annually. A 2% yield on a $100,000 portfolio means ~$2,000/year in "
            "income before taxes. High yield isn't always good — it can signal distressed companies "
            "or REITs with unsustainable payouts. Focus on yield sustainability, not just the number."
        ),
        "formula": "Yield_p = sum(w_i * yield_i)",
    },
    "expense_ratio": {
        "category": "Income",
        "short": "Weighted average annual fee charged by the funds in your portfolio.",
        "long": (
            "The expense ratio is what funds charge you annually for management. A 0.03% ER (like VTI) "
            "costs $3 per $10,000 invested. A 1.0% ER costs $100. Over 30 years, a 1% fee drag can "
            "reduce your ending wealth by 25%+. Keep expenses low — they are the one return predictor "
            "that actually works consistently."
        ),
        "formula": "ER_p = sum(w_i * ER_i)",
    },

    # ── Factor Models ─────────────────────────────────────────────────────
    "alpha": {
        "category": "Factor Models",
        "short": "Return above what the factor model predicts — the 'unexplained' outperformance.",
        "long": (
            "Alpha is the intercept of a factor regression. It represents the portion of returns not "
            "explained by exposure to known risk factors. Positive alpha means the portfolio (or asset) "
            "earned more than its factor loadings would predict. However, alpha is only meaningful if "
            "it's statistically significant (|t-stat| > 2.0). Most portfolios have alpha "
            "indistinguishable from zero."
        ),
        "formula": "R - Rf = alpha + beta1*MktRF + beta2*SMB + ... + epsilon",
    },
    "r_squared": {
        "category": "Factor Models",
        "short": "How much of the portfolio's return variation is explained by the factor model.",
        "long": (
            "R-squared ranges from 0 to 1. An R-squared of 0.90 means 90% of your portfolio's daily "
            "return movements can be explained by the five Fama-French factors. The remaining 10% is "
            "idiosyncratic (stock-specific) risk. Well-diversified portfolios typically have R-squared "
            "above 0.85. Individual stocks are much lower (0.2-0.6)."
        ),
        "interpretation": {
            "< 0.5": ("Low — model explains little", YELLOW),
            "0.5 - 0.8": ("Moderate", GREEN),
            "> 0.8": ("High — well explained", GREEN),
        },
        "thresholds": {"excellent": 0.8, "good": 0.5, "adequate": 0.3},
    },
    "mkt_rf": {
        "category": "Factor Models",
        "short": "Market Risk Premium — how much the portfolio moves with the overall stock market.",
        "long": (
            "Mkt-RF (Market minus Risk-Free) measures your sensitivity to the broad equity market. "
            "A loading of 1.0 means you move 1:1 with the market. Below 1.0 means you're less "
            "sensitive (often because you hold bonds or defensive assets). Above 1.0 means you "
            "amplify market moves — both up and down. This is typically the dominant factor."
        ),
    },
    "smb": {
        "category": "Factor Models",
        "short": "Small Minus Big — exposure to small-cap stocks vs large-cap stocks.",
        "long": (
            "SMB captures the size premium. A positive loading means your portfolio tilts toward "
            "smaller companies, which historically have earned higher returns (but with more risk and "
            "long periods of underperformance). Negative means you tilt toward large-caps. The size "
            "premium has been weak in recent decades, especially in U.S. markets."
        ),
    },
    "hml": {
        "category": "Factor Models",
        "short": "High Minus Low — exposure to value stocks (cheap) vs growth stocks (expensive).",
        "long": (
            "HML captures the value premium. Positive loading means you tilt toward stocks with high "
            "book-to-market ratios (value stocks — think banks, energy, industrials). Negative means "
            "you tilt toward growth stocks (tech, biotech). The value premium averaged ~3-5% annually "
            "over the long run but has been negative for much of 2010-2020."
        ),
    },
    "rmw": {
        "category": "Factor Models",
        "short": "Robust Minus Weak — exposure to highly profitable companies.",
        "long": (
            "RMW captures the profitability premium. Positive loading means you hold companies with "
            "high operating profitability (robust margins). Negative means you hold more speculative, "
            "less profitable companies. High-profitability stocks have historically outperformed, "
            "and this premium has been more stable than size or value."
        ),
    },
    "cma": {
        "category": "Factor Models",
        "short": "Conservative Minus Aggressive — exposure to companies that invest conservatively.",
        "long": (
            "CMA captures the investment premium. Positive loading means your portfolio tilts toward "
            "companies that reinvest conservatively (retain earnings, pay dividends). Negative means "
            "you hold companies that spend aggressively on expansion (high capex, M&A). Conservative "
            "firms have historically earned higher risk-adjusted returns."
        ),
    },
    "fama_french": {
        "category": "Factor Models",
        "short": "A model explaining returns through 5 systematic risk factors.",
        "long": (
            "The Fama-French Five-Factor Model decomposes stock returns into five systematic risk "
            "factors: Market (Mkt-RF), Size (SMB), Value (HML), Profitability (RMW), and Investment "
            "(CMA). The idea is that stocks earn higher returns not because of luck or skill, but "
            "because they are exposed to these risk factors. Understanding your factor exposures tells "
            "you what risks you're actually being paid for."
        ),
    },

    # ── Simulation ────────────────────────────────────────────────────────
    "monte_carlo": {
        "category": "Simulation",
        "short": "Thousands of random simulations to estimate the range of possible outcomes.",
        "long": (
            "Monte Carlo simulation generates thousands of possible future return paths by randomly "
            "sampling from the historical return distribution. Each simulation produces a different "
            "ending portfolio value. By looking at the distribution of outcomes, you can estimate "
            "probabilities — 'What's the chance I lose money?' or 'What's the chance I hit my goal?' "
            "It captures uncertainty that single-point forecasts miss."
        ),
    },
    "efficient_frontier": {
        "category": "Portfolio Theory",
        "short": "The set of portfolios offering the highest return for each level of risk.",
        "long": (
            "The efficient frontier is a curve showing the best possible risk-return tradeoff. "
            "Portfolios on the frontier are 'efficient' — you can't get more return without taking "
            "more risk, and you can't reduce risk without giving up return. Portfolios below the "
            "frontier are suboptimal. The tangency point (where a line from the risk-free rate touches "
            "the frontier) is the maximum Sharpe ratio portfolio."
        ),
    },
    "capm": {
        "category": "Portfolio Theory",
        "short": "Capital Asset Pricing Model — expected return based on market beta alone.",
        "long": (
            "CAPM says an asset's expected return equals the risk-free rate plus its beta times the "
            "market risk premium. E[R] = Rf + beta * (E[Rm] - Rf). It's the simplest factor model "
            "(just one factor: the market). CAPM produces more conservative, forward-looking return "
            "estimates than historical means, making it useful for long-term planning."
        ),
        "formula": "E[R] = Rf + beta * (E[Rm] - Rf)",
    },

    # ── Optimization ──────────────────────────────────────────────────────
    "max_sharpe": {
        "category": "Optimization",
        "short": "Find the portfolio with the highest risk-adjusted return (Sharpe Ratio).",
        "long": (
            "Maximizing Sharpe finds the allocation that delivers the most excess return per unit of "
            "risk. This is the tangency portfolio on the efficient frontier. It's the most popular "
            "objective but can concentrate heavily in a few assets if weight constraints are loose."
        ),
    },
    "min_volatility": {
        "category": "Optimization",
        "short": "Find the portfolio with the lowest possible risk.",
        "long": (
            "Minimum volatility finds the allocation that minimizes total portfolio risk regardless of "
            "return. This typically overweights bonds, treasuries, and low-beta assets. It's useful for "
            "conservative investors who prioritize capital preservation over growth."
        ),
    },
    "return_method_arithmetic": {
        "category": "Portfolio Theory",
        "short": "Simple average of historical returns — tends to overstate future returns.",
        "long": (
            "The arithmetic mean just averages all historical period returns. It's mathematically the "
            "best unbiased estimator of expected single-period return, but it overstates what you'll "
            "actually earn over time because it ignores compounding drag (volatility tax). Use it for "
            "single-period analysis; prefer geometric mean or CAPM for multi-year projections."
        ),
    },
    "return_method_geometric": {
        "category": "Portfolio Theory",
        "short": "Compound growth rate — what you actually earned historically.",
        "long": (
            "The geometric mean is the annualized compound growth rate. It accounts for the fact that "
            "a -50% loss followed by a +50% gain doesn't get you back to even — you're still down 25%. "
            "It's always lower than the arithmetic mean (the gap is roughly half the variance). Use it "
            "when you want to know 'what did this actually return over the period?'"
        ),
    },
}


def explain(key: str) -> str:
    """Return the short tooltip for a concept."""
    entry = DEFINITIONS.get(key, {})
    return entry.get("short", "")


def why_it_matters(key: str) -> str:
    """Return the long educational explanation."""
    entry = DEFINITIONS.get(key, {})
    return entry.get("long", "")


def get_formula(key: str) -> str | None:
    """Return the formula if available."""
    entry = DEFINITIONS.get(key, {})
    return entry.get("formula")


def interpret_value(key: str, value: float) -> tuple[str, str]:
    """Return (label, color) interpretation for a numeric value."""
    entry = DEFINITIONS.get(key, {})
    interp = entry.get("interpretation", {})
    if not interp:
        return ("", "")
    # Return the last matching bracket
    result = ("", "")
    for bracket, (label, color) in interp.items():
        result = (label, color)  # fallback to last
    return result


def get_interpretation_scale(key: str) -> dict:
    """Return the full interpretation scale dict."""
    entry = DEFINITIONS.get(key, {})
    return entry.get("interpretation", {})


def get_by_category(category: str) -> dict[str, dict]:
    """Return all definitions in a category."""
    return {k: v for k, v in DEFINITIONS.items() if v.get("category") == category}


def all_categories() -> list[str]:
    """Return sorted list of all categories."""
    cats = set(v.get("category", "") for v in DEFINITIONS.values())
    return sorted(c for c in cats if c)


def render_definition_card(key: str) -> str:
    """Render a glass card HTML block for a definition."""
    entry = DEFINITIONS.get(key, {})
    if not entry:
        return ""

    title = key.replace("_", " ").title()
    short = entry.get("short", "")
    long_text = entry.get("long", "")
    formula = entry.get("formula")

    inner = (
        f'<div style="margin-bottom:6px">'
        f'<span style="color:#f1f5f9;font-weight:700;font-size:1.05em">{title}</span>'
        f'<span style="color:#64748b;font-size:0.85em;margin-left:8px">{entry.get("category", "")}</span>'
        f'</div>'
        f'<div style="color:#94a3b8;font-size:0.9em;margin-bottom:8px">{short}</div>'
        f'<div style="color:#64748b;font-size:0.85em;line-height:1.5">{long_text}</div>'
    )

    if formula:
        inner += (
            f'<div style="margin-top:8px;padding:6px 12px;background:rgba(99,102,241,0.08);'
            f'border-radius:8px;font-family:monospace;font-size:0.85em;color:#a78bfa">'
            f'{formula}</div>'
        )

    # Interpretation scale
    interp = entry.get("interpretation", {})
    if interp:
        scale_items = ""
        for bracket, (label, color) in interp.items():
            scale_items += (
                f'<span style="display:inline-block;margin-right:12px;font-size:0.8em">'
                f'<span style="color:{color};font-weight:600">{bracket}</span> '
                f'<span style="color:#64748b">{label}</span></span>'
            )
        inner += f'<div style="margin-top:8px">{scale_items}</div>'

    return glass_card(inner)
