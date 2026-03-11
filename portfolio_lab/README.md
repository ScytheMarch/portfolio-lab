# Portfolio Lab

A portfolio construction, optimization, simulation, and diagnostics platform for ETFs, mutual funds, and U.S. Treasuries.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r portfolio_lab/requirements.txt

# Run the application
streamlit run portfolio_lab/app.py

# Run tests
pytest portfolio_lab/tests/ -v
```

## Architecture

```
portfolio_lab/
├── app.py                  # Streamlit entry point
├── config/settings.py      # Global defaults, all assumptions explicit
├── data/                   # Data fetching layer (yfinance + French data library)
│   ├── market_data.py      # Price and dividend data
│   ├── fund_metadata.py    # Expense ratios, yields, categories
│   ├── treasury_data.py    # Risk-free rate, Treasury proxies
│   └── factor_data.py      # Fama-French 5-factor data
├── analytics/              # Core computation engine
│   ├── returns.py          # Arithmetic, geometric, CAGR, log returns
│   ├── risk.py             # Covariance, correlation, drawdown, risk contributions
│   ├── performance.py      # Sharpe, Sortino ratios
│   ├── factor_model.py     # FF5 regressions, factor-implied returns
│   ├── income.py           # Yield, fee-adjusted income projections
│   ├── inflation.py        # Real return, purchasing power analysis
│   ├── reporting.py        # Deterministic post-simulation report builder
│   └── diagnostics.py      # Rule-based stress flags and takeaways
├── optimization/           # Portfolio optimization
│   ├── objectives.py       # Objective functions (Sharpe, vol, income)
│   ├── constraints.py      # Constraint builders
│   ├── solver.py           # scipy.optimize wrapper with multi-start
│   └── efficient_frontier.py
├── simulation/             # Monte Carlo engine
│   ├── monte_carlo.py      # GBM parametric + historical bootstrap
│   ├── withdrawal_model.py # Contribution/withdrawal schedules
│   └── scenario_engine.py  # End-to-end scenario orchestration
├── ui/                     # Streamlit presentation layer
│   ├── components.py       # Input widgets
│   ├── charts.py           # Plotly chart builders
│   └── pages.py            # Report section renderers
└── tests/                  # pytest test suite
```

## Features

### Return Calculations
- Arithmetic mean, geometric mean, CAGR — clearly distinguished
- Log returns for compounding-consistent analysis
- Exponentially weighted estimates
- Multiple expected return methods (selectable in UI)

### Risk Analytics
- Full covariance/correlation matrix
- Portfolio variance via w'Σw
- Marginal and component risk contributions
- Maximum drawdown
- Downside deviation

### Diversification Quality Score
- Diversification Ratio = weighted avg asset vol / portfolio vol
- Bands: Poor (<1.2), Moderate (1.2-1.5), Good (1.5-2.0), Excellent (>2.0)
- Identifies most/least diversifying assets

### Optimization
- **Max Sharpe**: Maximizes risk-adjusted returns
- **Min Volatility**: Lowest-risk feasible portfolio
- **Target Return**: Minimum vol for a specified return
- **Max Income / Max Net Income**: Yield-focused with expense awareness
- Constraints: weight bounds, max vol, min yield, min diversification

### Monte Carlo Simulation
- GBM with Jensen's inequality adjustment (drift = μ - σ²/2)
- Fixed or stochastic inflation
- Configurable rebalancing frequency
- Contribution and withdrawal support
- Percentile bands, drawdown statistics, goal probability

### Fama-French 5-Factor Model
- Market, SMB, HML, RMW, CMA factor regressions
- Alpha, factor loadings, R-squared, t-statistics
- Portfolio-level factor exposures from weights
- Factor-implied expected return estimates

### Deterministic Post-Simulation Report
Nine sections generated entirely from rule-based logic:
1. Portfolio Summary
2. Monte Carlo Outcome Summary (with goal hit/miss rates)
3. Return Driver Breakdown
4. Risk Driver Breakdown
5. Diversification Breakdown
6. Inflation Impact Breakdown
7. Goal Attainment & Probability Analysis
8. Stress Flags / Weaknesses
9. Key Takeaways

No AI/LLM commentary. All outputs are deterministic.

## Key Assumptions & Limitations

- **Data source**: yfinance. Subject to availability, delays, and data quality.
- **Expense ratios**: Not always available via yfinance. Missing ERs generate explicit warnings — they are NOT silently assumed to be zero.
- **Risk-free rate**: Fetched from ^IRX (13-week T-bill). Falls back to BIL ETF proxy or 5% default.
- **Treasury instruments**: Represented via ETF proxies (BIL, SHV, IEF, TLT, TIP). Direct bond analytics (duration, convexity) are limited to proxy-based estimates.
- **Factor data**: Downloaded from Kenneth French's data library. Network availability required.
- **Monte Carlo**: Uses geometric Brownian motion (parametric). Historical bootstrap also available. Neither captures fat tails, regime changes, or autocorrelation.
- **Optimization**: Uses scipy SLSQP with multi-start. Not a global optimizer — complex constraint sets may find local optima.
- **Expected returns are NOT predictions**. Historical geometric mean is realized performance, not forward guidance.
- **Not investment advice**.

## Demo Portfolio

Default tickers: VTI, VXUS, BND, VNQ, GLD, TLT, BIL

## TODOs for Advanced Enhancements

- [ ] Black-Litterman expected return model
- [ ] CVaR / Conditional Value at Risk optimization
- [ ] Risk parity objective
- [ ] Direct Treasury bond analytics (duration, convexity from FRED/Treasury API)
- [ ] Ledoit-Wolf shrinkage covariance estimator
- [ ] Regime-switching Monte Carlo (bull/bear states)
- [ ] Tax-aware optimization (after-tax returns)
- [ ] Multi-period dynamic optimization
- [ ] FastAPI + React frontend migration
- [ ] PDF report export
- [ ] Benchmark comparison (vs 60/40, S&P 500)
- [ ] Stress testing with historical scenarios (2008, COVID, etc.)
- [ ] Custom factor model support
- [ ] Asset class constraints in optimization
- [ ] Rebalancing cost modeling
