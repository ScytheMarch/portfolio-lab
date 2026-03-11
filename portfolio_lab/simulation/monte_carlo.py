"""
Monte Carlo simulation engine.

Supports:
- Parametric simulation (expected returns + covariance)
- Optional historical bootstrap
- Inflation-aware (fixed or stochastic)
- Configurable rebalancing frequency
- Contributions and withdrawals
- Terminal value statistics and goal probability analysis
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from portfolio_lab.analytics.inflation import generate_stochastic_inflation

logger = logging.getLogger(__name__)

REBALANCE_MAP = {
    "annual": 1,
    "semi-annual": 2,
    "quarterly": 4,
    "monthly": 12,
}


@dataclass
class MonteCarloResult:
    """Structured Monte Carlo simulation output."""
    # Path data: (n_simulations, n_periods+1) including initial value
    nominal_paths: np.ndarray
    real_paths: np.ndarray
    inflation_paths: np.ndarray

    # Terminal values
    nominal_terminal: np.ndarray  # (n_simulations,)
    real_terminal: np.ndarray

    # Configuration used
    n_simulations: int = 0
    horizon_years: int = 0
    initial_investment: float = 0.0
    inflation_rate: float = 0.0
    stochastic_inflation: bool = False
    rebalance_frequency: str = "annual"

    # Statistics
    stats: dict = field(default_factory=dict)

    def compute_stats(self, target_value: Optional[float] = None):
        """Compute summary statistics from terminal values."""
        nom = self.nominal_terminal
        real = self.real_terminal
        initial = self.initial_investment

        self.stats = {
            "nominal_mean": float(np.mean(nom)),
            "nominal_median": float(np.median(nom)),
            "nominal_p10": float(np.percentile(nom, 10)),
            "nominal_p25": float(np.percentile(nom, 25)),
            "nominal_p75": float(np.percentile(nom, 75)),
            "nominal_p90": float(np.percentile(nom, 90)),
            "nominal_min": float(np.min(nom)),
            "nominal_max": float(np.max(nom)),
            "real_mean": float(np.mean(real)),
            "real_median": float(np.median(real)),
            "real_p10": float(np.percentile(real, 10)),
            "real_p25": float(np.percentile(real, 25)),
            "real_p75": float(np.percentile(real, 75)),
            "real_p90": float(np.percentile(real, 90)),
            "prob_nominal_loss": float(np.mean(nom < initial)),
            "prob_real_loss": float(np.mean(real < initial)),
        }

        if target_value is not None:
            self.stats["target_value"] = target_value
            self.stats["goal_hit_rate"] = float(np.mean(nom >= target_value))
            self.stats["goal_miss_rate"] = 1.0 - self.stats["goal_hit_rate"]

        # Drawdown stats from paths
        self._compute_drawdown_stats()

    def _compute_drawdown_stats(self):
        """Compute max drawdown statistics across simulations."""
        # Compute max drawdown for each path
        max_dds = []
        for i in range(self.nominal_paths.shape[0]):
            path = self.nominal_paths[i]
            cummax = np.maximum.accumulate(path)
            dd = (path - cummax) / np.where(cummax > 0, cummax, 1)
            max_dds.append(float(np.min(dd)))

        max_dds = np.array(max_dds)
        self.stats["max_drawdown_mean"] = float(np.mean(max_dds))
        self.stats["max_drawdown_median"] = float(np.median(max_dds))
        self.stats["max_drawdown_p10"] = float(np.percentile(max_dds, 10))  # worst 10%
        self.stats["max_drawdown_p90"] = float(np.percentile(max_dds, 90))  # best 10%


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for portfolio projections.
    """

    def __init__(
        self,
        expected_return: float,
        volatility: float,
        initial_investment: float = 100_000,
        horizon_years: int = 10,
        n_simulations: int = 5000,
        inflation_rate: float = 0.03,
        stochastic_inflation: bool = False,
        inflation_std: float = 0.01,
        rebalance_frequency: str = "annual",
        annual_contribution: float = 0.0,
        annual_withdrawal: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            expected_return: Annualized expected return (net of fees).
            volatility: Annualized portfolio volatility.
            initial_investment: Starting portfolio value.
            horizon_years: Simulation time horizon.
            n_simulations: Number of Monte Carlo paths.
            inflation_rate: Fixed annual inflation rate.
            stochastic_inflation: Use random inflation if True.
            inflation_std: Std dev for stochastic inflation.
            rebalance_frequency: How often to rebalance.
            annual_contribution: Annual additional investment.
            annual_withdrawal: Annual withdrawal (positive = withdrawing).
            seed: Random seed for reproducibility.
        """
        self.expected_return = expected_return
        self.volatility = volatility
        self.initial_investment = initial_investment
        self.horizon_years = horizon_years
        self.n_simulations = n_simulations
        self.inflation_rate = inflation_rate
        self.stochastic_inflation = stochastic_inflation
        self.inflation_std = inflation_std
        self.rebalance_frequency = rebalance_frequency
        self.annual_contribution = annual_contribution
        self.annual_withdrawal = annual_withdrawal
        self.seed = seed

    def run(self) -> MonteCarloResult:
        """
        Run the Monte Carlo simulation.

        Uses geometric Brownian motion for compounding-consistent paths:
            ln(S_{t+dt}/S_t) = (μ - σ²/2)*dt + σ*√dt*Z

        This ensures the expected compound return matches the input.
        """
        rng = np.random.default_rng(self.seed)

        periods_per_year = REBALANCE_MAP.get(self.rebalance_frequency, 1)
        n_periods = self.horizon_years * periods_per_year
        dt = 1.0 / periods_per_year

        # GBM parameters per period
        # drift adjusted for Jensen's inequality
        drift = (self.expected_return - 0.5 * self.volatility ** 2) * dt
        diffusion = self.volatility * np.sqrt(dt)

        # Generate random shocks: (n_sim, n_periods)
        Z = rng.standard_normal((self.n_simulations, n_periods))

        # Log returns per period
        log_returns = drift + diffusion * Z  # (n_sim, n_periods)

        # Build nominal paths with contributions/withdrawals
        nominal_paths = np.zeros((self.n_simulations, n_periods + 1))
        nominal_paths[:, 0] = self.initial_investment

        contrib_per_period = self.annual_contribution / periods_per_year
        withdrawal_per_period = self.annual_withdrawal / periods_per_year
        net_flow = contrib_per_period - withdrawal_per_period

        for t in range(n_periods):
            # Compound returns
            nominal_paths[:, t + 1] = (
                nominal_paths[:, t] * np.exp(log_returns[:, t]) + net_flow
            )
            # Floor at zero (can't go negative in portfolio value)
            nominal_paths[:, t + 1] = np.maximum(nominal_paths[:, t + 1], 0)

        # Inflation paths
        if self.stochastic_inflation:
            inflation_annual = generate_stochastic_inflation(
                self.n_simulations, self.horizon_years,
                self.inflation_rate, self.inflation_std,
                seed=self.seed,
            )
            # Expand to per-period
            inflation_per_period = np.repeat(
                inflation_annual, periods_per_year, axis=1
            )[:, :n_periods]
            # Convert annual to per-period
            inflation_per_period = (1 + inflation_per_period) ** dt - 1
        else:
            inflation_per_period_val = (1 + self.inflation_rate) ** dt - 1
            inflation_per_period = np.full(
                (self.n_simulations, n_periods), inflation_per_period_val
            )

        # Cumulative inflation factor
        cum_inflation = np.ones((self.n_simulations, n_periods + 1))
        for t in range(n_periods):
            cum_inflation[:, t + 1] = cum_inflation[:, t] * (1 + inflation_per_period[:, t])

        # Real paths
        real_paths = nominal_paths / cum_inflation

        # Build result
        result = MonteCarloResult(
            nominal_paths=nominal_paths,
            real_paths=real_paths,
            inflation_paths=cum_inflation,
            nominal_terminal=nominal_paths[:, -1],
            real_terminal=real_paths[:, -1],
            n_simulations=self.n_simulations,
            horizon_years=self.horizon_years,
            initial_investment=self.initial_investment,
            inflation_rate=self.inflation_rate,
            stochastic_inflation=self.stochastic_inflation,
            rebalance_frequency=self.rebalance_frequency,
        )

        return result


class HistoricalBootstrapEngine:
    """
    Historical bootstrap Monte Carlo engine.

    Resamples from actual historical return blocks to build paths.
    Preserves cross-asset correlation structure within each block.
    """

    def __init__(
        self,
        historical_returns: np.ndarray,
        weights: np.ndarray,
        initial_investment: float = 100_000,
        horizon_years: int = 10,
        n_simulations: int = 5000,
        block_size: int = 21,  # ~1 month of daily data
        inflation_rate: float = 0.03,
        seed: Optional[int] = None,
    ):
        self.historical_returns = historical_returns  # (n_days, n_assets)
        self.weights = weights
        self.initial_investment = initial_investment
        self.horizon_years = horizon_years
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.inflation_rate = inflation_rate
        self.seed = seed

    def run(self) -> MonteCarloResult:
        """Run bootstrap simulation."""
        rng = np.random.default_rng(self.seed)

        n_days_needed = self.horizon_years * 252
        n_blocks = int(np.ceil(n_days_needed / self.block_size))
        n_available = len(self.historical_returns)

        if n_available < self.block_size:
            raise ValueError(
                f"Need at least {self.block_size} historical observations, "
                f"got {n_available}."
            )

        # Portfolio returns from historical data
        port_returns = self.historical_returns @ self.weights  # (n_days,)

        # Resample blocks
        nominal_paths = np.zeros((self.n_simulations, n_days_needed + 1))
        nominal_paths[:, 0] = self.initial_investment

        for sim in range(self.n_simulations):
            # Draw random block start indices
            max_start = n_available - self.block_size
            block_starts = rng.integers(0, max_start + 1, size=n_blocks)

            # Build resampled return sequence
            resampled = []
            for start in block_starts:
                resampled.extend(port_returns[start:start + self.block_size])

            resampled = np.array(resampled[:n_days_needed])

            # Compound returns to build path
            cumulative = np.cumprod(1 + resampled)
            nominal_paths[sim, 1:] = self.initial_investment * cumulative

        # Inflation adjustment
        daily_inflation = (1 + self.inflation_rate) ** (1 / 252) - 1
        inflation_factors = np.cumprod(
            np.full(n_days_needed, 1 + daily_inflation)
        )
        inflation_all = np.ones(n_days_needed + 1)
        inflation_all[1:] = inflation_factors

        real_paths = nominal_paths / inflation_all[np.newaxis, :]

        # Sample at year boundaries for consistent output
        year_indices = [0] + [i * 252 for i in range(1, self.horizon_years + 1)]
        year_indices = [min(i, n_days_needed) for i in year_indices]

        nominal_yearly = nominal_paths[:, year_indices]
        real_yearly = real_paths[:, year_indices]
        inflation_yearly = inflation_all[year_indices]

        return MonteCarloResult(
            nominal_paths=nominal_yearly,
            real_paths=real_yearly,
            inflation_paths=np.tile(inflation_yearly, (self.n_simulations, 1)),
            nominal_terminal=nominal_paths[:, -1],
            real_terminal=real_paths[:, -1],
            n_simulations=self.n_simulations,
            horizon_years=self.horizon_years,
            initial_investment=self.initial_investment,
            inflation_rate=self.inflation_rate,
            stochastic_inflation=False,
            rebalance_frequency="daily",
        )
