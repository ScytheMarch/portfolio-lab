"""Tests for Monte Carlo simulation engine."""
import numpy as np
import pytest

from portfolio_lab.simulation.monte_carlo import MonteCarloEngine, MonteCarloResult
from portfolio_lab.analytics.inflation import nominal_to_real_return


class TestMonteCarloEngine:
    def test_basic_run(self):
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=10,
            n_simulations=1000,
            inflation_rate=0.03,
            seed=42,
        )
        result = mc.run()

        assert isinstance(result, MonteCarloResult)
        assert result.nominal_paths.shape[0] == 1000
        assert result.nominal_paths.shape[1] == 11  # 10 years + initial
        assert result.nominal_terminal.shape == (1000,)

    def test_initial_value(self):
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=50_000,
            horizon_years=5,
            n_simulations=100,
            seed=42,
        )
        result = mc.run()

        # All paths should start at initial investment
        np.testing.assert_array_almost_equal(
            result.nominal_paths[:, 0], 50_000
        )

    def test_mean_terminal_reasonable(self):
        """Mean terminal value should be near expected growth."""
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=10,
            n_simulations=10_000,
            seed=42,
        )
        result = mc.run()
        result.compute_stats()

        # Expected: 100k * exp(0.08 * 10) ≈ 222k (using GBM drift)
        # With large n, mean should be close
        mean_terminal = result.stats["nominal_mean"]
        expected = 100_000 * np.exp(0.08 * 10)  # E[S_T] for GBM
        # Allow 15% tolerance due to finite samples
        assert abs(mean_terminal - expected) / expected < 0.15

    def test_real_less_than_nominal(self):
        """With positive inflation, real terminal should be < nominal."""
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=10,
            n_simulations=1000,
            inflation_rate=0.03,
            seed=42,
        )
        result = mc.run()
        result.compute_stats()

        assert result.stats["real_median"] < result.stats["nominal_median"]

    def test_zero_inflation(self):
        """With zero inflation, real == nominal."""
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=5,
            n_simulations=100,
            inflation_rate=0.0,
            seed=42,
        )
        result = mc.run()
        np.testing.assert_array_almost_equal(
            result.nominal_terminal, result.real_terminal, decimal=2
        )

    def test_stochastic_inflation(self):
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=10,
            n_simulations=100,
            inflation_rate=0.03,
            stochastic_inflation=True,
            inflation_std=0.01,
            seed=42,
        )
        result = mc.run()
        assert result.real_paths.shape == result.nominal_paths.shape

    def test_goal_hit_rate(self):
        mc = MonteCarloEngine(
            expected_return=0.08,
            volatility=0.15,
            initial_investment=100_000,
            horizon_years=10,
            n_simulations=5000,
            seed=42,
        )
        result = mc.run()
        result.compute_stats(target_value=150_000)

        # Should have a goal hit rate between 0 and 1
        assert 0 <= result.stats["goal_hit_rate"] <= 1
        assert abs(result.stats["goal_hit_rate"] + result.stats["goal_miss_rate"] - 1.0) < 1e-10

    def test_contributions(self):
        """With contributions, terminal value should be higher."""
        mc_no_contrib = MonteCarloEngine(
            expected_return=0.08, volatility=0.15,
            initial_investment=100_000, horizon_years=10,
            n_simulations=1000, seed=42,
        )
        mc_with_contrib = MonteCarloEngine(
            expected_return=0.08, volatility=0.15,
            initial_investment=100_000, horizon_years=10,
            n_simulations=1000, annual_contribution=10_000, seed=42,
        )
        res1 = mc_no_contrib.run()
        res2 = mc_with_contrib.run()

        assert np.mean(res2.nominal_terminal) > np.mean(res1.nominal_terminal)

    def test_drawdown_stats(self):
        mc = MonteCarloEngine(
            expected_return=0.08, volatility=0.15,
            initial_investment=100_000, horizon_years=10,
            n_simulations=100, seed=42,
        )
        result = mc.run()
        result.compute_stats()

        assert "max_drawdown_mean" in result.stats
        assert result.stats["max_drawdown_mean"] < 0  # drawdowns are negative


class TestInflationConversion:
    def test_fisher_equation(self):
        real = nominal_to_real_return(0.08, 0.03)
        expected = (1.08 / 1.03) - 1
        assert abs(real - expected) < 1e-10

    def test_zero_inflation(self):
        real = nominal_to_real_return(0.08, 0.0)
        assert abs(real - 0.08) < 1e-10
