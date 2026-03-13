"""Tests for optimization engine."""
import numpy as np
import pytest

from portfolio_lab.optimization.solver import optimize_portfolio, OptimizationResult


@pytest.fixture
def two_asset_data():
    """Two-asset optimization setup."""
    tickers = ["A", "B"]
    expected_returns = np.array([0.10, 0.05])
    # Cov matrix: A vol=20%, B vol=10%, corr=0.3
    cov = np.array([
        [0.04, 0.006],
        [0.006, 0.01],
    ])
    return tickers, expected_returns, cov


class TestMaxSharpe:
    def test_success(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov, risk_free_rate=0.02,
            objective="max_sharpe",
        )
        assert result.success
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_weights_in_bounds(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov, risk_free_rate=0.02,
            objective="max_sharpe",
            min_weight=0.0, max_weight=0.7,
        )
        assert result.success
        for w in result.weights.values():
            assert w >= -1e-6
            assert w <= 0.7 + 1e-6


class TestMinVolatility:
    def test_success(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov, risk_free_rate=0.02,
            objective="min_volatility",
        )
        assert result.success
        # Min vol should favor lower-vol asset B
        assert result.weights["B"] > result.weights["A"]

    def test_lower_vol_than_any_single_asset(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov, objective="min_volatility",
        )
        vol_A = np.sqrt(cov[0, 0])
        vol_B = np.sqrt(cov[1, 1])
        assert result.volatility <= min(vol_A, vol_B) + 1e-6


class TestTargetReturn:
    def test_achieves_target(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        target = 0.07
        result = optimize_portfolio(
            tickers, mu, cov, risk_free_rate=0.02,
            objective="target_return",
            target_return=target,
        )
        assert result.success
        assert abs(result.expected_return - target) < 0.01


class TestInfeasible:
    def test_impossible_target(self, two_asset_data):
        """Target return above max possible should fail."""
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov,
            objective="target_return",
            target_return=0.50,  # 50% - impossible with 10% and 5% assets
            min_weight=0.0,
        )
        # Should either fail or weights should be at boundary
        # With no shorting, can't exceed max(mu)=10%
        if result.success:
            assert result.expected_return <= max(mu) + 0.01


class TestSharpeCalculation:
    def test_sharpe_positive(self, two_asset_data):
        tickers, mu, cov = two_asset_data
        result = optimize_portfolio(
            tickers, mu, cov, risk_free_rate=0.02,
            objective="max_sharpe",
        )
        assert result.sharpe_ratio > 0
