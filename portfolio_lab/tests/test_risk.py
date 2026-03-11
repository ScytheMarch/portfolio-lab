"""Tests for risk calculations."""
import numpy as np
import pandas as pd
import pytest

from portfolio_lab.analytics.risk import (
    covariance_matrix,
    correlation_matrix,
    portfolio_variance,
    portfolio_volatility,
    max_drawdown,
    downside_deviation,
    component_contribution_to_risk,
    risk_contribution_pct,
    diversification_ratio,
    diversification_benefit,
)


@pytest.fixture
def returns_df():
    """Two-asset return DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    return pd.DataFrame({
        "A": np.random.normal(0.0004, 0.01, 252),
        "B": np.random.normal(0.0002, 0.015, 252),
    }, index=dates)


class TestCovarianceMatrix:
    def test_shape(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        assert cov.shape == (2, 2)

    def test_symmetric(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_positive_diagonal(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        assert all(np.diag(cov.values) > 0)


class TestPortfolioVariance:
    def test_equal_weight(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([0.5, 0.5])
        var = portfolio_variance(w, cov.values)
        assert var > 0

    def test_single_asset(self, returns_df):
        """w = [1, 0] should give variance of asset A."""
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([1.0, 0.0])
        var = portfolio_variance(w, cov.values)
        assert abs(var - cov.values[0, 0]) < 1e-10

    def test_portfolio_vol_less_than_weighted_avg(self, returns_df):
        """Diversification benefit: portfolio vol < weighted average vol."""
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([0.5, 0.5])
        port_vol = portfolio_volatility(w, cov.values)
        weighted_avg = 0.5 * np.sqrt(cov.values[0, 0]) + 0.5 * np.sqrt(cov.values[1, 1])
        # Should hold unless perfectly correlated
        assert port_vol <= weighted_avg + 1e-10


class TestMaxDrawdown:
    def test_known(self):
        """Peak 120, trough 50 -> drawdown = (50-120)/120 = -58.33%."""
        prices = pd.Series([100, 120, 80, 50, 90])
        dd = max_drawdown(prices)
        expected = (50 - 120) / 120  # -0.5833
        assert abs(dd - expected) < 0.01

    def test_no_drawdown(self):
        """Monotonically increasing prices."""
        prices = pd.Series([100, 110, 120, 130])
        dd = max_drawdown(prices)
        assert dd == 0.0


class TestRiskContributions:
    def test_sum_to_portfolio_vol(self, returns_df):
        """Component contributions should sum to portfolio volatility."""
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([0.6, 0.4])
        ccr = component_contribution_to_risk(w, cov.values)
        port_vol = portfolio_volatility(w, cov.values)
        assert abs(ccr.sum() - port_vol) < 1e-8

    def test_pct_sums_to_one(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([0.6, 0.4])
        rpct = risk_contribution_pct(w, cov.values)
        assert abs(rpct.sum() - 1.0) < 1e-8


class TestDiversification:
    def test_single_asset(self):
        """Single asset should have DR = 1."""
        cov = np.array([[0.04]])
        w = np.array([1.0])
        dr = diversification_ratio(w, cov)
        assert abs(dr - 1.0) < 1e-10

    def test_uncorrelated(self):
        """Two uncorrelated equal-vol assets, equal weight -> DR = sqrt(2)."""
        vol = 0.2
        cov = np.array([[vol**2, 0], [0, vol**2]])
        w = np.array([0.5, 0.5])
        dr = diversification_ratio(w, cov)
        expected = (0.5 * vol + 0.5 * vol) / np.sqrt(w @ cov @ w)
        assert abs(dr - expected) < 1e-10
        assert dr > 1.0

    def test_benefit_positive(self, returns_df):
        cov = covariance_matrix(returns_df, annualize=True, periods_per_year=252)
        w = np.array([0.5, 0.5])
        benefit = diversification_benefit(w, cov.values)
        assert benefit >= 0
