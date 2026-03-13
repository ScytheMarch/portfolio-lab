"""Tests for return calculations."""
import numpy as np
import pandas as pd
import pytest

from portfolio_lab.analytics.returns import (
    arithmetic_mean_return,
    geometric_mean_return,
    cagr,
    log_returns,
    simple_returns,
    annualize_return,
)


@pytest.fixture
def daily_returns():
    """Simple daily return series."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    returns = pd.Series(np.random.normal(0.0004, 0.01, 252), index=dates)
    return returns


@pytest.fixture
def price_series():
    """Simple price series."""
    dates = pd.bdate_range("2020-01-01", periods=252)
    prices = pd.Series(100 * np.cumprod(1 + np.random.RandomState(42).normal(0.0004, 0.01, 252)), index=dates)
    return prices


class TestArithmeticMean:
    def test_basic(self, daily_returns):
        result = arithmetic_mean_return(daily_returns, annualize=False)
        assert isinstance(result, float)

    def test_annualized(self, daily_returns):
        result = arithmetic_mean_return(daily_returns, annualize=True, periods_per_year=252)
        # Annual return should be roughly 252x daily
        daily = arithmetic_mean_return(daily_returns, annualize=False)
        assert abs(result - daily * 252) < 1e-10

    def test_dataframe(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({
            "A": np.random.normal(0.001, 0.01, 100),
            "B": np.random.normal(0.0005, 0.02, 100),
        }, index=dates)
        result = arithmetic_mean_return(df, annualize=True, periods_per_year=252)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestGeometricMean:
    def test_basic(self, daily_returns):
        result = geometric_mean_return(daily_returns, annualize=False)
        assert isinstance(result, float)

    def test_less_than_arithmetic(self, daily_returns):
        """Geometric mean should be <= arithmetic mean (Jensen's inequality)."""
        geo = geometric_mean_return(daily_returns, annualize=True, periods_per_year=252)
        arith = arithmetic_mean_return(daily_returns, annualize=True, periods_per_year=252)
        assert geo <= arith + 1e-10

    def test_known_values(self):
        """Test with known returns: +10%, -10% should give negative geo mean."""
        returns = pd.Series([0.10, -0.10])
        geo = geometric_mean_return(returns, annualize=False)
        # (1.1 * 0.9)^(1/2) - 1 = sqrt(0.99) - 1 ≈ -0.005
        expected = np.sqrt(1.1 * 0.9) - 1
        assert abs(geo - expected) < 1e-10


class TestCAGR:
    def test_basic(self, price_series):
        result = cagr(price_series)
        assert isinstance(result, float)

    def test_known_growth(self):
        """100 to 200 over 3 years = ~26% CAGR."""
        dates = pd.date_range("2020-01-01", "2023-01-01", periods=756)
        prices = pd.Series(np.linspace(100, 200, 756), index=dates)
        result = cagr(prices)
        expected = (200 / 100) ** (1 / 3.0) - 1
        assert abs(result - expected) < 0.01

    def test_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=252)
        df = pd.DataFrame({
            "A": np.linspace(100, 120, 252),
            "B": np.linspace(100, 90, 252),
        }, index=dates)
        result = cagr(df)
        assert isinstance(result, pd.Series)
        assert result["A"] > 0
        assert result["B"] < 0


class TestAnnualize:
    def test_compound(self):
        # 0.04% daily -> ~10.6% annual
        daily_ret = 0.0004
        annual = annualize_return(daily_ret, 252, method="compound")
        assert abs(annual - ((1.0004 ** 252) - 1)) < 1e-10

    def test_simple(self):
        daily_ret = 0.0004
        annual = annualize_return(daily_ret, 252, method="simple")
        assert abs(annual - 0.0004 * 252) < 1e-10


class TestLogReturns:
    def test_basic(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"A": np.arange(100, 110, dtype=float)}, index=dates)
        lr = log_returns(prices)
        assert len(lr) == 9
        # Log return should equal ln(P_t / P_{t-1})
        expected = np.log(101.0 / 100.0)
        assert abs(lr.iloc[0, 0] - expected) < 1e-10
