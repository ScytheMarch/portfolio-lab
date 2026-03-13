"""Tests for deterministic reporting engine."""
import numpy as np
import pytest

from portfolio_lab.analytics.reporting import (
    build_portfolio_summary,
    build_simulation_summary,
    build_return_driver_breakdown,
    build_risk_driver_breakdown,
    build_diversification_summary,
    build_inflation_summary,
    build_goal_attainment_summary,
    generate_post_simulation_report,
)
from portfolio_lab.analytics.diagnostics import (
    generate_stress_flags,
    generate_key_takeaways,
)


@pytest.fixture
def sample_terminal_values():
    """Simulated terminal values."""
    np.random.seed(42)
    return np.random.lognormal(mean=np.log(150_000), sigma=0.3, size=5000)


@pytest.fixture
def sample_real_terminal(sample_terminal_values):
    """Inflation-adjusted terminal values."""
    return sample_terminal_values / 1.3  # ~30% cumulative inflation


class TestSimulationSummary:
    def test_basic_structure(self, sample_terminal_values, sample_real_terminal):
        result = build_simulation_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            inflation_rate=0.03,
            target_value=200_000,
            horizon_years=10,
        )
        assert result["section"] == "Monte Carlo Outcome Summary"
        assert result["n_simulations"] == 5000
        assert result["nominal"]["median"] > 0
        assert result["real"]["median"] > 0
        assert result["real"]["median"] < result["nominal"]["median"]

    def test_goal_hit_rate(self, sample_terminal_values, sample_real_terminal):
        result = build_simulation_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            inflation_rate=0.03,
            target_value=150_000,
            horizon_years=10,
        )
        assert 0 <= result["goal_hit_rate"] <= 1
        assert abs(result["goal_hit_rate"] + result["goal_miss_rate"] - 1.0) < 1e-10

    def test_no_target(self, sample_terminal_values, sample_real_terminal):
        result = build_simulation_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            inflation_rate=0.03,
        )
        assert result["goal_hit_rate"] is None


class TestGoalAttainment:
    def test_target_value(self, sample_terminal_values, sample_real_terminal):
        result = build_goal_attainment_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            goals={"target_value": 200_000},
            horizon_years=10,
        )
        assert "target_value_hit_rate" in result
        assert 0 <= result["target_value_hit_rate"] <= 1

    def test_target_return(self, sample_terminal_values, sample_real_terminal):
        result = build_goal_attainment_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            goals={"target_return": 0.07},
            horizon_years=10,
        )
        assert "target_return_hit_rate" in result

    def test_multiple_goals(self, sample_terminal_values, sample_real_terminal):
        result = build_goal_attainment_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            goals={
                "target_value": 200_000,
                "target_return": 0.07,
                "stretch_target": 300_000,
            },
            horizon_years=10,
        )
        assert "target_value_hit_rate" in result
        assert "target_return_hit_rate" in result
        assert "stretch_target_hit_rate" in result

    def test_principal_loss_prob(self, sample_terminal_values, sample_real_terminal):
        result = build_goal_attainment_summary(
            sample_terminal_values, sample_real_terminal,
            initial_investment=100_000,
            goals={},
            horizon_years=10,
        )
        assert "prob_below_principal_nominal" in result
        assert 0 <= result["prob_below_principal_nominal"] <= 1


class TestStressFlags:
    def test_low_goal_hit_rate_flag(self):
        report = {
            "simulation_summary": {"goal_hit_rate": 0.30, "return_goal_hit_rate": None},
            "inflation_impact": {"nominal_cagr": 0.08, "real_cagr": 0.04, "purchasing_power_reduction_pct": 0.1},
            "risk_drivers": {"concentration_warnings": []},
            "diversification": {"quality_band": "Good", "diversification_ratio": 1.6},
            "portfolio_summary": {"weighted_expense_ratio": 0.001},
            "return_drivers": {"total_fee_drag": 0.001, "gross_return": 0.08},
        }
        result = generate_stress_flags(report)
        flags = result["flags"]
        assert any(f["category"] == "goal_attainment" for f in flags)

    def test_no_flags_for_good_portfolio(self):
        report = {
            "simulation_summary": {"goal_hit_rate": 0.85, "return_goal_hit_rate": 0.85,
                                    "prob_nominal_loss": 0.05, "prob_real_loss": 0.10},
            "inflation_impact": {"nominal_cagr": 0.08, "real_cagr": 0.06, "purchasing_power_reduction_pct": 0.15},
            "risk_drivers": {"concentration_warnings": []},
            "diversification": {"quality_band": "Good", "diversification_ratio": 1.6},
            "portfolio_summary": {"weighted_expense_ratio": 0.003},
            "return_drivers": {"total_fee_drag": 0.003, "gross_return": 0.08},
        }
        result = generate_stress_flags(report)
        assert result["high_severity_count"] == 0


class TestKeyTakeaways:
    def test_generates_takeaways(self):
        report = {
            "diversification": {"diversification_ratio": 1.7, "quality_band": "good"},
            "risk_drivers": {"top_3_risk_contributors": [("VTI", 0.40)]},
            "simulation_summary": {"goal_hit_rate": 0.75, "initial_investment": 100000},
            "inflation_impact": {"inflation_drag_dollars": 25000},
            "return_drivers": {"total_fee_drag": 0.008},
            "portfolio_summary": {"return_estimation_method": "Historical Arithmetic", "expected_net_return": 0.07},
        }
        result = generate_key_takeaways(report)
        assert len(result["takeaways"]) > 0


class TestDiversificationSummary:
    def test_poor(self):
        result = build_diversification_summary(
            div_ratio=1.1, div_benefit=0.09,
            tickers=["A", "B"], weights={"A": 0.8, "B": 0.2},
            risk_pct={"A": 0.9, "B": 0.1},
            correlation_matrix={},
        )
        assert result["quality_band"] == "Poor"

    def test_excellent(self):
        result = build_diversification_summary(
            div_ratio=2.5, div_benefit=0.60,
            tickers=["A", "B", "C"], weights={"A": 0.33, "B": 0.33, "C": 0.34},
            risk_pct={"A": 0.33, "B": 0.33, "C": 0.34},
            correlation_matrix={},
        )
        assert result["quality_band"] == "Excellent"


class TestFullReport:
    def test_all_sections_present(self, sample_terminal_values, sample_real_terminal):
        report = generate_post_simulation_report(
            tickers=["VTI", "BND"],
            weights={"VTI": 0.6, "BND": 0.4},
            objective="max_sharpe",
            return_method="Historical Arithmetic Mean",
            expected_gross_return=0.08,
            expected_net_return=0.075,
            expected_volatility=0.12,
            sharpe=0.46,
            sortino=0.65,
            weighted_er=0.005,
            advisory_fee=0.0,
            income_yield=0.02,
            diversification_score=1.5,
            diversification_benefit_val=0.33,
            factor_exposures={"Mkt-RF": 0.6, "SMB": -0.1},
            risk_free_rate=0.05,
            risk_free_source="^IRX",
            expected_returns_per_asset={"VTI": 0.10, "BND": 0.04},
            income_contributions={"VTI": 0.01, "BND": 0.01},
            component_risk={"VTI": 0.08, "BND": 0.02},
            marginal_risk={"VTI": 0.12, "BND": 0.06},
            risk_pct={"VTI": 0.75, "BND": 0.25},
            correlation_observations=["VTI and BND are negatively correlated (-0.20)"],
            correlation_matrix_flat={"VTI_BND": -0.20},
            terminal_values=sample_terminal_values,
            real_terminal_values=sample_real_terminal,
            initial_investment=100_000,
            inflation_rate=0.03,
            horizon_years=10,
            goals={"target_value": 200_000},
        )

        assert "portfolio_summary" in report
        assert "simulation_summary" in report
        assert "return_drivers" in report
        assert "risk_drivers" in report
        assert "diversification" in report
        assert "inflation_impact" in report
        assert "goal_attainment" in report
        assert "stress_flags" in report
        assert "key_takeaways" in report
