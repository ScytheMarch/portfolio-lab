"""
Withdrawal and contribution modeling for Monte Carlo simulations.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CashFlowSchedule:
    """Defines contributions and withdrawals over the simulation horizon."""
    annual_contribution: float = 0.0
    annual_withdrawal: float = 0.0
    contribution_growth_rate: float = 0.0  # annual increase in contributions
    withdrawal_growth_rate: float = 0.0    # annual increase in withdrawals (e.g., for inflation)
    start_withdrawal_year: int = 0         # year to begin withdrawals
    stop_contribution_year: Optional[int] = None  # year to stop contributing

    def get_net_flow(self, year: int) -> float:
        """
        Net cash flow for a given year.

        Positive = net inflow, Negative = net outflow.
        """
        contrib = 0.0
        if self.stop_contribution_year is None or year <= self.stop_contribution_year:
            contrib = self.annual_contribution * (1 + self.contribution_growth_rate) ** year

        withdrawal = 0.0
        if year >= self.start_withdrawal_year:
            years_withdrawing = year - self.start_withdrawal_year
            withdrawal = self.annual_withdrawal * (
                1 + self.withdrawal_growth_rate
            ) ** years_withdrawing

        return contrib - withdrawal

    def get_flows_for_horizon(self, horizon_years: int) -> list[float]:
        """Return list of net flows for each year of the horizon."""
        return [self.get_net_flow(y) for y in range(1, horizon_years + 1)]
