"""
Inflation analytics: real return conversion, purchasing power analysis.
"""
import numpy as np


def nominal_to_real_return(
    nominal_return: float,
    inflation_rate: float,
) -> float:
    """
    Convert nominal return to real return using Fisher equation.

    real = (1 + nominal) / (1 + inflation) - 1

    This is the exact Fisher equation, not the linear approximation.
    """
    return (1 + nominal_return) / (1 + inflation_rate) - 1


def real_to_nominal_return(
    real_return: float,
    inflation_rate: float,
) -> float:
    """Convert real return to nominal return."""
    return (1 + real_return) * (1 + inflation_rate) - 1


def purchasing_power_factor(
    inflation_rate: float,
    years: int,
) -> float:
    """
    What $1 today is worth in real terms after `years` of inflation.

    Returns a value < 1 (e.g., 0.74 means $1 buys $0.74 worth in today's terms).
    """
    return 1.0 / (1 + inflation_rate) ** years


def inflation_drag(
    nominal_cagr: float,
    inflation_rate: float,
    years: int,
    initial_value: float = 100_000,
) -> dict:
    """
    Quantify inflation's impact on a portfolio over time.

    Returns:
        Dict with nominal terminal value, real terminal value,
        nominal CAGR, real CAGR, and total purchasing power loss.
    """
    nominal_terminal = initial_value * (1 + nominal_cagr) ** years
    real_cagr = nominal_to_real_return(nominal_cagr, inflation_rate)
    real_terminal = initial_value * (1 + real_cagr) ** years
    pp_loss = nominal_terminal - real_terminal

    return {
        "nominal_terminal_value": nominal_terminal,
        "real_terminal_value": real_terminal,
        "nominal_cagr": nominal_cagr,
        "real_cagr": real_cagr,
        "purchasing_power_loss": pp_loss,
        "purchasing_power_loss_pct": pp_loss / nominal_terminal if nominal_terminal > 0 else 0,
        "inflation_rate": inflation_rate,
        "years": years,
    }


def generate_stochastic_inflation(
    n_simulations: int,
    n_periods: int,
    mean_inflation: float = 0.03,
    std_inflation: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate stochastic inflation paths.

    Uses a normal distribution truncated to prevent negative inflation below -2%.

    Args:
        n_simulations: Number of simulation paths.
        n_periods: Number of periods per path.
        mean_inflation: Mean annual inflation rate.
        std_inflation: Standard deviation of annual inflation.
        seed: Random seed for reproducibility.

    Returns:
        ndarray of shape (n_simulations, n_periods) with annual inflation rates.
    """
    rng = np.random.default_rng(seed)
    inflation = rng.normal(mean_inflation, std_inflation, (n_simulations, n_periods))
    # Floor at -2% (deflation limit)
    inflation = np.clip(inflation, -0.02, None)
    return inflation
