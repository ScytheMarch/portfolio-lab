"""Random portfolio basket generation with locking and exclusion support."""

from __future__ import annotations

import random

DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B",
    "JPM", "V", "MA", "UNH", "HD", "PG", "XOM", "CVX",
    "KO", "PEP", "ABBV", "MRK", "COST", "AVGO", "LLY", "WMT",
    "BAC", "ADBE", "CRM", "NFLX", "ORCL", "TMO", "MCD", "LIN",
    "ACN", "DHR", "ABT", "AMD", "NKE", "TXN", "PM", "QCOM",
    "BMY", "HON", "UPS", "LOW", "IBM", "CAT", "GS", "SBUX",
    "AMGN", "SPGI", "RTX", "INTU", "DE", "BLK", "MDT", "ADP",
    "GILD", "BKNG", "ISRG", "SYK",
]


def generate_random_portfolio(
    universe: list[str],
    min_picks: int = 30,
    max_picks: int = 60,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Randomly select tickers from the universe.

    Parameters
    ----------
    universe : list[str]
        Master list of eligible tickers.
    min_picks, max_picks : int
        Random count will be chosen in [min_picks, max_picks].
    locked : list[str] | None
        Tickers that must always be included.
    excluded : list[str] | None
        Tickers temporarily removed from the pool.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    list[str]
        Selected tickers (locked + randomly sampled), sorted alphabetically.
    """
    locked = list(locked or [])
    excluded = set(excluded or [])

    # Build eligible pool: universe minus excluded minus already-locked
    locked_set = set(locked)
    eligible = [t for t in universe if t not in excluded and t not in locked_set]

    # Validate constraints
    if min_picks > max_picks:
        raise ValueError(f"min_picks ({min_picks}) cannot exceed max_picks ({max_picks})")

    total_available = len(locked) + len(eligible)
    if min_picks > total_available:
        raise ValueError(
            f"min_picks ({min_picks}) exceeds available tickers "
            f"({len(locked)} locked + {len(eligible)} eligible = {total_available})"
        )

    # Cap max_picks to what's actually available
    effective_max = min(max_picks, total_available)
    effective_min = min(min_picks, effective_max)

    # Choose random count
    rng = random.Random(seed)
    target_count = rng.randint(effective_min, effective_max)

    # How many additional tickers to sample beyond locked ones
    additional_needed = max(0, target_count - len(locked))
    additional_needed = min(additional_needed, len(eligible))

    sampled = rng.sample(eligible, additional_needed)

    result = sorted(set(locked + sampled))
    return result
