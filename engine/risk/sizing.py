"""Position sizing: Kelly, fractional Kelly, vol targeting, CVaR.

The four sizers a real book actually uses, exposed as simple functions
returning a *fraction of capital*. Combine them — vol-target the Kelly
weight, cap at CVaR-tolerable size, etc. — in the strategy layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(mean: float, variance: float) -> float:
    """Continuous Kelly for a Gaussian edge.

    For a per-period return distribution with mean ``mu`` and variance
    ``sigma^2``, the optimal fraction-of-bankroll is ``mu / sigma^2``.
    """
    if variance <= 0:
        return 0.0
    return float(mean / variance)


def fractional_kelly(mean: float, variance: float, fraction: float = 0.5) -> float:
    """Half-Kelly (or any fraction) for drawdown control.

    Full Kelly is the *growth-maximising* bet and is brutal on drawdowns.
    Production books almost universally bet a fraction (commonly 0.25 to
    0.5) of full Kelly. This function multiplies through.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return fraction * kelly_fraction(mean, variance)


def vol_target_weight(
    realised_vol: float,
    target_vol: float,
    max_weight: float = 1.0,
) -> float:
    """Scale to a target *annualised* volatility.

    Pass ``realised_vol`` and ``target_vol`` in the same units (annualised
    fraction — e.g. 0.15 for 15%). Result is the fraction of capital to
    allocate so the portfolio's expected volatility equals the target.
    """
    if realised_vol <= 0:
        return 0.0
    w = target_vol / realised_vol
    return float(min(w, max_weight))


def historical_var(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Value-at-Risk at the given level (e.g. 0.05 = 5% VaR).

    Returns a *positive* number representing the magnitude of loss.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    q = float(np.quantile(r, level))
    return float(-q)


def historical_cvar(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Conditional VaR (expected shortfall) at the given level.

    The average loss in the worst ``level`` fraction of observations,
    returned as a positive magnitude.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    q = float(np.quantile(r, level))
    tail = r[r <= q]
    if len(tail) == 0:
        return float(-q)
    return float(-tail.mean())


def cvar_sized_weight(
    returns: pd.Series,
    max_cvar: float,
    level: float = 0.05,
    max_weight: float = 1.0,
) -> float:
    """Cap position size so portfolio CVaR doesn't exceed ``max_cvar``.

    ``max_cvar`` is the absolute fractional loss you'll tolerate at the
    given confidence level (e.g. 0.02 = "5% CVaR no worse than 2%
    drawdown per bar"). Returns the weight that achieves this; ``max_weight``
    caps it from above.
    """
    cvar = historical_cvar(returns, level)
    if cvar <= 0:
        return max_weight
    w = max_cvar / cvar
    return float(min(w, max_weight))
