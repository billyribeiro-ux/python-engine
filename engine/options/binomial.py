"""Cox-Ross-Rubinstein binomial tree for American & European options.

Implements the Module 14 chapter 4 algorithm: discretise time into N steps,
build the terminal payoff layer, walk backwards with the early-exercise
``max(continuation, exercise)`` rule. Suitable for any European or American
call/put on an underlying paying a continuous dividend yield.

Pure NumPy. For N=500 a single price is well under a millisecond.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Kind = Literal["call", "put"]
Style = Literal["european", "american"]


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int = 500,
    kind: Kind = "call",
    style: Style = "american",
) -> float:
    """Price an option using the CRR binomial tree.

    Parameters
    ----------
    S : spot
    K : strike
    T : time to expiry in years (must be > 0)
    r : continuous risk-free rate
    q : continuous dividend yield
    sigma : volatility (annualised)
    n_steps : tree depth; 500 is sub-cent accurate for typical equity options
    kind : "call" or "put"
    style : "european" or "american"

    Returns
    -------
    The option price as a float.
    """
    if T <= 0:
        # At/after expiry: intrinsic.
        return float(max(0.0, (S - K) if kind == "call" else (K - S)))
    if sigma <= 0 or n_steps < 1:
        raise ValueError("sigma must be > 0 and n_steps >= 1")

    N = int(n_steps)
    dt = T / N
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    disc = float(np.exp(-r * dt))
    p = (float(np.exp((r - q) * dt)) - d) / (u - d)
    if not (0.0 < p < 1.0):
        # Choose more steps to bring p back into (0, 1) — guards against pathological
        # (r-q) vs sigma combinations.
        raise ValueError(
            f"risk-neutral probability {p:.4f} out of (0,1); try a larger n_steps "
            f"or check r, q, sigma"
        )

    # Terminal prices: S * u^(N-j) * d^j for j = 0..N (so j=0 is the up branch).
    j = np.arange(N + 1)
    prices = S * (u ** (N - j)) * (d**j)
    values = np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)

    # Backward induction.
    for step in range(N - 1, -1, -1):
        prices = prices[: step + 1] / u  # one level down in the tree
        continuation = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if style == "american":
            exercise = (
                np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)
            )
            values = np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])


def early_exercise_premium(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int = 500,
    kind: Kind = "put",
) -> float:
    """Convenience: American price minus European price (the early-exercise
    premium). Should be > 0 for American puts and for American calls on
    dividend-paying underlyings."""
    am = crr_price(S, K, T, r, q, sigma, n_steps, kind=kind, style="american")
    eu = crr_price(S, K, T, r, q, sigma, n_steps, kind=kind, style="european")
    return am - eu
