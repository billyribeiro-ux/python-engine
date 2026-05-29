"""Black-Scholes pricing and Greeks.

Implements the Black-Scholes-Merton formulas for European calls and puts on
an underlying paying a continuous dividend yield. All functions are
vectorised — pass scalars or arrays for any argument.

For American options use the binomial / LSMC modules; for stochastic-vol
surfaces use ``engine.options.heston`` / ``engine.options.sabr`` (later in
Module 15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionKind = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    charm: float
    vomma: float


def _d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    kind: OptionKind = "call",
) -> np.ndarray:
    """Black-Scholes price for a European option with continuous dividend yield."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r = np.exp(-r * np.asarray(T))
    df_q = np.exp(-q * np.asarray(T))
    S_arr = np.asarray(S)
    K_arr = np.asarray(K)
    if kind == "call":
        return S_arr * df_q * norm.cdf(d1) - K_arr * df_r * norm.cdf(d2)
    if kind == "put":
        return K_arr * df_r * norm.cdf(-d2) - S_arr * df_q * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def greeks(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    kind: OptionKind = "call",
) -> Greeks:
    """First- and second-order Greeks, including the cross Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    T_arr = np.asarray(T, dtype=float)
    sqrtT = np.sqrt(np.maximum(T_arr, 1e-12))
    df_r = np.exp(-r * T_arr)
    df_q = np.exp(-q * T_arr)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    pdf_d1 = norm.pdf(d1)
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            -S_arr * df_q * pdf_d1 * sigma / (2 * sqrtT)
            - r * K_arr * df_r * norm.cdf(d2)
            + q * S_arr * df_q * norm.cdf(d1)
        )
        rho = K_arr * T_arr * df_r * norm.cdf(d2)
        charm = -df_q * (
            pdf_d1 * (2 * (r - q) * T_arr - d2 * sigma * sqrtT) / (2 * T_arr * sigma * sqrtT)
            - q * norm.cdf(d1)
        )
    elif kind == "put":
        delta = -df_q * norm.cdf(-d1)
        theta = (
            -S_arr * df_q * pdf_d1 * sigma / (2 * sqrtT)
            + r * K_arr * df_r * norm.cdf(-d2)
            - q * S_arr * df_q * norm.cdf(-d1)
        )
        rho = -K_arr * T_arr * df_r * norm.cdf(-d2)
        charm = -df_q * (
            pdf_d1 * (2 * (r - q) * T_arr - d2 * sigma * sqrtT) / (2 * T_arr * sigma * sqrtT)
            + q * norm.cdf(-d1)
        )
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    gamma = df_q * pdf_d1 / (S_arr * sigma * sqrtT)
    vega = S_arr * df_q * pdf_d1 * sqrtT
    vanna = -df_q * pdf_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma
    return Greeks(
        delta=float(np.squeeze(delta)),
        gamma=float(np.squeeze(gamma)),
        vega=float(np.squeeze(vega)),
        theta=float(np.squeeze(theta)),
        rho=float(np.squeeze(rho)),
        vanna=float(np.squeeze(vanna)),
        charm=float(np.squeeze(charm)),
        vomma=float(np.squeeze(vomma)),
    )


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    kind: OptionKind = "call",
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Implied volatility via Brent's method with a safe bracket.

    Robust to deep-ITM / OTM options where Newton-Raphson on vega fails
    (vega → 0). Returns ``float('nan')`` if the price is outside the
    no-arbitrage bounds.
    """
    from scipy.optimize import brentq

    intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        if kind == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    )
    upper_bound = S * np.exp(-q * T) if kind == "call" else K * np.exp(-r * T)
    if not (intrinsic - tol <= target_price <= upper_bound + tol):
        return float("nan")

    def diff(sigma: float) -> float:
        return float(price(S, K, T, r, q, sigma, kind=kind) - target_price)

    lo, hi = 1e-6, 5.0
    f_lo, f_hi = diff(lo), diff(hi)
    # The function is monotone in sigma; if signs are equal at the bracket, no root
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(diff, lo, hi, xtol=tol, maxiter=max_iter))
