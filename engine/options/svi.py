"""Raw SVI smile fitter.

The Gatheral (2004) parameterisation: for log-moneyness ``k = log(K/F)``,
the total implied variance is

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)**2 + sigma**2) )

Five parameters per slice: ``(a, b, rho, m, sigma)``. The fitter does a
bounded least-squares fit; the helpers convert between total-variance and
implied-vol representations.

Adequate for the Module 15 chapter 1 use case: vol-smile dislocation
scanners, surface storage, residual computation. For arbitrage-free
multi-slice surfaces, see SSVI (described in the chapter; not implemented
here to keep the module dependency-light).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.b < 0:
            raise ValueError(f"b must be >= 0, got {self.b}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Evaluate the SVI total-variance function w(k)."""
    k = np.asarray(k, dtype=float)
    return params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )


def implied_vol_from_total_variance(w: np.ndarray, T: float) -> np.ndarray:
    """Convert total variance w to implied vol given time-to-expiry T."""
    if T <= 0:
        raise ValueError("T must be > 0")
    return np.sqrt(np.maximum(np.asarray(w) / T, 0.0))


def fit_svi_slice(
    k: np.ndarray,
    w_obs: np.ndarray,
    weights: np.ndarray | None = None,
    n_starts: int = 4,
    seed: int = 0,
) -> SVIParams:
    """Fit one SVI slice to observed total variances.

    Multi-start L-BFGS-B over a sensible bounded region. ``weights`` is
    optional per-strike weighting (e.g. inverse bid-ask spread).
    """
    k = np.asarray(k, dtype=float)
    w_obs = np.asarray(w_obs, dtype=float)
    if k.shape != w_obs.shape:
        raise ValueError("k and w_obs must have the same shape")
    if k.size < 5:
        raise ValueError("need at least 5 strikes to fit SVI")
    if weights is None:
        weights = np.ones_like(w_obs)
    weights = np.asarray(weights, dtype=float)

    def loss(params_array: np.ndarray) -> float:
        a, b, rho, m, sigma = params_array
        if not (-0.999 < rho < 0.999) or b < 0 or sigma <= 1e-6:
            return 1e12
        w_pred = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return float(np.sum(weights * (w_pred - w_obs) ** 2))

    bounds = [
        (0.0, np.max(w_obs) * 2.0 + 1e-3),  # a
        (0.0, 5.0),  # b
        (-0.99, 0.99),  # rho
        (float(k.min()) - 0.5, float(k.max()) + 0.5),  # m
        (1e-3, 5.0),  # sigma
    ]
    rng = np.random.default_rng(seed)
    best: SVIParams | None = None
    best_loss = np.inf
    starts = [
        np.array([np.mean(w_obs), 0.1, -0.3, 0.0, 0.1]),
    ]
    for _ in range(n_starts - 1):
        starts.append(
            np.array(
                [
                    float(rng.uniform(0, np.max(w_obs))),
                    float(rng.uniform(0.05, 1.0)),
                    float(rng.uniform(-0.7, 0.3)),
                    float(rng.uniform(k.min(), k.max())),
                    float(rng.uniform(0.05, 0.5)),
                ]
            )
        )
    for x0 in starts:
        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        if result.fun < best_loss and np.isfinite(result.fun):
            best_loss = float(result.fun)
            a, b, rho, m, sigma = result.x
            best = SVIParams(
                a=float(a),
                b=float(b),
                rho=float(np.clip(rho, -0.99, 0.99)),
                m=float(m),
                sigma=float(max(sigma, 1e-3)),
            )
    if best is None:
        raise RuntimeError("SVI fit did not converge from any start")
    return best


def smile_residuals(
    k: np.ndarray,
    iv_obs: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
) -> tuple[SVIParams, np.ndarray]:
    """Convenience: fit SVI to ``(k, iv_obs)`` and return ``(params, residuals)``
    where ``residuals = iv_obs - iv_fit``. Useful for the vol-smile
    dislocation scanner (Module 18 chapter 3): large standardised residuals
    flag mispriced strikes.
    """
    w_obs = (np.asarray(iv_obs) ** 2) * T
    params = fit_svi_slice(np.asarray(k), w_obs, weights=weights)
    w_fit = svi_total_variance(np.asarray(k), params)
    iv_fit = implied_vol_from_total_variance(w_fit, T)
    return params, np.asarray(iv_obs) - iv_fit
