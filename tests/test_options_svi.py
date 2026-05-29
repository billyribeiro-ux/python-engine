"""Tests for engine.options.svi."""

from __future__ import annotations

import numpy as np
import pytest

from engine.options import (
    SVIParams,
    fit_svi_slice,
    implied_vol_from_total_variance,
    smile_residuals,
    svi_total_variance,
)


def test_svi_params_validation() -> None:
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=0.2, rho=1.5, m=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=0.2, rho=0.0, m=0.0, sigma=0.0)


def test_svi_total_variance_smile_shape() -> None:
    params = SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.1)
    k = np.linspace(-0.3, 0.3, 21)
    w = svi_total_variance(k, params)
    # ATM is the min when m=0 and rho=0; with rho=-0.3 the min shifts to k > 0.
    assert np.all(w > 0)
    # No NaN/Inf
    assert np.isfinite(w).all()


def test_fit_svi_recovers_known_smile() -> None:
    truth = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.05, sigma=0.1)
    k = np.linspace(-0.4, 0.4, 31)
    w_true = svi_total_variance(k, truth)
    rng = np.random.default_rng(0)
    w_noisy = w_true + rng.normal(0, 1e-4, size=k.size)
    fit = fit_svi_slice(k, w_noisy, n_starts=6, seed=0)
    w_fit = svi_total_variance(k, fit)
    # Recovered curve should match truth tightly even with noise.
    rms = float(np.sqrt(np.mean((w_fit - w_true) ** 2)))
    assert rms < 5e-3


def test_implied_vol_from_total_variance_roundtrip() -> None:
    w = np.array([0.04, 0.05, 0.06])
    T = 0.5
    iv = implied_vol_from_total_variance(w, T)
    np.testing.assert_allclose(iv**2 * T, w)


def test_implied_vol_invalid_T_raises() -> None:
    with pytest.raises(ValueError):
        implied_vol_from_total_variance(np.array([0.04]), 0.0)


def test_smile_residuals_small_for_consistent_smile() -> None:
    # Build IVs from a known SVI, then fit and check residuals are tiny.
    truth = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.0, sigma=0.1)
    T = 0.25
    k = np.linspace(-0.3, 0.3, 21)
    w = svi_total_variance(k, truth)
    iv = np.sqrt(w / T)
    _, resid = smile_residuals(k, iv, T)
    assert np.max(np.abs(resid)) < 1e-3


def test_fit_svi_rejects_too_few_strikes() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi_slice(np.array([0.0, 0.1]), np.array([0.04, 0.05]))
