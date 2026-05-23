"""Tests for engine.options."""

from __future__ import annotations

import numpy as np

from engine.options import greeks, implied_vol, price


def test_atm_call_price_known() -> None:
    # Known case: S=K=100, T=1, r=0.05, q=0, sigma=0.2 → call ~ 10.4506
    p = float(price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, kind="call"))
    assert abs(p - 10.4506) < 1e-3


def test_put_call_parity() -> None:
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.04, 0.01, 0.25
    c = float(price(S, K, T, r, q, sigma, kind="call"))
    p = float(price(S, K, T, r, q, sigma, kind="put"))
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8


def test_implied_vol_roundtrip() -> None:
    S, K, T, r, q = 100.0, 105.0, 0.25, 0.03, 0.0
    target_sigma = 0.30
    target_price = float(price(S, K, T, r, q, target_sigma, kind="call"))
    iv = implied_vol(target_price, S, K, T, r, q, kind="call")
    assert abs(iv - target_sigma) < 1e-5


def test_implied_vol_outside_bounds_returns_nan() -> None:
    # Asking for a price above the upper bound should return NaN
    iv = implied_vol(1000.0, 100.0, 100.0, 1.0, 0.05, 0.0, kind="call")
    assert np.isnan(iv)


def test_greeks_signs() -> None:
    g = greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, kind="call")
    assert 0 < g.delta < 1
    assert g.gamma > 0
    assert g.vega > 0
    assert g.theta < 0  # calls decay


def test_put_greeks_delta_negative() -> None:
    g = greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, kind="put")
    assert -1 < g.delta < 0
