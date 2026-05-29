"""Tests for engine.options.binomial."""

from __future__ import annotations

import pytest

from engine.options import crr_price, early_exercise_premium, price


def test_european_crr_converges_to_black_scholes() -> None:
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    bs = float(price(S, K, T, r, q, sigma, kind="call"))
    crr = crr_price(S, K, T, r, q, sigma, n_steps=1000, kind="call", style="european")
    assert abs(crr - bs) < 1e-2


def test_european_put_crr_matches_bs() -> None:
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.03, 0.01, 0.25
    bs = float(price(S, K, T, r, q, sigma, kind="put"))
    crr = crr_price(S, K, T, r, q, sigma, n_steps=1000, kind="put", style="european")
    assert abs(crr - bs) < 1e-2


def test_american_put_premium_positive() -> None:
    # American puts should be worth at least as much as European puts.
    prem = early_exercise_premium(100.0, 110.0, 1.0, 0.05, 0.0, 0.3, kind="put")
    assert prem > 0


def test_american_call_no_dividend_equals_european() -> None:
    # Classical result: American call on a non-dividend-paying stock = European.
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.05, 0.0, 0.25
    am = crr_price(S, K, T, r, q, sigma, n_steps=500, kind="call", style="american")
    eu = crr_price(S, K, T, r, q, sigma, n_steps=500, kind="call", style="european")
    assert abs(am - eu) < 1e-6


def test_at_expiry_returns_intrinsic() -> None:
    assert crr_price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, kind="call") == 10.0
    assert crr_price(90.0, 100.0, 0.0, 0.05, 0.0, 0.2, kind="put") == 10.0
    assert crr_price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, kind="put") == 0.0


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 1.0, 0.05, 0.0, -0.2)  # negative sigma
