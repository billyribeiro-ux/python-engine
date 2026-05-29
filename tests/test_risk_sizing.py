"""Tests for engine.risk.sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk import (
    cvar_sized_weight,
    fractional_kelly,
    historical_cvar,
    historical_var,
    kelly_fraction,
    vol_target_weight,
)


def test_kelly_zero_when_no_edge() -> None:
    assert kelly_fraction(0.0, 0.04) == 0.0


def test_kelly_zero_variance_returns_zero() -> None:
    assert kelly_fraction(0.01, 0.0) == 0.0


def test_kelly_scales_with_edge() -> None:
    f_small = kelly_fraction(0.001, 0.01)
    f_big = kelly_fraction(0.01, 0.01)
    assert f_big > f_small


def test_fractional_kelly_halves_kelly() -> None:
    full = kelly_fraction(0.005, 0.02)
    half = fractional_kelly(0.005, 0.02, fraction=0.5)
    assert abs(half - 0.5 * full) < 1e-12


def test_fractional_kelly_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        fractional_kelly(0.01, 0.01, fraction=1.5)


def test_vol_target_weight_matches_target() -> None:
    # A 30% vol asset, targeting 15% → 0.5 weight.
    w = vol_target_weight(realised_vol=0.30, target_vol=0.15)
    assert abs(w - 0.5) < 1e-12


def test_vol_target_weight_capped() -> None:
    # A 5% vol asset targeting 15% wants 3x leverage; we cap at 1.0.
    w = vol_target_weight(realised_vol=0.05, target_vol=0.15, max_weight=1.0)
    assert w == 1.0


def test_vol_target_zero_vol_returns_zero() -> None:
    assert vol_target_weight(realised_vol=0.0, target_vol=0.15) == 0.0


def test_var_and_cvar_positive_for_loss_distribution() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, size=1000))
    var = historical_var(r, level=0.05)
    cvar = historical_cvar(r, level=0.05)
    assert var > 0
    assert cvar >= var  # CVaR is at least VaR


def test_cvar_sized_weight_scales_down_for_risky_returns() -> None:
    rng = np.random.default_rng(0)
    safe = pd.Series(rng.normal(0, 0.005, size=500))
    risky = pd.Series(rng.normal(0, 0.05, size=500))
    w_safe = cvar_sized_weight(safe, max_cvar=0.01)
    w_risky = cvar_sized_weight(risky, max_cvar=0.01)
    assert w_safe > w_risky


def test_historical_var_empty_returns_zero() -> None:
    assert historical_var(pd.Series([], dtype=float)) == 0.0
    assert historical_cvar(pd.Series([], dtype=float)) == 0.0
