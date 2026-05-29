"""Tests for engine.risk.portfolio (HRP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk import correlation_distance, hrp_weights


def test_hrp_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    cols = ["A", "B", "C", "D", "E"]
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(500, 5)), columns=cols)
    w = hrp_weights(ret)
    assert abs(w.sum() - 1.0) < 1e-9


def test_hrp_weights_long_only() -> None:
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(500, 4)), columns=list("ABCD"))
    w = hrp_weights(ret)
    assert (w >= 0).all()


def test_hrp_concentrates_on_low_vol_asset() -> None:
    # Two assets, one with 4x the vol of the other; HRP (via inverse-variance
    # at every split) should give more weight to the low-vol one.
    rng = np.random.default_rng(0)
    n = 1000
    low = rng.normal(0, 0.005, size=n)
    high = rng.normal(0, 0.02, size=n)
    ret = pd.DataFrame({"low": low, "high": high})
    w = hrp_weights(ret)
    assert w["low"] > w["high"]


def test_correlation_distance_diagonal_zero() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    d = correlation_distance(corr)
    np.testing.assert_allclose(np.diag(d.values), 0.0, atol=1e-12)
    # Distance is symmetric and >= 0
    np.testing.assert_allclose(d.values, d.values.T)
    assert (d.values >= 0).all()


def test_hrp_rejects_single_asset() -> None:
    ret = pd.DataFrame({"A": [0.01, 0.02, -0.005]})
    with pytest.raises(ValueError, match="at least 2"):
        hrp_weights(ret)
