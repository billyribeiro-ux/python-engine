"""Tests for engine.features.labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.labels import frac_diff, frac_diff_weights, triple_barrier_labels


def _walk(n: int = 2000, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, sigma, n))), index=idx, name="close")


def test_frac_diff_weights_basic() -> None:
    w0 = frac_diff_weights(d=0.0, size=5)
    # d=0 -> only the last weight is 1, others are 0
    assert w0[-1] == 1.0 and (w0[:-1] == 0).all()
    w1 = frac_diff_weights(d=1.0, size=3)
    # d=1 -> [..., -1, 1] (the first-difference operator)
    np.testing.assert_allclose(w1[-2:], [-1.0, 1.0])


def test_frac_diff_runs_and_aligns() -> None:
    close = _walk()
    out = frac_diff(close, d=0.4)
    assert isinstance(out, pd.Series)
    assert out.index.equals(close.index)
    # Some of the series should be finite after the warm-up
    finite = int(np.isfinite(out.values).sum())
    assert finite > 0


def test_frac_diff_preserves_more_memory_than_first_diff() -> None:
    close = _walk()
    fd_low = frac_diff(close, d=0.3).dropna()
    fd_full = frac_diff(close, d=1.0).dropna()
    # Pearson correlation with the raw series: fractional should be much higher
    corr_low = float(np.corrcoef(close.loc[fd_low.index], fd_low)[0, 1])
    corr_full = float(np.corrcoef(close.loc[fd_full.index], fd_full)[0, 1])
    assert abs(corr_low) > abs(corr_full)


def test_triple_barrier_profit_first() -> None:
    # Synthetic: monotonically rising close → profit barrier hit immediately
    idx = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")
    close = pd.Series(100.0 * (1.0 + 0.01 * np.arange(10)), index=idx, name="close")
    events = pd.Index([idx[0]])
    out = triple_barrier_labels(close, events=events, pt_sl=(0.02, 0.05), horizon_bars=5)
    assert len(out) == 1
    assert int(out["label"].iloc[0]) == 1
    assert float(out["ret"].iloc[0]) >= 0.02 - 1e-9


def test_triple_barrier_stop_first() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")
    # Monotonically falling close → stop barrier hit
    close = pd.Series(100.0 * (1.0 - 0.01 * np.arange(10)), index=idx, name="close")
    events = pd.Index([idx[0]])
    out = triple_barrier_labels(close, events=events, pt_sl=(0.05, 0.02), horizon_bars=5)
    assert int(out["label"].iloc[0]) == -1


def test_triple_barrier_time_barrier() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")
    # Flat → neither barrier hit
    close = pd.Series(100.0, index=idx, name="close")
    events = pd.Index([idx[0]])
    out = triple_barrier_labels(close, events=events, pt_sl=(0.05, 0.05), horizon_bars=3)
    assert int(out["label"].iloc[0]) == 0
