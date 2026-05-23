"""Tests for engine.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest import (
    PurgedKFold,
    WalkForward,
    deflated_sharpe,
    vectorized_backtest,
)


def _returns(n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx, name="ret")


def test_vectorized_backtest_shifts_signal_by_default() -> None:
    r = _returns()
    # Signal = +1 on all bars. With shift, position is +1 from second bar onward.
    pos = pd.Series(1.0, index=r.index)
    res = vectorized_backtest(r, pos)
    assert res.positions.iloc[0] == 0.0
    assert res.positions.iloc[1] == 1.0
    # P&L on first bar is zero (no position yet)
    assert res.pnl.iloc[0] == 0.0
    # On second bar, P&L equals the asset's return (no cost)
    np.testing.assert_allclose(res.pnl.iloc[1], r.iloc[1])


def test_vectorized_backtest_costs_reduce_pnl() -> None:
    r = _returns()
    pos = pd.Series(np.where(r.index.dayofweek < 3, 1.0, -1.0), index=r.index)
    no_cost = vectorized_backtest(r, pos, cost_per_unit_turnover=0.0).pnl.sum()
    with_cost = vectorized_backtest(r, pos, cost_per_unit_turnover=10e-4).pnl.sum()
    assert with_cost < no_cost


def test_vectorized_backtest_stats_shape() -> None:
    r = _returns()
    pos = pd.Series(1.0, index=r.index)
    stats = vectorized_backtest(r, pos).stats
    assert set(stats.keys()) >= {
        "sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "calmar",
        "n_days",
        "turnover_ann",
    }


def test_purged_kfold_no_train_test_overlap() -> None:
    cv = PurgedKFold(n_splits=5, purge=3, embargo=3)
    for train, test in cv.split(n=100):
        assert len(set(train) & set(test)) == 0


def test_purged_kfold_purge_removes_neighbours() -> None:
    cv_no_purge = PurgedKFold(n_splits=5, purge=0, embargo=0)
    cv_purge = PurgedKFold(n_splits=5, purge=5, embargo=5)
    for (tr0, _te), (tr1, _) in zip(cv_no_purge.split(100), cv_purge.split(100), strict=True):
        # Purged train must be a strict subset of the unpurged train
        assert set(tr1.tolist()).issubset(set(tr0.tolist()))


def test_walk_forward_train_is_strictly_before_test() -> None:
    wf = WalkForward(initial_train=20, test_size=10, embargo=2)
    for train, test in wf.split(n=100):
        assert train.max() < test.min()
        assert test.min() - train.max() >= 2


def test_deflated_sharpe_within_unit_interval() -> None:
    rng = np.random.default_rng(0)
    sharpes = rng.normal(0, 0.3, size=50)
    champion = float(np.max(sharpes))
    p = deflated_sharpe(observed=champion, sharpes=sharpes, T=252 * 3)
    assert 0.0 <= p <= 1.0
