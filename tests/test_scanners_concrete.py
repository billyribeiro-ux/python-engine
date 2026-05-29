"""Tests for engine.scanners.concrete using an in-memory fake feed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.scanners import RealisedVolRankScanner, ZScoreMomentumScanner


class _FakeFeed:
    """Minimal Feed: hands back stored bars by symbol, slicing on date."""

    def __init__(self, store: dict[str, pd.DataFrame]) -> None:
        self.store = store

    def bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if symbol not in self.store:
            return pd.DataFrame()
        df = self.store[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask].copy()


def _make_series(seed: int, mu: float, sigma: float, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, size=n)
    prices = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0},
        index=idx,
    )


def test_zscore_scanner_returns_dataframe_with_expected_columns() -> None:
    feed = _FakeFeed(
        {
            "AAA": _make_series(seed=0, mu=0.0005, sigma=0.01),
            "BBB": _make_series(seed=1, mu=-0.0005, sigma=0.01),
        }
    )
    s = ZScoreMomentumScanner(feed=feed, lookback_days=10)
    out = s.scan(["AAA", "BBB"], asof=pd.Timestamp("2024-06-01"))
    assert {"symbol", "last_return", "mean", "std", "zscore", "abs_zscore"} <= set(out.columns)
    # Sorted descending by abs_zscore
    assert (out["abs_zscore"].diff().dropna() <= 0).all()


def test_zscore_scanner_direction_filter_long_only() -> None:
    feed = _FakeFeed(
        {
            "AAA": _make_series(seed=0, mu=0.0005, sigma=0.01),
            "BBB": _make_series(seed=1, mu=-0.0005, sigma=0.01),
        }
    )
    s = ZScoreMomentumScanner(feed=feed, lookback_days=10, direction="long")
    out = s.scan(["AAA", "BBB"], asof=pd.Timestamp("2024-06-01"))
    # "long" candidates have negative z-scores (mean-reversion buy after a down move).
    if not out.empty:
        assert (out["zscore"] < 0).all()


def test_zscore_scanner_handles_unknown_symbol_gracefully() -> None:
    feed = _FakeFeed({"AAA": _make_series(seed=0, mu=0.0005, sigma=0.01)})
    s = ZScoreMomentumScanner(feed=feed, lookback_days=10)
    out = s.scan(["AAA", "MISSING"], asof=pd.Timestamp("2024-06-01"))
    assert "MISSING" not in out["symbol"].values


def test_rvol_rank_scanner_outputs_rank_in_unit_interval() -> None:
    feed = _FakeFeed(
        {
            "AAA": _make_series(seed=0, mu=0.0, sigma=0.01),
            "BBB": _make_series(seed=1, mu=0.0, sigma=0.02),
            "CCC": _make_series(seed=2, mu=0.0, sigma=0.005),
        }
    )
    s = RealisedVolRankScanner(feed=feed, short_window_days=20, history_days=200)
    out = s.scan(["AAA", "BBB", "CCC"], asof=pd.Timestamp("2024-06-01"))
    assert (out["rvol_rank"].between(0.0, 1.0)).all()
    assert {"rvol_current", "rvol_min", "rvol_max", "rvol_rank"} <= set(out.columns)


def test_concrete_scanners_registered() -> None:
    from engine.scanners import registry

    assert "zscore-momentum" in registry
    assert "rvol-rank" in registry
