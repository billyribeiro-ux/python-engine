"""Tests for the Parquet cache wrapper."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data import Feed, OptionsChain, ParquetCache


class _FixtureFeed:
    """A deterministic feed that hands out a known frame and counts calls."""

    name = "fixture"

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.bars_calls = 0

    def bars(self, symbol, start, end, interval="1d"):
        self.bars_calls += 1
        return self.df

    def option_chain(self, symbol, expiry=None) -> OptionsChain:
        raise NotImplementedError


@pytest.fixture
def fixture_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
            "volume": [1_000_000] * 10,
        },
        index=idx,
    ).astype(float)


def test_cache_satisfies_protocol(fixture_bars, tmp_path) -> None:
    cache = ParquetCache(_FixtureFeed(fixture_bars), tmp_path)
    assert isinstance(cache, Feed)
    assert cache.name == "parquet-cache"


def test_first_call_hits_upstream_second_serves_from_disk(fixture_bars, tmp_path) -> None:
    upstream = _FixtureFeed(fixture_bars)
    cache = ParquetCache(upstream, tmp_path)

    out1 = cache.bars("SPY", "2024-01-02", "2024-01-11")
    assert upstream.bars_calls == 1
    assert not out1.empty
    assert out1.columns.tolist() == ["open", "high", "low", "close", "volume"]

    out2 = cache.bars("SPY", "2024-01-02", "2024-01-11")
    assert upstream.bars_calls == 1  # served from disk, no upstream call
    pd.testing.assert_frame_equal(out1, out2)


def test_cache_returns_only_requested_window(fixture_bars, tmp_path) -> None:
    cache = ParquetCache(_FixtureFeed(fixture_bars), tmp_path)
    cache.bars("SPY", "2024-01-02", "2024-01-11")
    narrow = cache.bars("SPY", "2024-01-05", "2024-01-08")
    assert len(narrow) == 4
    assert narrow.index.min() >= pd.Timestamp("2024-01-05", tz="UTC")
    assert narrow.index.max() <= pd.Timestamp("2024-01-08", tz="UTC")
