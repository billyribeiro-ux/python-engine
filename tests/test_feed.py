"""Smoke tests for the data adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data import Feed, YFinanceFeed, default_feed
from engine.data.feed import PolygonFeed, _normalise_bars
from engine.utils import has_network


def test_yfinance_feed_satisfies_protocol() -> None:
    f = YFinanceFeed()
    assert isinstance(f, Feed)
    assert f.name == "yfinance"


def test_default_feed_falls_back_to_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHON_ENGINE_FEED", raising=False)
    assert isinstance(default_feed(), YFinanceFeed)


def test_default_feed_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHON_ENGINE_FEED", "nope")
    with pytest.raises(ValueError, match="Unknown feed"):
        default_feed()


def test_stub_feed_explains_itself() -> None:
    with pytest.raises(NotImplementedError, match="placeholder"):
        PolygonFeed()


def test_normalise_bars_empty() -> None:
    out = _normalise_bars(pd.DataFrame())
    assert out.empty
    assert out.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "timestamp"


def test_normalise_bars_renames_and_sorts() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Volume": [100, 200],
            "Adj Close": [1.1, 2.1],  # should be dropped
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )
    out = _normalise_bars(raw)
    assert out.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert out.index.is_monotonic_increasing
    assert out.index.tz is not None


@pytest.mark.network
@pytest.mark.skipif(not has_network(), reason="no network")
def test_yfinance_pulls_spy() -> None:
    f = YFinanceFeed()
    bars = f.bars("SPY", "2024-01-02", "2024-01-15", interval="1d")
    # Yahoo can throttle/empty in CI; tolerate empties but if non-empty, it
    # must have the canonical shape.
    if not bars.empty:
        assert bars.columns.tolist() == ["open", "high", "low", "close", "volume"]
        assert bars.index.tz is not None
