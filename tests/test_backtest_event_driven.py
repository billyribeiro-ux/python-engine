"""Tests for engine.backtest.event_driven."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest import (
    Broker,
    Order,
    Portfolio,
    Position,
    event_driven_backtest,
)


def _make_bars(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=pd.date_range("2024-01-02", periods=len(prices), freq="B"),
    )


def test_position_apply_fill_opens_then_closes() -> None:
    pos = Position(symbol="X")
    pos.apply_fill("BUY", 10, 100.0)
    assert pos.qty == 10
    assert pos.avg_price == 100.0
    realised = pos.apply_fill("SELL", 10, 110.0)
    assert pos.qty == 0
    assert pos.avg_price == 0.0
    assert realised == 100.0  # 10 shares * $10 profit


def test_position_apply_fill_partial_close() -> None:
    pos = Position(symbol="X")
    pos.apply_fill("BUY", 10, 100.0)
    realised = pos.apply_fill("SELL", 4, 110.0)
    assert pos.qty == 6
    assert pos.avg_price == 100.0  # remaining unchanged
    assert realised == 40.0


def test_position_apply_fill_flip() -> None:
    pos = Position(symbol="X")
    pos.apply_fill("BUY", 10, 100.0)
    realised = pos.apply_fill("SELL", 15, 110.0)
    assert pos.qty == -5
    assert pos.avg_price == 110.0  # leftover opens at fill price
    assert realised == 100.0  # closed 10 shares at $10 profit


def test_broker_market_order_fills_with_slippage() -> None:
    pf = Portfolio(cash=10_000.0)
    broker = Broker(portfolio=pf, slippage_bps=10.0)
    order = Order(symbol="X", side="BUY", qty=10)
    broker.submit(order)
    bar = {"X": {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}}
    broker.match(pd.Timestamp("2024-01-02"), bar)
    assert order.status == "FILLED"
    assert order.avg_fill_price > 100.0  # paid slippage
    assert pf.position("X").qty == 10
    assert pf.cash < 10_000.0


def test_broker_limit_order_fills_only_when_touched() -> None:
    pf = Portfolio(cash=10_000.0)
    broker = Broker(portfolio=pf, slippage_bps=0.0)
    order = Order(symbol="X", side="BUY", qty=10, order_type="LIMIT", limit_price=95.0)
    broker.submit(order)
    # Bar high above limit, low above limit — no fill.
    bar = {"X": {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}}
    broker.match(pd.Timestamp("2024-01-02"), bar)
    assert order.status == "OPEN"
    # Next bar drops low through the limit — fills.
    bar2 = {"X": {"open": 96.0, "high": 96.0, "low": 94.0, "close": 95.5}}
    broker.match(pd.Timestamp("2024-01-03"), bar2)
    assert order.status == "FILLED"
    assert order.avg_fill_price == 95.0


def test_event_driven_buy_and_hold() -> None:
    bars = {"X": _make_bars([100.0, 101.0, 102.0, 103.0, 104.0])}

    submitted = {"done": False}

    def strategy(ts, bar, broker):
        if not submitted["done"]:
            broker.submit(Order(symbol="X", side="BUY", qty=10))
            submitted["done"] = True

    result = event_driven_backtest(bars, strategy, initial_cash=10_000.0, slippage_bps=0.0)
    # 10 shares * ($104 - $101) buy price = $30 of P&L by end + initial cash.
    final_equity = float(result.equity.iloc[-1])
    assert final_equity > 10_000.0
    assert len(result.fills) == 1


def test_event_driven_no_lookahead() -> None:
    """A strategy submitting on bar t cannot fill at bar t — only t+1."""
    bars = {"X": _make_bars([100.0, 200.0, 50.0, 100.0])}
    fills_count = {"n": 0}

    def strategy(ts, bar, broker):
        if fills_count["n"] == 0:
            broker.submit(Order(symbol="X", side="BUY", qty=1))
            fills_count["n"] += 1

    result = event_driven_backtest(bars, strategy, initial_cash=1_000.0, slippage_bps=0.0)
    # Submitted on bar 0; should fill at bar 1 open = $200, NOT bar 0 open = $100.
    assert len(result.fills) == 1
    assert float(result.fills.iloc[0]["price"]) == 200.0


def test_event_driven_stats_shape() -> None:
    rng = np.random.default_rng(0)
    prices = (100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=60))).tolist()
    bars = {"X": _make_bars(prices)}

    def strategy(ts, bar, broker):
        pass

    result = event_driven_backtest(bars, strategy, initial_cash=10_000.0)
    assert set(result.stats.keys()) >= {
        "sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "n_fills",
    }
