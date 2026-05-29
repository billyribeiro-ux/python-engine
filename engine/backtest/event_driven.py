"""Event-driven backtester.

A simple but honest event loop:

    for bar in bars:
        strategy.on_bar(bar, broker)        # may submit orders
        broker.match(bar)                   # fills against this bar's prices
        recorder.snapshot(bar.timestamp, broker.portfolio)

Compared to the vectorised engine in ``engine.backtest.vectorized``, this
exposes per-order fills, partial executions, slippage at the trade level,
and a portfolio object you can interrogate at any point. Slower, but the
right tool for anything multi-asset or with realistic execution rules.

Designed for daily-or-coarser equity bars. For intraday/order-book work
the engine in ``engine.execution`` is more appropriate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = "MARKET"
    limit_price: float | None = None
    submitted_at: pd.Timestamp | None = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    status: Literal["OPEN", "FILLED", "CANCELLED"] = "OPEN"

    def signed_qty(self) -> float:
        return self.qty if self.side == "BUY" else -self.qty


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_price)

    def apply_fill(self, side: Side, qty: float, price: float) -> float:
        """Apply a fill. Returns realised P&L on closed quantity."""
        signed = qty if side == "BUY" else -qty
        new_qty = self.qty + signed
        realised = 0.0
        if self.qty == 0 or np.sign(self.qty) == np.sign(signed):
            # Opening or adding to a position — re-weight average price.
            total_cost = self.avg_price * self.qty + price * signed
            self.avg_price = total_cost / new_qty if new_qty != 0 else 0.0
        else:
            # Reducing or flipping — book realised P&L on the closed portion.
            closed = min(abs(signed), abs(self.qty))
            realised = closed * (price - self.avg_price) * np.sign(self.qty)
            if abs(signed) > abs(self.qty):
                # Position flipped — leftover qty opens at this price.
                self.avg_price = price
            # else: avg_price unchanged on partial close
        self.qty = new_qty
        if self.qty == 0:
            self.avg_price = 0.0
        return float(realised)


@dataclass(slots=True)
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def equity(self, prices: dict[str, float]) -> float:
        mv = sum(p.market_value(prices.get(p.symbol, p.avg_price)) for p in self.positions.values())
        return float(self.cash + mv)


@dataclass(slots=True)
class Broker:
    """Simple bar-close matching engine with linear slippage.

    MARKET orders fill at next-bar close + ``slippage_bps`` adverse.
    LIMIT orders fill only when the next-bar OHLC crosses the limit.
    """

    portfolio: Portfolio
    slippage_bps: float = 1.0
    commission_per_share: float = 0.0
    open_orders: list[Order] = field(default_factory=list)
    fills: list[tuple[pd.Timestamp, str, Side, float, float]] = field(default_factory=list)

    def submit(self, order: Order) -> None:
        self.open_orders.append(order)

    def cancel_all(self) -> None:
        for o in self.open_orders:
            if o.status == "OPEN":
                o.status = "CANCELLED"
        self.open_orders = [o for o in self.open_orders if o.status == "OPEN"]

    def match(self, ts: pd.Timestamp, bar: dict[str, dict[str, float]]) -> None:
        """Try to fill each open order against this bar.

        ``bar`` is ``{symbol: {open, high, low, close}}``. MARKET orders
        fill at ``open + slippage`` (we already advanced one bar before
        calling this — see EventDrivenBacktest). LIMIT orders fill only if
        the limit is touched intrabar; the fill price is the limit.
        """
        still_open: list[Order] = []
        for order in self.open_orders:
            if order.status != "OPEN" or order.symbol not in bar:
                still_open.append(order)
                continue
            ohlc = bar[order.symbol]
            fill_price: float | None = None
            if order.order_type == "MARKET":
                slip = self.slippage_bps * 1e-4 * ohlc["open"]
                fill_price = ohlc["open"] + (slip if order.side == "BUY" else -slip)
            else:
                assert order.limit_price is not None
                if (order.side == "BUY" and ohlc["low"] <= order.limit_price) or (
                    order.side == "SELL" and ohlc["high"] >= order.limit_price
                ):
                    fill_price = order.limit_price
            if fill_price is None:
                still_open.append(order)
                continue
            # Execute.
            qty_to_fill = order.qty - order.filled_qty
            position = self.portfolio.position(order.symbol)
            realised = position.apply_fill(order.side, qty_to_fill, fill_price)
            cost = qty_to_fill * fill_price + self.commission_per_share * qty_to_fill
            self.portfolio.cash += (
                -cost if order.side == "BUY" else cost - 2 * self.commission_per_share * qty_to_fill
            )
            self.portfolio.realised_pnl += realised
            order.filled_qty = order.qty
            order.avg_fill_price = fill_price
            order.status = "FILLED"
            self.fills.append((ts, order.symbol, order.side, qty_to_fill, fill_price))
        self.open_orders = still_open


@dataclass(frozen=True, slots=True)
class EventDrivenResult:
    equity: pd.Series
    fills: pd.DataFrame
    final_positions: dict[str, Position]
    realised_pnl: float

    @property
    def stats(self) -> dict[str, float]:
        r = self.equity.pct_change().dropna()
        if r.empty:
            return {
                "sharpe": float("nan"),
                "ann_return": 0.0,
                "ann_vol": 0.0,
                "max_drawdown": 0.0,
                "n_fills": len(self.fills),
            }
        ann = 252
        ann_ret = r.mean() * ann
        ann_vol = r.std(ddof=1) * np.sqrt(ann)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        peak = self.equity.cummax()
        dd = (self.equity / peak - 1.0).min()
        return {
            "sharpe": float(sharpe),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "max_drawdown": float(dd),
            "n_fills": len(self.fills),
        }


StrategyFn = Callable[[pd.Timestamp, dict[str, dict[str, float]], Broker], None]


def event_driven_backtest(
    bars: dict[str, pd.DataFrame],
    strategy: StrategyFn,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 1.0,
    commission_per_share: float = 0.0,
) -> EventDrivenResult:
    """Run an event-driven backtest.

    Parameters
    ----------
    bars : dict[symbol -> DataFrame with columns open/high/low/close]
        All DataFrames must share an index (the timeline).
    strategy : callable
        ``strategy(ts, bar, broker)`` — called once per bar, *before* the
        bar's open is observable for execution. Submits orders via
        ``broker.submit(...)``. Orders submitted on bar t execute against
        bar t+1's prices — this is what makes the backtest honest.
    initial_cash : starting cash balance.
    slippage_bps : linear slippage on market orders.
    commission_per_share : per-share commission charged on both buy & sell.

    Returns
    -------
    EventDrivenResult
    """
    if not bars:
        raise ValueError("bars must contain at least one symbol")
    symbols = list(bars)
    index = bars[symbols[0]].index
    for s in symbols[1:]:
        if not bars[s].index.equals(index):
            raise ValueError("all bar DataFrames must share an index")

    portfolio = Portfolio(cash=initial_cash)
    broker = Broker(
        portfolio=portfolio,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
    )
    equity_curve: list[float] = []

    for i, ts in enumerate(index):
        bar_view = {s: bars[s].iloc[i].to_dict() for s in symbols}
        # Strategy sees bar i, submits orders. They execute against bar i+1.
        strategy(ts, bar_view, broker)
        if i + 1 < len(index):
            next_bar = {s: bars[s].iloc[i + 1].to_dict() for s in symbols}
            broker.match(index[i + 1], next_bar)
        prices = {s: float(bars[s].iloc[i]["close"]) for s in symbols}
        equity_curve.append(portfolio.equity(prices))

    fills_df = pd.DataFrame(broker.fills, columns=["timestamp", "symbol", "side", "qty", "price"])
    return EventDrivenResult(
        equity=pd.Series(equity_curve, index=index, name="equity"),
        fills=fills_df,
        final_positions=dict(portfolio.positions),
        realised_pnl=portfolio.realised_pnl,
    )
