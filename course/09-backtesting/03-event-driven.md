# An event-driven engine

The vectorised backtester works when the strategy can be expressed as "position is a function of past data". For anything where the *order itself* has a life of its own — limit orders that may or may not fill, stop-losses, take-profits, partial fills, OCO orders, options exercise — you need an event-driven engine that processes bars one at a time and maintains explicit order state.

This chapter sketches the architecture and shows you a minimal working implementation. For production, the open-source [`backtrader`](https://www.backtrader.com/), [`nautilus_trader`](https://nautilustrader.io/), and [`zipline`](https://github.com/zipline-live/zipline-trader) are all reasonable bases.

## The architecture

An event-driven backtester has four components:

```
data feed (bars) ──► strategy ──► orders ──► matching engine ──► fills ──► broker state
                          ▲                                                    │
                          └─────────────── position / P&L ◄────────────────────┘
```

On each bar:

1. **Data feed** publishes the next bar.
2. **Strategy** sees the bar plus its history, decides on orders.
3. **Matching engine** simulates the broker: for each open order, decides whether it fills on this bar and at what price.
4. **Broker state** updates positions, cash, P&L.

Repeat.

## A minimal implementation

```python
from dataclasses import dataclass, field
from collections import deque
from typing import Literal

@dataclass(slots=True)
class Order:
    side: Literal["BUY", "SELL"]
    qty: float
    kind: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    limit: float | None = None
    stop: float | None = None
    status: str = "WORKING"
    fill_qty: float = 0.0
    fill_price: float = 0.0
    id: int = 0

@dataclass(slots=True)
class Position:
    qty: float = 0.0
    avg_px: float = 0.0
    realized: float = 0.0

    def apply_fill(self, side, qty, price):
        signed = qty if side == "BUY" else -qty
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            # adding to position
            new_qty = self.qty + signed
            self.avg_px = (self.avg_px * self.qty + price * signed) / new_qty if new_qty != 0 else 0
            self.qty = new_qty
        else:
            # reducing / flipping
            close_qty = min(abs(signed), abs(self.qty))
            self.realized += close_qty * (price - self.avg_px) * (1 if self.qty > 0 else -1)
            self.qty += signed
            if (self.qty > 0 and signed < 0) or (self.qty < 0 and signed > 0):
                # didn't flip
                pass
            elif self.qty != 0:
                self.avg_px = price
            else:
                self.avg_px = 0


class Broker:
    def __init__(self):
        self.orders: list[Order] = []
        self.position = Position()
        self.next_id = 1

    def submit(self, order: Order) -> int:
        order.id = self.next_id
        self.next_id += 1
        self.orders.append(order)
        return order.id

    def match(self, bar) -> list[Order]:
        """For each working order, decide if it fills on this bar."""
        filled = []
        for o in self.orders:
            if o.status != "WORKING":
                continue
            if o.kind == "MARKET":
                # next-bar-open fill
                price = bar["open"]
            elif o.kind == "LIMIT":
                if o.side == "BUY" and bar["low"] <= o.limit:
                    price = min(o.limit, bar["open"])      # fill at limit or better
                elif o.side == "SELL" and bar["high"] >= o.limit:
                    price = max(o.limit, bar["open"])
                else:
                    continue
            elif o.kind == "STOP":
                if o.side == "BUY" and bar["high"] >= o.stop:
                    price = max(o.stop, bar["open"])       # stop becomes market
                elif o.side == "SELL" and bar["low"] <= o.stop:
                    price = min(o.stop, bar["open"])
                else:
                    continue
            o.fill_qty = o.qty
            o.fill_price = price
            o.status = "FILLED"
            self.position.apply_fill(o.side, o.qty, price)
            filled.append(o)
        return filled


class EventDrivenBacktest:
    def __init__(self, bars, strategy):
        self.bars = bars                       # iterable of bar dicts
        self.strategy = strategy
        self.broker = Broker()
        self.history = []

    def run(self):
        for bar in self.bars:
            self.broker.match(bar)
            self.strategy.on_bar(bar, self.broker)
            self.history.append({
                "ts": bar["ts"],
                "close": bar["close"],
                "position": self.broker.position.qty,
                "realized": self.broker.position.realized,
                "unrealized": self.broker.position.qty * (bar["close"] - self.broker.position.avg_px),
            })
        return self.history
```

About 100 lines. A working strategy:

```python
class TurtleStrategy:
    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.history = deque(maxlen=max(entry_window, exit_window) + 1)
        self.entry_high = self.exit_low = None

    def on_bar(self, bar, broker):
        self.history.append(bar)
        if len(self.history) < self.entry_window:
            return
        highs = [b["high"] for b in list(self.history)[-self.entry_window:]]
        lows = [b["low"] for b in list(self.history)[-self.exit_window:]]
        if broker.position.qty == 0 and bar["close"] > max(highs[:-1]):
            broker.submit(Order(side="BUY", qty=100, kind="MARKET"))
        elif broker.position.qty > 0 and bar["close"] < min(lows[:-1]):
            broker.submit(Order(side="SELL", qty=broker.position.qty, kind="MARKET"))
```

## When event-driven is necessary

- **Limit orders.** The strategy issues a limit; whether it fills depends on intrabar dynamics. Vectorised can't represent this.
- **Stops and brackets.** A stop-loss order has a price contingent on the bar's high/low — vectorised assumes you knew the result ahead of time.
- **Options.** Exercise decisions and assignment are events.
- **Realistic queue position.** "How much of the limit at this price filled?" requires a matching engine that knows the book.

## When event-driven is overkill

- Daily-bar strategies with no limit orders or stops.
- Cross-sectional strategies (long-short portfolios re-balanced periodically).
- Anything where "position = function of past data" expresses the strategy completely.

For the rest of this course, **vectorised is the default**. The few strategies that need event-driven are flagged.

## Speed and the case for production engines

A pure-Python event-driven backtester runs at ~10k–100k bars per second. For minute data over 5 years × 500 symbols, that's hours. The fixes:

- **`nautilus_trader`** — Rust-backed event-driven engine; ~10× faster than pure Python.
- **`vectorbt`** — vectorised but with limit/stop simulation via numba; orders of magnitude faster for sweeps.
- **`backtesting.py`** — small, ergonomic; about 2× faster than naive Python.

For one-off strategy R&D, the hand-rolled engine above is fine. For production parameter sweeps, use one of the libraries.

## Pitfalls

!!! warning "Optimistic limit fills"
    The naive "if low ≤ limit, fill at limit" assumes queue priority and that everyone trading at your limit gets filled. In reality, with shared queues, you may not fill. Add a `fill_probability` knob or model queue position explicitly.

!!! warning "Stop slippage"
    Real stops fill at the *next available price*, not the stop trigger. Especially on gap downs. Model stop fills with explicit slippage past the stop (chapter 6).

!!! warning "Same-bar logic ordering"
    Did the stop trigger before the new entry? In a single bar, that's ambiguous. Use a deterministic ordering (e.g. stops checked first, then strategy logic). Document it.

!!! warning "Cash and margin"
    The minimal broker above doesn't track cash. For leverage-constrained strategies, you need margin checks, mark-to-market, and maintenance calls.

## Bottom line

For most quant strategies, the vectorised engine in chapter 2 is correct and fast. Build the event-driven version only when the strategy logic genuinely depends on order events — stops, limits, brackets, options exercise.

Continue to **[Cross-validation that doesn't leak](04-cv.md)**.
