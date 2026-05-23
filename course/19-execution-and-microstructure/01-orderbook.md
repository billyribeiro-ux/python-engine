# Order book dynamics — what's actually happening

An order book is the list of all standing buy and sell limit orders at each price level. The top-of-book bid and ask are the best prices; depth at each level tells you how much you can trade without moving the price. Order book dynamics — how the book updates with each new order, fill, and cancel — are the mechanics underneath every price tick.

This chapter is the working primer.

## The structure

```
ASK side (sell limits):
  $100.05  x 200 shares
  $100.04  x 500
  $100.03  x 300
  $100.02  x 800
  $100.01  x 1000           ← best ask
  ━━━━━━━━━━━━━━━━━━━━━━
  $100.00  x 700            ← best bid
  $99.99   x 1200
  $99.98   x 400
  $99.97   x 900
  $99.96   x 600
BID side (buy limits):
```

The **spread** is best_ask − best_bid (here: $0.01).

The **depth** at the best is `bid_size + ask_size` (here: 1700).

## Three event types

The book updates on:

1. **New limit order** — adds to one side at a price level.
2. **Cancel** — removes part or all of an existing order.
3. **Trade (cross)** — a market order eats the top of one side, possibly down multiple levels.

The order book is a **state machine**. Every public market data feed is just a stream of these events; the order book at any moment is the result of replaying them from the start of the session.

## Reconstructing the book from a feed

For a US equity from a SIP (consolidated) or proprietary L2 feed:

```python
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class OrderBook:
    bids: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    asks: dict[float, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, side: str, price: float, size: int) -> None:
        book = self.bids if side == "B" else self.asks
        book[price] += size

    def cancel(self, side: str, price: float, size: int) -> None:
        book = self.bids if side == "B" else self.asks
        book[price] -= size
        if book[price] <= 0:
            del book[price]

    def trade(self, side: str, price: float, size: int) -> None:
        # Trade hits the resting side; e.g., "B" trade lifts the ask
        book = self.asks if side == "B" else self.bids
        if price in book:
            book[price] -= size
            if book[price] <= 0:
                del book[price]

    @property
    def best_bid(self) -> tuple[float, int] | None:
        if not self.bids: return None
        p = max(self.bids); return p, self.bids[p]

    @property
    def best_ask(self) -> tuple[float, int] | None:
        if not self.asks: return None
        p = min(self.asks); return p, self.asks[p]

    @property
    def mid(self) -> float | None:
        bb = self.best_bid; ba = self.best_ask
        if bb is None or ba is None: return None
        return (bb[0] + ba[0]) / 2

    @property
    def spread_bps(self) -> float | None:
        bb = self.best_bid; ba = self.best_ask
        if bb is None or ba is None: return None
        return (ba[0] - bb[0]) / self.mid * 10000
```

For production, use a high-performance container — Python dicts are fine up to ~10k events/sec; beyond that, use `sortedcontainers.SortedDict` or move to a Rust/C++ extension.

## Microstructural features from the book

Useful features for any short-horizon predictor:

- **Spread (bps)** — `(best_ask - best_bid) / mid * 10000`.
- **Depth ratio** — `bid_size / ask_size` at the top. Positive imbalance → upward pressure.
- **Order-flow imbalance (OFI)** — signed sum of size changes at the top over a short window.
- **Effective spread** — for a trade at price p with mid m, `2 * |p - m|`. Measures the realised cost of crossing.
- **Realised spread** — for a trade at price p with mid m and mid m' some time later, `2 * (p - m')`. Captures the post-trade reversion.

```python
def order_flow_imbalance(book_t0: OrderBook, book_t1: OrderBook) -> int:
    """Signed change in top-of-book bid and ask size between two snapshots."""
    bid_change = book_t1.best_bid[1] - book_t0.best_bid[1] if book_t0.best_bid and book_t1.best_bid else 0
    ask_change = book_t1.best_ask[1] - book_t0.best_ask[1] if book_t0.best_ask and book_t1.best_ask else 0
    return bid_change - ask_change   # positive: bid grew or ask shrunk
```

Cont, Stoikov, Talreja (2010) showed OFI predicts the next-minute mid-price change at non-trivial information levels — one of the cleanest short-horizon signals.

## Queue position

For your resting limit order, *where in line* you are at your price level matters. If you're at the back of a 100k-share queue and 50k shares get traded, you haven't filled. The market typically uses **price-time priority**: first to arrive at the price level gets first fill.

Estimating queue position from public feed:

```python
def estimate_queue_position(my_order_price: float, my_order_time: float,
                              feed_history: list[dict], my_side: str) -> int:
    """Estimate the number of shares ahead of me at my price level.
    Walk through historical events at the price level, ordered by time."""
    ahead = 0
    for ev in feed_history:
        if ev["side"] != my_side or ev["price"] != my_order_price:
            continue
        if ev["timestamp"] < my_order_time:
            if ev["type"] == "add":
                ahead += ev["size"]
            elif ev["type"] in ("cancel", "trade"):
                ahead -= ev["size"]
    return max(0, ahead)
```

For HFT market-makers, knowing your queue position decides whether you'll fill on the next adverse print. For everyone else, the implications are mostly: don't expect a thin-spread limit order to fill quickly.

## A worked example: spread regime monitor

```python
import pandas as pd

def spread_regime_alert(book_history: list[OrderBook], window: int = 60):
    """Alert when spread is much wider than recent baseline."""
    spreads = pd.Series([b.spread_bps for b in book_history])
    rolling_median = spreads.rolling(window).median()
    rolling_p90 = spreads.rolling(window).quantile(0.9)
    current = spreads.iloc[-1]
    alerts = []
    for i in range(window, len(spreads)):
        if spreads.iloc[i] > rolling_p90.iloc[i] * 2:
            alerts.append({"index": i, "spread": spreads.iloc[i],
                            "baseline_p90": rolling_p90.iloc[i]})
    return alerts
```

Spread regime shifts (compression → expansion) are very strong leading indicators for vol regime shifts. The microstructure scanner in Module 18 is built on this idea.

## Pitfalls

!!! warning "Hidden orders"
    Most modern exchanges support hidden / iceberg orders that don't show in the visible book. Your queue position estimate is wrong by however many hidden shares are at your price.

!!! warning "Multi-venue fragmentation"
    A US equity trades on 10+ venues. The SIP book is the union; vendor-specific feeds may show only one venue. Using one-venue depth as "the depth" is misleading.

!!! warning "Latency vs causality"
    A "trade at ask" followed (in your feed) by "ask reduced" is the same event from different perspectives. Don't double-count.

!!! warning "Lit vs dark"
    A meaningful fraction of US equity volume executes in dark pools. The visible book misses it. For depth-aware analytics, augment with TRF (trade-reporting facility) data.

## Bottom line

The order book is:

- A **state machine** — every public update changes it.
- The **source of microstructure features** (spread, depth, OFI, queue).
- **Reconstruction-cost-sensitive** — production reconstructions are in optimised C++/Rust.

For most retail and small-fund work, you'll consume a vendor's pre-computed L1 stats. For serious execution work, reconstruct the book yourself.

Continue to **[Kyle's lambda and VPIN](02-kyle-vpin.md)**.
