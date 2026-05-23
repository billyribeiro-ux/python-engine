# Live data — websockets and order events

Live trading is not "backtesting, but real". The shape of the data is different (it pushes; you don't pull), the failure modes are different (the socket drops at 3:47 PM and you need to recover without missed events), and the consequences of mistakes are immediate.

This chapter is the minimum a Python developer needs to handle live feeds reliably. It does not turn you into a low-latency engineer — for that, you'd be writing C++ on a tuned kernel — but it will keep a Python-driven strategy honest at the millisecond-to-second timescales that most retail and small-fund work targets.

## The shape of a live feed

A vendor exposes a WebSocket. You connect, authenticate, subscribe to symbols, and receive messages forever. Three message types matter:

1. **Quote** — bid/ask change.
2. **Trade** — a print on the tape.
3. **Bar** — a vendor-aggregated minute (or 5s, etc.) bar.

Plus heartbeats, status changes, and error messages.

## A minimum-viable WebSocket consumer

```python
import asyncio, json
import websockets

async def consume(uri: str, api_key: str, symbols: list[str]):
    async for ws in websockets.connect(uri, ping_interval=20, ping_timeout=20):
        try:
            await ws.send(json.dumps({"action": "auth", "key": api_key}))
            await ws.send(json.dumps({"action": "subscribe", "params": "T." + ",T.".join(symbols)}))
            async for raw in ws:
                msg = json.loads(raw)
                await handle(msg)
        except websockets.ConnectionClosed:
            print("socket dropped, reconnecting")
            continue
```

The `async for ws in websockets.connect(...)` pattern is the library's built-in auto-reconnect. Every iteration is a fresh connection; the loop runs forever.

`ping_interval=20` and `ping_timeout=20` are essential — without keep-alives, network equipment silently drops "idle" connections after a few minutes and you don't notice until you wonder why no trades have arrived for an hour.

## Handling backpressure

Your `handle` function runs sequentially per connection. If it's slow (writes to disk, runs a model), messages back up in the WebSocket library's internal buffer and eventually it disconnects you for being too slow to read.

The fix is to **decouple I/O from processing** with a queue:

```python
async def consume(uri, key, symbols, queue: asyncio.Queue):
    async for ws in websockets.connect(uri, ping_interval=20):
        try:
            await ws.send(...)
            async for raw in ws:
                try:
                    queue.put_nowait(raw)
                except asyncio.QueueFull:
                    # back-pressure: drop the oldest, log it
                    queue.get_nowait()
                    queue.put_nowait(raw)
        except websockets.ConnectionClosed:
            continue


async def process(queue: asyncio.Queue):
    while True:
        raw = await queue.get()
        msg = json.loads(raw)
        await handle(msg)


async def main():
    q = asyncio.Queue(maxsize=10_000)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(consume(URI, KEY, SYMBOLS, q))
        tg.create_task(process(q))
```

The `consume` task only does I/O and parses nothing heavy. The `process` task does the work. The queue absorbs spikes; when it fills, you have a *policy decision* to make (drop oldest, drop newest, alert, etc.) instead of silently disconnecting.

## State reconstruction after a disconnect

When the socket reconnects, you've missed messages. Three coping strategies:

1. **Snapshot + delta** — On (re)connect, request a full state snapshot from a REST endpoint, then start applying deltas. Most exchanges expose a "snapshot" endpoint for exactly this.
2. **Resubscribe and accept gap** — for non-critical signals (e.g. trade tape for a slow-moving strategy), just resubscribe and move on.
3. **Replay from the vendor's history** — if the vendor offers a recent-history endpoint, query for messages since the last seen sequence number.

Most retail-grade strategies are fine with (2). Production market-making, never. Pick the right one for your latency tolerance.

## Sequence numbers

If the vendor sends a sequence number on each message, **check for gaps**:

```python
last_seq = -1

async def handle(msg):
    global last_seq
    seq = msg.get("seq")
    if seq is None:
        return
    if last_seq >= 0 and seq != last_seq + 1:
        print(f"GAP: jumped from {last_seq} to {seq} ({seq - last_seq - 1} missed)")
        await resync()
    last_seq = seq
```

A silent gap is the #1 reason production strategies trade on stale data.

## Order events

Once you go live, you also have a *second* WebSocket — your broker's order-events stream. ACKs, fills, rejects, cancels. Two non-obvious points:

1. **The stream is the source of truth.** Don't trust the REST response from your `POST /orders` call; trust the ACK that comes back on the event stream. (REST responses can race with the matching engine.)
2. **You can receive an event for an order you don't recognise.** Order-id collisions across processes, partial fills resurfacing after a reconnect, manually-entered orders from another terminal. Make `handle_fill` idempotent and resilient to unknown IDs.

```python
async def handle_event(msg):
    oid = msg["order_id"]
    state = orders.get(oid)
    if state is None:
        log.warning("event for unknown order %s: %r", oid, msg)
        # could be a manual order; ignore or recover from REST
        return
    state.apply(msg)
    if state.is_terminal:
        del orders[oid]
```

The `orders` dict is your in-memory view; reconciliation against the broker's view should run periodically.

## A simple bar aggregator from trades

You're subscribed to the trade tape and want to construct your own 1-minute bars (because the vendor's bars lag, or because you want a custom interval):

```python
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: int
    n: int = 0

class BarAggregator:
    def __init__(self, interval_sec: int):
        self.interval = interval_sec
        self.bars: dict[tuple[str, int], Bar] = {}
        self.completed: list[tuple[str, int, Bar]] = []

    def on_trade(self, symbol: str, ts_sec: int, price: float, size: int) -> None:
        bucket = ts_sec - (ts_sec % self.interval)
        key = (symbol, bucket)
        bar = self.bars.get(key)
        if bar is None:
            # close prior bars for this symbol
            for (s, b), prior in list(self.bars.items()):
                if s == symbol and b < bucket:
                    self.completed.append((s, b, prior))
                    del self.bars[(s, b)]
            self.bars[key] = Bar(price, price, price, price, size, 1)
        else:
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += size
            bar.n += 1
```

Live aggregation belongs in a tight Python loop — too fast for pandas, too small for numba. A class with mutable state is the right shape.

## Persistence: a write-ahead log

For anything you might want to replay later, write to disk *before* you process:

```python
import aiofiles

async def process(queue: asyncio.Queue, wal_path: str):
    async with aiofiles.open(wal_path, "ab") as wal:
        while True:
            raw = await queue.get()
            await wal.write(raw + b"\n")
            await wal.flush()
            msg = orjson.loads(raw)
            await handle(msg)
```

The write-ahead log gives you a perfect replay of every message you ever saw. The trading-engine equivalent of a backup. Use `orjson` (~10× faster than stdlib `json`) and an append-only file format.

## Latency budget

For Python-driven strategies (most of this course):

| Stage | Typical latency |
|---|---|
| Socket → Python | 0.1–1 ms |
| JSON parse (orjson) | 5–20 µs |
| Bar aggregation | 1–5 µs |
| Model inference (XGBoost) | 0.1–2 ms |
| Strategy decision | 10–100 µs |
| Order submission (REST) | 50–500 ms |
| Order submission (FIX) | 1–5 ms |

Python is fine for anything that lives at "tens of milliseconds" or slower — which is most retail and low-frequency systematic. For "microseconds matter" — HFT, market making in active products — you're writing C++ or Rust, and Python is just the research environment.

## Things that will bite you

!!! danger "The clock"
    Most vendors send exchange-local timestamps. Convert to UTC at the boundary. Time-of-day strategies that work in dev and silently break in production are usually clock bugs.

!!! danger "Numeric types"
    Prices come as strings or floats. Decimals matter for orders — `0.1 + 0.2 != 0.3` in float — so for order prices, round to the tick size with `decimal.Decimal` or integer ticks. For analytics, float64 is fine.

!!! danger "Authentication tokens expire"
    Long-running socket sessions sometimes need to re-auth. Vendors handle this differently. Read the docs; implement a graceful re-auth path.

!!! danger "Markets close"
    Your socket may stay open all night, sending nothing. Don't interpret silence as a problem during off-hours; do interpret it as a problem during regular trading hours.

## Bottom line

A robust live-data pipeline is:

```
WebSocket ──► consume() ──► Queue ──► process() ──► WAL ──► handlers ──► trade/log/state
                                                                              │
                                          (snapshot + delta on reconnect) ◄──┘
```

You can fit that on one page of Python. Get this right and the rest of live trading is "your strategy logic, but with real consequences."

## End of Module 5

You now have the data layer that everything from Module 7 onward will draw from. The next module is **statistics + probability for traders** — where we deliberately *break* most of what you learned in undergrad stats class, because finance violates almost every assumption it teaches.

Continue to **[Module 6 — Statistics + Probability](../06-stats-and-probability/index.md)**.
