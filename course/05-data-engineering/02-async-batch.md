# Async batch loaders with backoff

You have 5,000 symbols and a vendor that takes 200 ms per request. Sequentially that's 17 minutes. With careful concurrency, it's 30 seconds. With careless concurrency, it's a ban. This chapter is the careful version.

## The minimum viable batch loader

```python
import asyncio
import pandas as pd
from engine.data import Feed

async def fetch_one(feed: Feed, sem: asyncio.Semaphore, symbol: str, start, end) -> pd.DataFrame:
    async with sem:
        return await asyncio.to_thread(feed.bars, symbol, start, end)

async def fetch_many(feed: Feed, symbols: list[str], start, end, concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)
    async with asyncio.TaskGroup() as tg:
        tasks = {sym: tg.create_task(fetch_one(feed, sem, sym, start, end)) for sym in symbols}
    return {sym: t.result() for sym, t in tasks.items()}

results = asyncio.run(fetch_many(feed, ["SPY", "QQQ", "IWM", "TLT", "GLD"], "2024-01-01", "2024-06-01"))
```

The pieces:

- **`asyncio.to_thread`** wraps the *synchronous* `feed.bars` call. Most vendor SDKs (including `yfinance`) are blocking; you don't rewrite them, you just offload them to a thread pool.
- **`asyncio.Semaphore`** bounds concurrency. Without it, `TaskGroup` would fire all N requests at once and you'd be rate-limited.
- **`asyncio.TaskGroup`** (3.11+) is the structured replacement for `asyncio.gather`. If any task fails, the others get cancelled cleanly and you get an `ExceptionGroup`.

For a few thousand symbols, this scales beautifully. Concurrency of 8–16 is typically right for free APIs; for paid APIs check their rate-limit docs and set `Semaphore(N)` to one less than the published per-second limit.

## Retry with exponential backoff and jitter

Free APIs fail. The right response is to retry, but with **exponential backoff** (so the load decays under stress) and **jitter** (so all your retries don't fire at the same moment).

```python
import asyncio, random

async def with_retry(coro_factory, *, attempts: int = 5, base: float = 0.5, factor: float = 2.0):
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = base * (factor ** i) * (1 + random.random())
            await asyncio.sleep(delay)
```

Usage:

```python
df = await with_retry(lambda: fetch_one(feed, sem, "SPY", start, end))
```

The `coro_factory` is a callable that returns a *fresh* coroutine each call — important because you can't `await` the same coroutine twice.

For production, [`tenacity`](https://github.com/jd/tenacity) has the same logic plus declarative conditions ("only retry on `aiohttp.ClientError`"), circuit breakers, and metrics hooks.

## The full pattern

```python
async def fetch_one_with_retry(feed, sem, symbol, start, end):
    async def _go():
        async with sem:
            return await asyncio.to_thread(feed.bars, symbol, start, end)
    return await with_retry(_go)
```

Almost everything else stays the same.

## Don't fire-and-forget

A common mistake:

```python
async def main():
    asyncio.create_task(slow_thing())   # never awaited, may never run
    print("done")
```

The created task might be garbage-collected before it runs. Always either `await` it, save a reference, or — best — use a `TaskGroup`. The course's standard pattern is `TaskGroup` for everything; you can't accidentally lose a task.

## When you do control the HTTP

If you're talking directly to a REST API (not through a vendor SDK), use a real async client and skip the `to_thread` round-trip:

```python
import aiohttp

async def fetch_polygon_bars(session, symbol, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    async with session.get(url, params={"apiKey": API_KEY}) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    return _polygon_to_bars(payload)


async def main(symbols):
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        sem = asyncio.Semaphore(16)
        async with asyncio.TaskGroup() as tg:
            tasks = {s: tg.create_task(_bounded(sem, fetch_polygon_bars, session, s, start, end))
                     for s in symbols}
    return {s: t.result() for s, t in tasks.items()}
```

`aiohttp.ClientTimeout` is essential — without it, a stuck connection hangs forever. The `total` covers the whole request; `connect` covers just the TCP handshake.

## Streaming a result set as it lands

For very large universes (or live workflows), iterating results as they come in beats waiting for all of them:

```python
async def fetch_as_completed(feed, symbols, start, end, concurrency=16):
    sem = asyncio.Semaphore(concurrency)

    async def go(sym):
        async with sem:
            return sym, await asyncio.to_thread(feed.bars, sym, start, end)

    tasks = [asyncio.create_task(go(s)) for s in symbols]
    for fut in asyncio.as_completed(tasks):
        sym, df = await fut
        yield sym, df
```

The consumer can write each result to Parquet immediately, freeing memory before the next arrives.

```python
async def main():
    async for sym, df in fetch_as_completed(feed, symbols, "2024-01-01", "2024-06-01"):
        df.to_parquet(f"data/bars/{sym}.parquet")
```

## Polite rate limiting

A semaphore caps **concurrency**. It does not cap **requests per second** — under a tight semaphore + fast vendor, you can still exceed a per-second limit.

For RPS-bounded APIs, a small async rate-limiter:

```python
import asyncio, time

class RPSLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps
        self.next_ok = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self.next_ok - now)
            self.next_ok = max(now, self.next_ok) + self.min_interval
        if wait:
            await asyncio.sleep(wait)
```

```python
limiter = RPSLimiter(rps=5)

async def fetch_one(feed, sym, start, end):
    await limiter.acquire()
    return await asyncio.to_thread(feed.bars, sym, start, end)
```

The lock serialises the booking of "next allowed time"; the actual sleep happens outside the lock so concurrent requests don't pile up.

## Caching + async: the productive combo

The pattern we use throughout the course:

```python
cache = ParquetCache(YFinanceFeed(), root="data/bars")

async def warm(symbols):
    sem = asyncio.Semaphore(8)
    async with asyncio.TaskGroup() as tg:
        for sym in symbols:
            tg.create_task(asyncio.to_thread(cache.bars, sym, "2020-01-01", "2024-12-31"))

asyncio.run(warm(SP500_TICKERS))
```

First run: pulls everything in parallel, writes to disk, ~5 minutes for the S&P 500.
Subsequent runs: zero network, sub-second.

You only pay the upstream cost once per `(symbol, window)`. The cache is the durable record; the async layer is the throughput burst.

## Common pitfalls

!!! warning "Don't share clients across loops"
    `aiohttp.ClientSession` belongs to the loop that created it. Don't create it at module scope and use it from inside `asyncio.run(...)` calls in different threads. Create the session inside `async def main()`.

!!! warning "Watch for thread-pool starvation"
    `asyncio.to_thread` uses a default pool of `min(32, os.cpu_count() + 4)` threads. If you fire 5,000 `to_thread` calls and your semaphore allows 1,000 concurrent, the pool will be the bottleneck. Either lower the semaphore or set `loop.set_default_executor(ThreadPoolExecutor(max_workers=64))`.

!!! warning "Cancellation requires `await`"
    A `to_thread` call cannot be cancelled mid-flight — the underlying blocking function runs to completion. If your tasks are slow and you need to cancel them, you need a vendor SDK that supports cancellation, or you need to switch to native async (`aiohttp`/`httpx`).

## Bottom line

For batch pulls:

- `asyncio.to_thread` to wrap blocking SDKs.
- `asyncio.Semaphore` for concurrency cap.
- `RPSLimiter` for RPS cap.
- Exponential backoff + jitter on retry.
- `asyncio.TaskGroup` to compose cleanly.
- A `ParquetCache` so you only pay once.

That's a full vendor pipeline in <50 lines of plumbing. The next chapter is where the real money is — the silent ways your historical data lies to you.

Continue to **[Survivorship bias, corporate actions, point-in-time](03-bias-and-corp-actions.md)**.
