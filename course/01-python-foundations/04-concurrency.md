# Async, threads, processes, and the GIL

Concurrency in Python is one of the most-misunderstood topics in the language, mostly because the right answer depends on what you're actually doing. This chapter gives you a clear decision tree, the truth about the GIL (including the free-threaded build), and patterns that will see you through real production code.

## The decision tree

> **CPU-bound work** (numeric loops, model training, simulation) → **multiprocessing** (or NumPy / Numba / native code that releases the GIL, or JAX/PyTorch on a GPU).
>
> **I/O-bound work** (HTTP, websockets, database, file system) → **`asyncio`**.
>
> **Blocking I/O you can't make async** (a synchronous SDK you don't control) → **threads** (a thread pool, run from inside asyncio with `asyncio.to_thread`).
>
> **Single-machine parallelism over data** → **`concurrent.futures.ProcessPoolExecutor`** or **`joblib`** or **`ray`**.

That's it. The rest of the chapter is the *why* and the patterns.

## The GIL, accurately

The Global Interpreter Lock is a mutex inside CPython that ensures only one thread executes Python bytecode at a time. The lock is held during the execution of bytecode and released:

- around long-running C extensions that opt out (NumPy, lz4, much of `os` and `socket`),
- at periodic intervals,
- when the thread blocks on I/O.

What this means in practice:

| Workload | Threads help? |
|---|---|
| Pure-Python loops (for `i in range(N): a += i*i`) | **No.** Single-core wall time, plus context-switch overhead. |
| NumPy linear algebra, FFT, etc. | **Yes**, if the operation is long enough that the GIL release amortises the thread overhead. |
| HTTP / disk I/O | **Yes**, threads block on the kernel and other threads run. |
| Subprocess / file system | **Yes**, same reason. |

### Free-threaded CPython (3.13+)

Python 3.13 ships an experimental **free-threaded** build (PEP 703) that removes the GIL. It's opt-in (`--disable-gil` at compile time) and slower in single-threaded code today, but it's the future. Once it ships fast and stable, the right answer for many CPU-bound workloads becomes "use threads". Until then, **multiprocessing remains the safe bet for CPU parallelism**.

You can detect it at runtime:

```python
import sys
sys.flags.no_gil  # True on a free-threaded build
```

## `concurrent.futures` — the boring, correct API

For the 80% case, this is what you want:

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def fetch(symbol):
    ...

symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# I/O-bound (HTTP)
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(fetch, symbols))

# CPU-bound (heavy NumPy / a tight Python loop)
with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(heavy_calc, payloads))
```

Three properties to internalise:

1. **`map` preserves order.** Result `i` came from input `i`.
2. **`map` raises lazily.** Exceptions surface when you iterate the result, not before.
3. **Workers reuse interpreter state** in `ProcessPoolExecutor`. The first call to each worker pays an import cost; subsequent calls are cheap. Pin big globals (a model, a config) at module scope so they're imported once.

### Submit-and-as-completed

When workloads have varying durations, you usually want results as they finish:

```python
from concurrent.futures import as_completed

with ProcessPoolExecutor() as pool:
    futures = {pool.submit(backtest, name): name for name in strategies}
    for fut in as_completed(futures):
        name = futures[fut]
        try:
            sharpe = fut.result()
            print(f"{name}: {sharpe:.2f}")
        except Exception as exc:
            print(f"{name} failed: {exc!r}")
```

## `asyncio` — for I/O concurrency

`asyncio` lets a single thread juggle thousands of in-flight I/O operations. The cost is a programming model where every I/O-doing function is "coloured" `async`.

### The minimum you need to know

```python
import asyncio, aiohttp

async def fetch_json(session, url):
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()

async def main():
    async with aiohttp.ClientSession() as s:
        symbols = ["SPY", "QQQ", "IWM"]
        urls = [f"https://example.com/quote/{sym}" for sym in symbols]
        results = await asyncio.gather(*(fetch_json(s, u) for u in urls))
        return dict(zip(symbols, results, strict=True))

quotes = asyncio.run(main())
```

`asyncio.gather` runs the awaitables concurrently. Sequential time would be `sum(latencies)`; concurrent time is `max(latencies)`.

### Structured concurrency with `TaskGroup` (3.11+)

`asyncio.gather` has rough edges around cancellation and error propagation. The modern replacement is `TaskGroup`:

```python
import asyncio

async def stream_ticks(symbol):
    ...

async def stream_quotes(symbol):
    ...

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(stream_ticks("SPY"))
        tg.create_task(stream_quotes("SPY"))
        tg.create_task(stream_ticks("QQQ"))
    # exiting the TaskGroup awaits all tasks; if any raised, the rest
    # are cancelled and an ExceptionGroup is raised here.

asyncio.run(main())
```

Three guarantees from `TaskGroup`:

- **All tasks finish or all get cancelled.** No orphaned background tasks leaking forever.
- **Exceptions aggregate into an `ExceptionGroup`.** You can `except*` on it to handle subgroups.
- **Cancellation is propagated correctly.** A `CancelledError` from one task is not swallowed.

Always prefer `TaskGroup` over hand-rolled `gather` + `try/except`.

### Mixing async with blocking code

If you have a blocking function you can't replace (a vendor SDK that returns synchronously), wrap it:

```python
async def get_bars(symbol):
    return await asyncio.to_thread(blocking_feed.bars, symbol, "2024-01-01", "2024-06-01")
```

`asyncio.to_thread` runs the call on a thread pool and `await`s the result. Don't do this for tight CPU loops — you'll just bounce between threads. Use it for blocking I/O.

### `aiohttp` for HTTP, `httpx` if you want both

`aiohttp` is the standard async HTTP client. `httpx` has the same API for both sync and async and is easier when you have mixed needs:

```python
import httpx

# sync
r = httpx.get(url)

# async
async with httpx.AsyncClient() as c:
    r = await c.get(url)
```

For market-data feeds (HTTP/WebSocket), `httpx` + `websockets` is a clean pair.

## A real production pattern: bounded concurrency

You want concurrency, but you don't want to DoS the vendor (or get yourself rate-limited). The pattern is a **semaphore**:

```python
import asyncio, aiohttp

async def fetch_one(sem, session, symbol):
    async with sem:                                # at most N concurrent
        async with session.get(f"https://api.example.com/{symbol}") as r:
            r.raise_for_status()
            return await r.json()

async def fetch_all(symbols, concurrency=8):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch_one(sem, session, s)) for s in symbols]
    return [t.result() for t in tasks]

results = asyncio.run(fetch_all(big_symbol_list, concurrency=8))
```

This bounds concurrency to 8, retries cleanly with `TaskGroup`, and parallelises across hundreds of symbols. The same pattern works for any rate-limited API.

## Retry with backoff (and the library you should use)

Roll-your-own retries usually look like this:

```python
import asyncio, random

async def with_retry(coro_factory, attempts=5, base=0.5):
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = base * (2 ** i) * (1 + random.random())   # exponential + jitter
            await asyncio.sleep(delay)
```

For production, use [`tenacity`](https://github.com/jd/tenacity) — it has the same logic, plus declarative retry conditions, circuit breakers, and good metrics integration.

## Multiprocessing in earnest

`ProcessPoolExecutor` is fine for fire-and-forget. For long-lived workers with state (e.g. per-symbol model servers), use `multiprocessing.Process` directly with `multiprocessing.Queue`:

```python
import multiprocessing as mp

def worker(in_q, out_q):
    model = load_model()           # expensive — pay it once per worker
    while True:
        x = in_q.get()
        if x is None: break
        out_q.put(model.predict(x))

if __name__ == "__main__":         # the spawn-safe guard, REQUIRED on macOS/Windows
    in_q, out_q = mp.Queue(), mp.Queue()
    procs = [mp.Process(target=worker, args=(in_q, out_q)) for _ in range(4)]
    for p in procs: p.start()
    for x in dataset: in_q.put(x)
    for _ in procs: in_q.put(None)
    for p in procs: p.join()
```

!!! warning "The `if __name__ == '__main__'` guard"
    On macOS (since Python 3.8) and Windows, multiprocessing uses *spawn*. The child re-imports your module. Without the guard, the child runs the parent's setup code recursively — fork bomb. Always include it.

### Sharing big data: `multiprocessing.shared_memory`

Process workers can't share memory by default. Copying a 5 GB array to each worker is wasteful. `shared_memory` (3.8+) is the answer:

```python
import numpy as np
from multiprocessing import shared_memory

# parent
arr = np.zeros(10_000_000, dtype=np.float64)
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
backed = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
backed[:] = arr[:]

# children get `shm.name` and reattach
```

Alternative: store the data in Parquet/Arrow on a tmpfs and let each worker memory-map it. `pyarrow` and `polars` handle that for you.

## Async pitfalls

!!! danger "Sync code in an async function blocks the loop"
    A `time.sleep(1)` in an async function freezes the entire event loop. Use `await asyncio.sleep(1)`. Same for `requests.get()` — use an async client.

!!! warning "Don't fire-and-forget"
    `asyncio.create_task(coro())` without keeping the reference can have the task garbage-collected before it runs. Either await it, store it, or use a `TaskGroup`.

!!! warning "Cancellation is cooperative"
    A task being "cancelled" means a `CancelledError` is injected at its next `await`. If your code has no `await`s (a tight CPU loop), it can't be cancelled. Yield periodically with `await asyncio.sleep(0)`.

## When in doubt, measure

Before parallelising, profile. Most "I need threads" problems are really "I have an O(n²) algorithm". The next chapter on profiling will save you weeks of fictitious parallelism.

Continue to **[Typing for engineers](05-typing.md)**.
