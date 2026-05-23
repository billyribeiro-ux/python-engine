# A partitioned Parquet cache

If you do any serious work against a free or rate-limited data source, the second time you re-run a notebook you will hit a wall. The cache is what unblocks you — and once you have one, you'll wonder how you ever lived without it.

The course ships a small Parquet cache in `engine.data.cache.ParquetCache`. This chapter walks through what it does and why, and shows you how to use it as a model for your own data stores.

## The shape

```python
from engine.data import YFinanceFeed
from engine.data.cache import ParquetCache

cache = ParquetCache(YFinanceFeed(), root="data/bars")

# first call: pulls SPY from Yahoo, writes to disk
bars = cache.bars("SPY", "2024-01-02", "2024-03-31")

# second call: serves from disk, no network hit
bars = cache.bars("SPY", "2024-01-02", "2024-03-31")
```

The wrapper satisfies the same `Feed` Protocol as the underlying adapter. Anything in the course that takes a `Feed` (strategies, scanners, backtests) takes the cache transparently.

## On-disk layout

```
data/bars/
  interval=1d/
    symbol=SPY/
      _manifest.json
      month=2024-01.parquet
      month=2024-02.parquet
      month=2024-03.parquet
    symbol=QQQ/
      ...
  interval=1h/
    symbol=SPY/
      ...
```

Three reasons for this layout:

1. **One file per month** — small enough to write quickly, large enough that file overhead doesn't dominate.
2. **Partitioned by interval + symbol** — a query like "SPY daily bars" reads exactly one directory.
3. **A `_manifest.json` per (symbol, interval)** — remembers which date spans were requested, so we never re-fetch a window we've already covered.

The manifest is the small trick. Without it, "give me SPY for 2024-01 to 2024-03" and "give me SPY for 2024-02 to 2024-04" would refetch February. With it, the cache realises the second window's overlap is already on disk and only requests the missing tail.

## The full implementation in one screen

```python
class ParquetCache:
    name = "parquet-cache"

    def __init__(self, upstream: Feed, root: str | Path):
        self.upstream = upstream
        self.root = Path(root)

    def bars(self, symbol, start, end, interval="1d") -> pd.DataFrame:
        symbol = symbol.upper()
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts   = pd.Timestamp(end, tz="UTC")

        manifest = _Manifest.load(self._manifest_path(symbol, interval))

        if not manifest.covers(start_ts, end_ts):
            fetched = _normalise_bars(self.upstream.bars(symbol, start_ts, end_ts, interval))
            if not fetched.empty:
                for month, slice_ in fetched.groupby(_month_key_series(fetched.index)):
                    self._write_month(self._month_path(symbol, interval, month), slice_)
                manifest.add(start_ts, end_ts)

        return self._read_window(symbol, interval, start_ts, end_ts)
```

Less than 30 lines for a real cache. The rest of the file (`_write_month`, `_read_window`, `_Manifest`) is straight file plumbing. Read the full source at `engine/data/cache.py`.

## Design choices worth knowing

### `zstd` for compression

```python
df.to_parquet(path, compression="zstd")
```

`zstd` is the right default in 2024: faster than `gzip`, smaller than `snappy`, supported everywhere. For OHLCV data, expect 5–10× compression.

### Append-with-dedup on write

```python
if path.exists():
    existing = pd.read_parquet(path)
    df = pd.concat([existing, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
df.to_parquet(path)
```

When two requests touch the same month, we read the existing partition, concat, dedup (keeping the *latest* row per timestamp, since vendor restatements happen), and rewrite. This is fast for monthly partitions; it would be too slow for tick-level data.

For tick data, the right pattern is **immutable daily files with batched compaction** — write each session as a new file, then compact monthly in the background.

### The manifest avoids a category of bugs

Without it:

- The cache stores rows. It doesn't know whether absence of rows for a given date means "we haven't asked" or "the market was closed."
- A request for a long-closed holiday window would re-fetch every time, because the cache would see "no rows" and assume "data missing."

With the manifest:

- The cache knows the span was requested. If no rows came back, that's the truth — the market was closed.

This is the single highest-leverage piece of cache design.

## Using it across the course

Almost every example from Module 8 onward looks like:

```python
from engine.data import YFinanceFeed, ParquetCache

feed = ParquetCache(YFinanceFeed(), root="data/bars")
spy  = feed.bars("SPY", "2020-01-01", "2024-12-31")
```

The first run takes a minute (Yahoo rate-limit included). Every subsequent run is sub-second.

## When to graduate from this cache

The `ParquetCache` above is fit for:

- Hundreds of symbols.
- Daily / hourly / 5m bars.
- A single research process at a time.

It is *not* fit for:

- Tick-level data at scale (write amplification on rewrite is too high).
- Multi-process concurrent writes (no file locking).
- Distributed setups.

For those, the right answer is a **proper time-series database** (QuestDB, kdb+/q, TimescaleDB, Clickhouse) or a **columnar lake on object storage** with a compaction job. Module 21 (Deployment) sketches both shapes.

## A useful pattern: cache as a drop-in `Feed`

Because `ParquetCache` implements the `Feed` Protocol, you can chain caches:

```python
fast_cache = ParquetCache(
    upstream=ParquetCache(YFinanceFeed(), root="data/cold"),
    root="data/hot",
)
```

The outer cache is a fast scratch store you can wipe; the inner is the permanent record. Strategies see "a feed"; the layers are an implementation detail.

## Testing the cache without hitting the network

```python
class StubFeed:
    name = "stub"
    def __init__(self, payload): self.payload = payload
    def bars(self, *_args, **_kw): return self.payload
    def option_chain(self, *_a, **_kw): raise NotImplementedError

stub = StubFeed(SPY_FIXTURE)
cache = ParquetCache(stub, tmp_path)
result = cache.bars("SPY", "2024-01-02", "2024-01-31")
assert (result.columns.tolist() == ["open","high","low","close","volume"])
```

Hand the cache a stub and you have a fast, deterministic test. The `Feed` Protocol makes this trivial — no mocking framework needed.

Continue to **[Async batch loaders with backoff](02-async-batch.md)**.
