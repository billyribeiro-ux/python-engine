# Index, MultiIndex, and the seven sins of pandas

The pandas `Index` is the most under-appreciated object in scientific Python. Treating it as decoration — "the thing on the left" — is the root of more bugs and slowdowns than any other pandas misunderstanding. Treating it as a typed, sorted, hash-indexed, alignment-defining data structure (which is what it actually is) changes how you write data code.

## What an `Index` actually is

A pandas `Index` is:

- a **typed** sequence (always one dtype — `int64`, `datetime64[ns, UTC]`, `string`, `object`, ...);
- an **alignment key**: operations between two Series/DataFrames align by index before computing;
- a **lookup structure**: under the hood, a hash table for fast `.loc[]`;
- **optionally sorted**, which unlocks `searchsorted`, slicing by range, and `merge_asof`.

The cheap mental model: an `Index` is a dictionary key set with an opinion about order and a known dtype.

## Sin #1 — using the default integer index for time series

```python
import pandas as pd
import numpy as np

# wrong — meaningless integer index
df = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

# right — DatetimeIndex
idx = pd.date_range("2024-01-02", periods=100, freq="D", tz="UTC")
df = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100}, index=idx)
```

With a `DatetimeIndex` you get:

- `df.loc["2024-01"]` → all rows in January, by string.
- `df.between_time("09:30", "16:00")` — minute-precision filtering.
- `df.resample("1W").last()` — calendar resampling.
- `df.tz_convert("America/New_York")` — timezone arithmetic, correctly.

Without it, you have a table that looks like a time series but isn't one.

## Sin #2 — losing the index by accident

Many pandas methods quietly reset or change the index. Three to watch:

```python
df.reset_index()           # turns the index into columns
df.merge(other, ...)       # discards the left index by default!
df.groupby("x").mean()     # the group key becomes the new index
```

The `merge` one bites everyone at least once. To keep the left index, use `df.merge(other, ..., left_index=True, ...)`.

## Sin #3 — operating on `df.index` without coercing dtype

```python
# read from CSV — comes back as object dtype
df = pd.read_csv("bars.csv", index_col="timestamp")
df.index.dtype                # object  ← strings!

# fix immediately
df.index = pd.to_datetime(df.index, utc=True)
df.index.dtype                # datetime64[ns, UTC]
```

A string index works but is 10–100× slower than a proper datetime index for everything: lookups, slicing, joins, resampling. **Coerce types at the boundary** (when reading) and treat them as correct everywhere downstream.

## MultiIndex — when one dimension isn't enough

A `MultiIndex` is just an `Index` with multiple levels. For panel data — multiple symbols, each with a time dimension — it's the natural representation.

```python
# A long-format panel
rows = []
for sym in ["SPY", "QQQ", "IWM"]:
    for ts in pd.date_range("2024-01-02", periods=5, freq="D", tz="UTC"):
        rows.append({"symbol": sym, "timestamp": ts, "close": 100 + np.random.randn()})
df = pd.DataFrame(rows).set_index(["symbol", "timestamp"])

df.head(8)
#                                          close
# symbol timestamp
# SPY    2024-01-02 00:00:00+00:00     101.49
#        2024-01-03 00:00:00+00:00     100.27
# ...
```

Now you can:

```python
df.loc["SPY"]                         # one symbol's series
df.loc[("SPY", "2024-01-03")]         # one cell
df.xs("2024-01-03", level="timestamp") # cross-section across symbols
df.unstack("symbol")                   # wide format: timestamp × symbol
df.stack("symbol")                     # back to long
```

The `unstack`/`stack` pair is the cleanest way to convert between long and wide panels.

### Sorting matters

```python
df.index.is_monotonic_increasing      # True/False
df = df.sort_index()                  # always sort after building
```

An unsorted MultiIndex makes `.loc` slicing 10× slower (and prints a `PerformanceWarning`). Always sort after building.

## Sin #4 — chained indexing

```python
# bug magnet — sometimes assigns, sometimes assigns to a copy
df[df.symbol == "SPY"]["close"] = 100

# always works
df.loc[df.symbol == "SPY", "close"] = 100
```

`df[mask]["close"] = ...` is chained indexing. Pandas can't tell whether the intermediate is a view or a copy, so it sometimes writes through and sometimes doesn't. The fix is the single `.loc[]` form with `(row_indexer, col_indexer)`.

**Copy-on-Write** (CoW) — opt-in in pandas 2.x and the **default in pandas 3.0** (shipped) — eliminates the ambiguity by making chained assignment predictably *not* write through (and removing the `SettingWithCopyWarning` era). Even with CoW on, the single `.loc[]` form is still the right way to write a conditional assignment. Pandas 3.0 also defaults string columns to **PyArrow-backed strings**, which are far more memory-efficient than the old `object` dtype — another reason to stop storing text as `object`.

## Sin #5 — `iterrows`

```python
# DON'T
for i, row in df.iterrows():
    df.at[i, "ret"] = row["close"] / row["open"] - 1
```

`iterrows()` constructs a `pd.Series` per row. For a million-row frame that's a million Series allocations.

```python
# DO
df["ret"] = df["close"] / df["open"] - 1
```

Vectorised, one expression, ~1000× faster.

When you really must iterate (event-driven backtests, online algorithms), use `itertuples(index=True, name=None)` — returns plain tuples, no Series ceremony, ~50× faster than `iterrows`.

## Sin #6 — using `apply` for things vectorisation can do

```python
# slow — Python function called per row
df["bp"] = df.apply(lambda r: 1 if r["close"] > r["open"] else -1, axis=1)

# fast — pure vectorised
df["bp"] = np.where(df["close"] > df["open"], 1, -1)
```

`df.apply(axis=1)` runs your callback in Python for every row. Reach for it only when the operation genuinely can't be vectorised, and even then test `numba`-jitted alternatives.

## Sin #7 — storing dates as strings or Python `datetime`

`datetime` objects produce an **object-dtype** column, which kills vectorisation. Always use `pd.Timestamp` / `pd.DatetimeIndex` (which is `datetime64[ns, tz]` under the hood). For dates without times, `pd.PeriodIndex`. For durations, `pd.TimedeltaIndex`.

```python
# bad
df["expiry"] = ["2024-06-21", "2024-07-19", ...]   # object dtype strings

# good
df["expiry"] = pd.to_datetime(["2024-06-21", "2024-07-19"])
```

Time arithmetic is then a single operation:

```python
df["days_to_expiry"] = (df["expiry"] - df.index.normalize()).dt.days
```

## `pd.NA` vs `np.nan`

Pandas 1.0+ ships `pd.NA`, an explicit missing-value sentinel that works for *all* dtypes including integers and booleans. `np.nan` only works for floats (and silently upcasts your int columns to float).

```python
s = pd.Series([1, 2, None, 4])              # dtype object → painful
s = pd.Series([1, 2, None, 4], dtype="Int64")  # nullable int — uses pd.NA, dtype preserved
```

For new code, prefer the **nullable** dtypes: `Int64`, `Float64`, `boolean`, `string`. They've been there since 1.0 and are stable.

## `.loc`, `.iloc`, `.at`, `.iat`

| | label | position |
|---|---|---|
| set/get many rows or slices | `.loc[]` | `.iloc[]` |
| single scalar | `.at[]` | `.iat[]` |

`.at` and `.iat` are 10× faster than `.loc`/`.iloc` for single-cell access — useful in tight loops you couldn't avoid.

## A real-world pattern: aligning a signal to bars

You have a `signals` DataFrame indexed by date and a `bars` DataFrame indexed by minute. To put the daily signal next to each intraday bar:

```python
aligned = pd.merge_asof(
    bars.sort_index(),
    signals.sort_index(),
    left_index=True, right_index=True,
    direction="backward",        # take the latest signal at or before each bar
)
```

`merge_asof` is the right tool. We give it its own chapter next.

## A real-world pattern: wide panel of returns

```python
# bars: long-format MultiIndex (symbol, timestamp), one 'close' column
prices = bars["close"].unstack(level="symbol")         # timestamp × symbol
returns = prices.pct_change()                          # vectorised across all symbols
correlations = returns.tail(252).corr()                # 252-day rolling correlation matrix
```

Three lines, no loops, scales to thousands of symbols. The trick is the long↔wide flip: long is right for storage, wide is right for cross-sectional math.

## When to leave pandas for polars

A short answer: when any of these hurts.

- Datasets above ~10 GB on a single machine — pandas balloons memory; polars stays compact.
- Tight inner loops in a backtest — polars expressions parallelise across cores by default.
- Strict type discipline — pandas's coercion-happy defaults bite at scale; polars refuses to guess.

We cover polars in detail in chapter 5 of this module. For most of the course's research-style code, pandas is the right tool and you'll be fine.

Continue to **[Joins, `merge_asof`, and tick alignment](02-joins-asof.md)**.
