# Polars from a pandas user's perspective

Polars is the credible challenger to pandas. It's written in Rust, uses Apache Arrow as its memory format, parallelises across cores by default, and ships a lazy query engine that optimises your code before running it. For panel data at scale, it's commonly 5–30× faster than pandas with a fraction of the memory.

This chapter is **not** "polars from scratch". It's how a pandas-fluent reader gets productive in polars in an afternoon.

## Three things that are different

1. **No index.** A polars DataFrame is just columns. No `.loc` mysteries. Sort or filter explicitly.
2. **Expressions, not Series.** You operate on columns symbolically: `pl.col("close").mean()` is a description of work, not the value.
3. **Lazy frames.** `pl.scan_parquet(...)` returns a query plan; nothing runs until `.collect()`. The planner reorders + prunes for you.

## Hello, polars

```python
import polars as pl
import numpy as np

df = pl.DataFrame({
    "symbol": ["SPY", "SPY", "QQQ", "QQQ"],
    "ts":     ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
    "close":  [475.0, 480.0, 410.0, 412.0],
})
df = df.with_columns(pl.col("ts").str.to_datetime("%Y-%m-%d"))
```

`with_columns` is the polars "add or replace a column" verb. It takes one or more expressions.

## The expression API

Pandas:

```python
df["ret"] = df["close"] / df.groupby("symbol")["close"].shift(1) - 1
```

Polars:

```python
df = df.sort(["symbol", "ts"]).with_columns(
    ret = pl.col("close") / pl.col("close").shift(1).over("symbol") - 1
)
```

Three things to notice:

- The whole expression is built **symbolically** with `pl.col`. No groupby boilerplate to assemble the result frame.
- `.over("symbol")` is polars's per-group window. It works on *any* expression — rolling, shift, cumsum, rank, custom.
- The frame must be sorted first (polars warns if it isn't). Sorting is explicit and cheap because it's parallel.

## Selecting, filtering, mutating

```python
# select columns
df.select(["symbol", "close"])
df.select(pl.col("close").alias("price"))

# filter rows
df.filter(pl.col("close") > 400)
df.filter((pl.col("symbol") == "SPY") & (pl.col("close") > 470))

# rename, drop, sort
df.rename({"close": "px"})
df.drop(["volume"])
df.sort("ts")

# pivot
df.pivot(on="symbol", index="ts", values="close")
```

The vocabulary is mostly self-explanatory if you've used pandas. The biggest mental shift is **always thinking in expressions** rather than columns-as-data.

## GroupBy without the pandas gotchas

```python
df.group_by("symbol").agg([
    pl.col("close").mean().alias("avg_close"),
    pl.col("close").std().alias("vol"),
    pl.col("close").count().alias("n"),
])
```

Several reductions in one call. Each reducer is an expression. No `as_index=False` ceremony.

For "per-symbol but keep all rows" (pandas's `transform`), use `.over("symbol")`:

```python
df.with_columns([
    pl.col("close").mean().over("symbol").alias("symbol_avg"),
    pl.col("close").rank().over("symbol").alias("rank_in_symbol"),
])
```

This is the operation pandas's `groupby.transform` does — but polars parallelises across groups automatically.

## Rolling, properly parallel

```python
df.sort(["symbol", "ts"]).with_columns([
    pl.col("close").rolling_mean(window_size=20).over("symbol").alias("ma20"),
    pl.col("close").ewm_mean(span=20).over("symbol").alias("ema20"),
    pl.col("close").rolling_std(window_size=20).over("symbol").alias("vol20"),
])
```

For a panel of thousands of symbols, this is dramatically faster than `df.groupby("symbol").rolling(20).mean()` in pandas, because polars uses every CPU core.

### Time-based rolling

```python
df.with_columns(
    pl.col("close").rolling_mean_by("ts", window_size="7d").over("symbol").alias("ma7d")
)
```

`window_size` accepts duration strings (`"5m"`, `"1h"`, `"7d"`). For irregularly sampled data, this is the right tool.

## Lazy frames — where the magic happens

```python
lf = pl.scan_parquet("data/bars/*.parquet")

result = (
    lf.filter(pl.col("symbol").is_in(["SPY", "QQQ", "IWM"]))
      .filter(pl.col("ts") >= pl.datetime(2024, 1, 1))
      .with_columns(pl.col("close").pct_change().over("symbol").alias("ret"))
      .group_by("symbol")
      .agg([pl.col("ret").mean(), pl.col("ret").std()])
      .collect()                              # the query runs here
)
```

The planner:

- pushes filters into the Parquet scan, so only matching row groups are read,
- prunes columns to only what you reference,
- parallelises everything that can be parallelised,
- streams when possible so the working set never exceeds memory.

For a 50 GB Parquet dataset, this is the difference between "OOM in seconds" and "results in a minute on a laptop".

## `scan_parquet` vs `read_parquet`

- `pl.read_parquet(path)` — eager, loads everything.
- `pl.scan_parquet(path)` — lazy, returns a query plan.

Always start with `scan_` when working with files on disk. Collect at the end.

## Interop with pandas and Arrow

```python
df.to_pandas()           # zero-copy when types align
pl.from_pandas(pdf)      # the other direction
df.to_arrow()            # zero-copy to PyArrow Table
pl.from_arrow(table)     # zero-copy from PyArrow
```

Arrow is the shared memory format. PyTorch and JAX speak Arrow. DuckDB speaks Arrow. Polars is Arrow-native. You can move data between them with no serialisation cost.

## When pandas is still the right tool

- **The data is small** (<1 GB) and you want the rich pandas ecosystem (statsmodels, scikit-learn, etc.).
- **You're doing ad-hoc analysis** in a notebook where `df.loc[...]` is faster to type.
- **You need a library that returns pandas** (most ML libraries still do).

A common pattern: use polars for data engineering (loading, joining, feature engineering on huge panels), `.to_pandas()` at the boundary into a model, train, get predictions back. The bottleneck moves out of data wrangling and into modelling, where it should be.

## A worked example: per-symbol z-scored returns from a 10M-row panel

```python
import polars as pl

result = (
    pl.scan_parquet("data/bars/*.parquet")
      .sort(["symbol", "ts"])
      .with_columns([
          pl.col("close").pct_change().over("symbol").alias("ret"),
      ])
      .with_columns([
          ((pl.col("ret") - pl.col("ret").mean().over("symbol")) /
            pl.col("ret").std().over("symbol")).alias("z"),
      ])
      .filter(pl.col("z").abs() > 3)              # extreme moves
      .collect()
)
```

The same computation in pandas typically uses 5–10× more peak memory and runs 10× slower.

## Where polars is *worse* than pandas

- **Statistical / scientific tooling.** statsmodels, scipy.stats integration is via `.to_pandas()` or `.to_numpy()`.
- **Time-aware indexing.** No `df.loc["2024-01"]` shorthand. You write the filter.
- **Plotting.** Polars has `df.plot.line()` but pandas's matplotlib integration is more mature.
- **Stability of the API.** Polars is still pre-2.0 at the time of writing; some methods get renamed between minor releases. Pin your version.

## A polars version of the bollinger bands

```python
df.sort(["symbol", "ts"]).with_columns([
    pl.col("close").rolling_mean(window_size=20).over("symbol").alias("mid"),
    pl.col("close").rolling_std(window_size=20).over("symbol").alias("std20"),
]).with_columns([
    (pl.col("mid") + 2 * pl.col("std20")).alias("upper"),
    (pl.col("mid") - 2 * pl.col("std20")).alias("lower"),
    ((pl.col("close") - pl.col("mid")) / pl.col("std20")).alias("z"),
])
```

Multi-step `with_columns` chains are how polars computations are normally written. Each block is a transformation on the previous.

## Bottom line

You don't have to choose. Use pandas for analysis and small data. Use polars for the data engineering layer underneath. Move data between them with `.to_pandas()` / `pl.from_pandas()` — it's free.

For Module 5 (Data engineering), polars is the default for anything bigger than a few hundred thousand rows. We'll set up a Parquet store and a query layer that lives at the polars / Arrow / DuckDB intersection.

Continue to **[Arrow, Parquet, DuckDB — the fast triangle](06-arrow-parquet-duckdb.md)**.
