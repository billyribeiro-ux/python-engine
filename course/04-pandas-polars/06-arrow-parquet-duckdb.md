# Arrow, Parquet, DuckDB — the fast triangle

Three pieces of technology, working together, define modern data engineering in Python:

- **Apache Arrow** — a columnar in-memory format. Languages and tools share data without serialisation.
- **Apache Parquet** — a columnar on-disk format. Designed for analytical reads of subsets of huge datasets.
- **DuckDB** — an embedded SQL engine that reads Parquet at memory bandwidth and feels like SQLite for analytics.

Knowing these three changes how you think about storage. The point of this chapter is to put them in your head with enough context that you reach for them naturally in Module 5.

## Arrow — the lingua franca

Arrow specifies a layout for columnar data in memory. The same bytes can be read by pandas, polars, DuckDB, PyTorch, Apache Spark, and BigQuery without translation. That's why you can do:

```python
import polars as pl
import duckdb

df = pl.read_parquet("bars.parquet")
result = duckdb.sql("SELECT symbol, AVG(close) FROM df GROUP BY symbol").pl()
```

`duckdb` sees the polars frame, runs a SQL query against its Arrow buffer, and gives you back another polars frame. No copies. No "convert to dict, JSON-serialise, send" antipatterns. Bytes in memory, queries on top.

Pandas converts to Arrow with `df.to_arrow()` since 2.0. NumPy interops via `np.asarray(arrow_array)`. PyTorch via `torch.from_numpy(np.asarray(arrow_array))`.

## Parquet — the canonical on-disk format

Parquet stores data in a **columnar** layout, with **row groups** (chunks of rows), and **predicate / column pushdown** so that reading "just the close column for SPY between 2023 and 2024" reads a tiny fraction of the file.

Writing:

```python
import polars as pl
df.write_parquet("bars/spy.parquet", compression="zstd", row_group_size=100_000)
```

Reading lazily:

```python
lf = pl.scan_parquet("bars/spy.parquet")  # nothing loaded yet
result = lf.filter(pl.col("ts") >= "2024").select(["ts", "close"]).collect()
# Only the timestamps after 2024, only the close column — actually read from disk
```

For a multi-GB file, this is the difference between waiting two seconds and waiting two minutes.

### Partitioning

For time-series, partition by date or by month:

```
bars/
  date=2024-01-02/data.parquet
  date=2024-01-03/data.parquet
  ...
```

```python
pl.scan_parquet("bars/**/*.parquet").filter(pl.col("date") == "2024-01-02").collect()
```

The scanner only touches the matching directory. For multi-year datasets, partition by year + month; finer partitioning increases file count and Parquet metadata overhead.

For options chains, partition by `expiry`:

```
chains/
  expiry=2024-06-21/data.parquet
  expiry=2024-07-19/data.parquet
```

A scanner that asks "all calls expiring this Friday" reads one file.

### Compression

`zstd` is the right default in 2024: fast to write, fast to read, smaller than `snappy` or `gzip`. Pass `compression="zstd"`.

## DuckDB — SQL on top of everything

DuckDB is a single-file embedded database with a query planner that rivals Spark for analytical workloads. Three killer properties:

1. **Reads Parquet natively** without import.
2. **Reads Arrow / pandas / polars frames in-place** without copying.
3. **Supports window functions, CTEs, and most analytical SQL** out of the box.

```python
import duckdb

# Query a directory of Parquet files
result = duckdb.sql("""
    SELECT symbol, AVG(close) AS avg_close, STDDEV(close) AS vol
    FROM read_parquet('bars/**/*.parquet')
    WHERE ts >= '2024-01-01'
    GROUP BY symbol
    ORDER BY vol DESC
    LIMIT 10
""").pl()                                # back into polars
```

You did not load a single byte into Python before the query. DuckDB pushed the filters and aggregations down into the Parquet scan and returned a polars frame.

### Window functions

The kind of thing pandas / polars can do but SQL says more clearly:

```python
duckdb.sql("""
    SELECT
        symbol, ts, close,
        AVG(close) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
        (close / LAG(close) OVER (PARTITION BY symbol ORDER BY ts) - 1) AS ret
    FROM read_parquet('bars/**/*.parquet')
""").pl()
```

For one-off analysis, SQL like this is often the most concise expression. For a permanent pipeline, the same logic in polars is more maintainable.

### When to reach for DuckDB

- **Ad-hoc analytics** across Parquet directories.
- **Joining datasets that don't fit in memory** — DuckDB will spill to disk if needed.
- **Sharing a query language** with colleagues who know SQL but not polars.
- **Building a small data mart** on a single machine before you scale up.

## A real trading-data store: layered design

Most quant data pipelines end up with a layered Parquet store:

```
data/
  bars/
    interval=1d/
      year=2024/
        month=01/
          part-000.parquet
  chains/
    expiry=2024-06-21/
      data.parquet
  fundamentals/
    asof=2024-Q1/
      data.parquet
  universe/
    asof=2024-01-31/
      data.parquet
```

The course's Module 5 builds this. The key designs:

- **One Parquet file per partition**. Multiple files per partition makes scans slower and adds file-system overhead.
- **Row groups sized for query patterns**. Daily bars → 50,000 row groups; tick → 500,000+.
- **A small metadata index** (a tiny SQLite or Parquet "table of contents") that knows what's been loaded, when, and from where.
- **Asof versioning** for fundamentals — every snapshot has its own partition so backtests can ask "what did we believe in March?" without contamination.

## Compatibility table

| | reads pandas | reads polars | reads Arrow | reads Parquet | writes Parquet |
|---|---|---|---|---|---|
| pandas | ✓ | via `to_arrow` | ✓ (since 2.0) | ✓ | ✓ |
| polars | ✓ | ✓ | ✓ | ✓ | ✓ |
| DuckDB | ✓ | ✓ | ✓ | ✓ | ✓ |
| Arrow | ✓ | ✓ | ✓ | ✓ | ✓ |

Everything talks to everything. Choose tools for their strengths, move data between them for free.

## A trap: Parquet schema evolution

You wrote a Parquet file with columns `[ts, open, close]`. Six months later you add `vwap`. If the reader doesn't handle schema evolution, the union scan errors.

- **polars** handles missing columns gracefully (fills NaN).
- **DuckDB** can read heterogeneous schemas if you turn on `union_by_name=true` in the scan.
- **PyArrow** requires explicit schema unification.

In practice: write a small migrator that backfills new columns into old partitions when you add them. Or use a schema-on-read approach with `union_by_name=true`. Either way, **decide your evolution story before you have 10,000 partitions**.

## A trap: too many tiny Parquet files

Each Parquet file has metadata overhead. A million tiny files is much slower to scan than 100 right-sized ones, because the planner spends all its time opening files instead of reading data.

Target file size: **64–512 MB**. If your daily partitions are smaller, group by month or quarter instead.

## Bottom line

For the data engineering work that starts in Module 5:

- **Storage** → Parquet, partitioned by date (and expiry, for options chains), `zstd` compression, row groups sized for your query pattern.
- **In-memory format** → Arrow. It's the air everything breathes.
- **Query layer** → polars expressions for pipelines; DuckDB SQL for ad-hoc analytics.
- **Analytics / ML in-memory** → pandas + scikit-learn / PyTorch, hopping in via `.to_pandas()` / `np.asarray(...)`.

## End of Module 4

You now have the data layer skills. The next module turns these into **a real cache and store** for market data — the universal feed adapter from Module 0 backed by a Parquet store, asyncio-driven downloads, and an options-chain layout that doesn't melt down at scale.

Continue to **[Module 5 — Data Engineering for Markets](../05-data-engineering/index.md)**.
