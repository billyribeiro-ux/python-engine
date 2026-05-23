# GroupBy: `apply` vs `agg` vs `transform`

GroupBy is where pandas earns its keep — split-apply-combine over panel data is the central operation in quant work. It's also where most pandas slowness lives. The fix is understanding which of `agg`, `transform`, and `apply` you actually want.

## The three operations, in one image

| Method | Per-group return shape | Output shape | When to use |
|---|---|---|---|
| **`agg`** | scalar (or dict of scalars) | one row per group | reductions: mean, max, sum, custom |
| **`transform`** | same shape as the group | same shape as the input | rolling stats, z-scores, ranks within group |
| **`apply`** | anything | varies | when neither of the others fits |

The rule of thumb: try `agg` first, then `transform`, fall back to `apply` only if you must. `apply` is the slow one because it round-trips through Python for each group.

## `agg` — the reduction

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "symbol": np.repeat(["SPY", "QQQ", "IWM"], 4),
    "return": np.random.randn(12) * 0.01,
})

df.groupby("symbol")["return"].agg("mean")
df.groupby("symbol")["return"].agg(["mean", "std", "max"])
df.groupby("symbol")["return"].agg(
    mean_ret="mean",
    vol=lambda s: s.std() * np.sqrt(252),
    sharpe=lambda s: s.mean() / s.std() * np.sqrt(252),
)
```

The third form (named aggregation) is the most powerful and the most readable. You name each output column, you pass the reducer. Everything is one pass.

For multi-column reductions:

```python
df.groupby("symbol").agg(
    avg_return=("return", "mean"),
    vol=("return", "std"),
    n=("return", "size"),
)
```

The `(col, reducer)` tuple form lets you reduce different columns differently in the same call.

## `transform` — same shape out

`transform` is the operation you reach for when you want "compute per group, paste back aligned with the original index." Typical uses:

```python
# Z-score each return relative to its symbol's mean and std
df["z"] = (df["return"] - df.groupby("symbol")["return"].transform("mean")) \
        / df.groupby("symbol")["return"].transform("std")

# Demean per group in one shot
df["demeaned"] = df["return"] - df.groupby("symbol")["return"].transform("mean")

# Cumulative sum per symbol (each symbol's path)
df["cum"] = df.groupby("symbol")["return"].transform("cumsum")
```

The output has the same length as the input frame, with each value replaced by the group statistic. No alignment headaches.

### `transform` with a callable

```python
df["rank"] = df.groupby("symbol")["return"].transform(lambda s: s.rank(pct=True))
```

For built-in operations, pass the string name (`"rank"`, `"cumsum"`, `"diff"`); pandas dispatches to a fast path. Lambdas drop to Python and are slower.

## `apply` — the escape hatch

`apply` is what you use when the per-group operation returns *anything* — a DataFrame, a different-length Series, a scalar, a dict.

```python
def top_two(group: pd.DataFrame) -> pd.DataFrame:
    return group.nlargest(2, "volume")

df.groupby("symbol").apply(top_two)
```

This is fine for low cardinality (a few hundred groups). For high cardinality (thousands of symbols), the Python overhead of calling your function once per group dominates. Watch for it in profiles.

## The performance hierarchy

For a typical "rolling z-score per symbol on a panel of 5,000 symbols × 5 years of daily bars":

| Approach | Wall time |
|---|---|
| `df.groupby("symbol")["return"].apply(my_zscore_fn)` | ~30 s |
| `df.groupby("symbol")["return"].transform("mean")` + arithmetic | ~1.5 s |
| Reshape to wide (symbol × time) and vectorise across symbols | ~0.3 s |
| Polars `.over("symbol")` expression | ~0.15 s |

The last two are 100–200× faster than the first, with the same answer. **Reshape-then-vectorise** is the most underused pandas pattern; polars's expression engine makes it the default.

## A worked example: cross-sectional rank per day

You have a long-format panel with `(symbol, date)` and a `signal` column. You want the daily cross-sectional rank.

```python
df["rank_today"] = df.groupby("date")["signal"].rank(pct=True) - 0.5
```

That's it. `rank` is a built-in groupby method; pandas dispatches to a Cython loop; `pct=True` normalises to [0, 1]; the subtraction centres on zero.

Compare to the `apply` version:

```python
df["rank_today"] = df.groupby("date")["signal"].apply(lambda s: s.rank(pct=True)) - 0.5
```

Same answer, ~50× slower on a large panel. Avoid `apply` whenever a built-in does the job.

## A worked example: per-symbol equity curve

```python
def equity_curve(group: pd.DataFrame) -> pd.Series:
    return (1 + group["return"]).cumprod()

df["equity"] = df.groupby("symbol").apply(equity_curve).droplevel(0)
```

You *can* do this with `transform("cumprod")` if you preprocess:

```python
df["equity"] = (1 + df["return"]).groupby(df["symbol"]).cumprod()
```

Same answer, much faster — `cumprod` is a built-in `transform`-compatible op.

The pattern: anything pandas has as a built-in groupby method (`cumsum`, `cumprod`, `cummax`, `cummin`, `diff`, `shift`, `pct_change`, `rank`, `nunique`, `first`, `last`, `nth`) goes through the fast path.

## A worked example: rolling per group (the slow way and the fast way)

```python
# Slow on large panels — groupby+rolling has per-group Python overhead
df["ma20"] = df.groupby("symbol")["close"].rolling(20).mean().reset_index(level=0, drop=True)

# Faster — reshape, roll, restack
wide = df.set_index(["timestamp", "symbol"])["close"].unstack("symbol")
ma20_wide = wide.rolling(20).mean()
ma20_long = ma20_wide.stack("symbol")
```

The reshape version operates on a single (T, N) array. The groupby version operates on N separate series. For N=5,000, the second is far slower.

For the absolute fastest path on big panels, polars (next chapter) or vectorised NumPy over a (T, N) array (Module 3) wins.

## `groupby(level=...)` for MultiIndex panels

When the symbol is part of the index, group by level:

```python
# panel is MultiIndex (symbol, timestamp)
panel.groupby(level="symbol")["return"].transform("cumsum")
```

This avoids extra column allocations and reads natively.

## `as_index=False` — when you want a tidy frame back

```python
df.groupby("symbol", as_index=False).agg(mean_ret=("return", "mean"))
#  symbol  mean_ret
# 0  IWM    0.0024
# 1  QQQ    0.0011
# 2  SPY    0.0007
```

By default `groupby` puts the key in the index; `as_index=False` keeps it as a column. Useful when feeding the result into another `merge`.

## `observed=True` — categorical-only

If you grouped by a `Categorical`, by default pandas includes every category — even ones with zero rows. Pass `observed=True` to skip them:

```python
df.groupby("sector", observed=True).agg(...)
```

A common silent bug source for sector / industry groupings where the categorical has more levels than the data does.

## `dropna=False` — when NaN is meaningful

```python
df.groupby("category", dropna=False).size()
```

By default groupby skips rows where the key is NaN. `dropna=False` keeps them in their own NaN group — important when "no data" is itself a category you want to track.

## A trap: `groupby.apply` returning DataFrames silently re-orders

`apply`'s output is stacked in the order the groups were processed. For sorted keys, that's the same as the input; for unsorted, you may get a re-ordered result. Pass `sort=False` to `groupby` to preserve original order:

```python
df.groupby("symbol", sort=False).apply(fn)
```

`sort=False` is also a small speed win for large group counts.

## A trap: `apply` running once for free

The first call to `apply` runs the function once to sniff the return type. If the function has side effects (a print, a database call, a mutation), they happen twice for one of the groups. Make your `apply` callbacks pure.

## Bottom line

1. **Reductions** → `groupby(...).agg(...)`, prefer named aggregation and built-ins over lambdas.
2. **Same-shape transforms** → `groupby(...).transform("name")`. Avoid lambdas.
3. **Cumulative / per-group sequential** → built-in `cumsum`/`cumprod`/`shift`/`diff`/`pct_change`.
4. **Rolling per group on big panels** → reshape to wide, vectorise, reshape back. Or use polars.
5. **`apply`** → only when nothing else fits, and consider polars/NumPy first.

Continue to **[Polars from a pandas user's perspective](05-polars.md)**.
