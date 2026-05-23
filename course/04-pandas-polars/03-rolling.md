# Rolling, expanding, and `ewm`

Moving statistics are the bread and butter of every signal you'll ever build. Pandas's `rolling`, `expanding`, and `ewm` get you 95% of the way; knowing the subtleties — `min_periods`, time-based windows, the numba engine, alignment — gets you the rest.

## `rolling` — fixed-size window

```python
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(1000).cumsum() + 100)

s.rolling(20).mean()        # 20-period SMA (the first 19 are NaN)
s.rolling(20).std()
s.rolling(20).quantile(0.95)
s.rolling(20).apply(lambda x: x[-1] - x.mean(), raw=True)
```

`raw=True` passes a NumPy array into the callback instead of a Series — dramatically faster for custom functions.

### `min_periods` — how soon to start producing values

By default `rolling(20)` produces NaN for the first 19 rows. Sometimes you want partial windows:

```python
s.rolling(20, min_periods=5).mean()
# Computes the mean from the 5th observation onward
```

This is the right pattern when you can't afford to throw away the start of a series — e.g. real-time signals on a fresh restart.

### Time-based windows

If the index is a `DatetimeIndex`, you can use a time string:

```python
s.rolling("7D").mean()       # last 7 calendar days, irregular spacing OK
s.rolling("60s").max()       # last 60 seconds
```

The win: this works correctly for **irregularly sampled data** (e.g. tick streams). A fixed-count `rolling(60)` would behave wildly differently on quiet vs busy minutes.

### The numba engine

For custom `apply` functions, pandas can JIT-compile via numba:

```python
s.rolling(252).apply(my_fn, raw=True, engine="numba", engine_kwargs={"parallel": True})
```

First call pays a compile cost; subsequent calls are C speed. Use it whenever an `apply` is the bottleneck.

## `expanding` — window grows from the start

```python
s.expanding(min_periods=10).mean()    # mean of all observations so far
s.expanding().std()                    # cumulative std
```

Useful for "all-history-up-to-now" statistics: a rolling Sharpe with no fixed lookback, a running quantile, a CDF you build online.

## `ewm` — exponentially weighted

The exponentially weighted moving average gives more weight to recent observations:

$$
\bar{x}_t = (1 - \alpha) \bar{x}_{t-1} + \alpha x_t
$$

Three equivalent specifications in pandas:

```python
s.ewm(alpha=0.05).mean()             # explicit decay factor
s.ewm(span=20).mean()                 # equivalent: alpha = 2/(span+1)
s.ewm(halflife=10).mean()             # decay so that weight halves in 10 periods
```

For volatility:

```python
realized = (returns ** 2).ewm(halflife=22).mean().pipe(np.sqrt)
```

Half-life of 22 trading days is a reasonable starting point for daily-bar realized vol — it's the "RiskMetrics" decay.

### `ewm.cov` and `ewm.corr` — exponentially weighted covariance

Multivariate version:

```python
# returns: T × N DataFrame
cov_path = returns.ewm(halflife=60).cov()         # MultiIndex (time, symbol_i, symbol_j)
latest_cov = cov_path.iloc[-len(returns.columns):]
```

You almost always want the latest covariance for portfolio construction — chapter 11 of Module 20 covers shrinkage, which fixes the noise.

### Time-based ewm (`times=`)

For irregularly spaced data, pass the time vector:

```python
s.ewm(halflife="1H", times=s.index).mean()
```

The decay is computed against actual elapsed time, not row count. Essential for tick-level vol.

## `rolling` vs `ewm` — which to choose

| | rolling | ewm |
|---|---|---|
| Behaviour at series start | undefined for first window | smoothly grows |
| Response to outliers | shock disappears after window | shock decays exponentially |
| Speed | very fast | very fast |
| Common use | moving averages of fixed look-back | volatility, alpha decay, noise filtering |

For trading signals, `ewm` is generally the better default — no hard cliff when an old observation drops out.

## A worked example: Bollinger bands

```python
def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + k * std,
        "lower": mid - k * std,
        "z": (close - mid) / std,            # z-score useful for entry rules
    })
```

Three rolling calls, one z-score, all vectorised. The z-score column is the practical signal — if you're not centering and scaling, you're not doing it right.

## A worked example: rolling Kalman half-life of mean-reversion

A quick-and-dirty cointegration spread half-life: regress the change on the level, take `-log(2) / log(1 + beta)`.

```python
def rolling_half_life(spread: pd.Series, window: int = 60) -> pd.Series:
    diff = spread.diff()
    lag = spread.shift(1)
    out = pd.Series(index=spread.index, dtype=float)
    for end in range(window, len(spread)):
        x = lag.iloc[end - window:end]
        y = diff.iloc[end - window:end]
        beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        hl = -np.log(2) / np.log1p(beta) if beta < 0 else np.nan
        out.iloc[end] = hl
    return out
```

For a *real* version, see Module 8 (Kalman pairs with online beta) and Module 11 (Numba-accelerated rolling regression). The structure above is a Python loop — fine for small frames, slow for big ones.

## A worked example: vol-targeting overlay

```python
def vol_targeted(returns: pd.Series, target_vol: float = 0.10, lookback: int = 60) -> pd.Series:
    realised = returns.rolling(lookback).std() * np.sqrt(252)
    # avoid div-by-zero at the start
    scale = (target_vol / realised).clip(upper=3.0).shift(1)
    return returns * scale
```

The `shift(1)` is the most important line — you scale today by yesterday's measured vol. Without it, you're using contemporaneous information. Module 9 has a chapter on this exact category of bug.

## A trap: `rolling` with NaN

`rolling(window).mean()` is **NaN if any element of the window is NaN**. For partial tolerance:

```python
s.rolling(20).apply(lambda x: np.nanmean(x), raw=True)
```

Slower, but correct when the data has holes. Better fix: clean the data upstream and don't have holes.

## A trap: rolling on a multi-index frame

If your panel is `(symbol, timestamp)` MultiIndex, a naive `df.rolling(20).mean()` rolls *across symbols* at the boundary. The fix:

```python
df.groupby(level="symbol").rolling(20).mean()
```

The groupby+rolling pattern is correct but slow for large panels. Polars handles this cleanly with expressions (see chapter 5).

## A trap: silently using a future window

```python
# WRONG — uses future bars
df["ma20"] = df["close"].rolling(window=20, center=True).mean()
```

`center=True` centres the window on the middle observation, which means **half of it is in the future**. Useful for smoothing already-finished historical data; **never** in a signal.

## Polars equivalents (preview)

```python
import polars as pl

# polars expressions — rolling is a method on expression
df.with_columns([
    pl.col("close").rolling_mean(window_size=20).alias("ma20"),
    pl.col("close").ewm_mean(span=20).alias("ema20"),
])
```

Polars rolling is parallelised by default, no `groupby` boilerplate for per-symbol windows (use `.over("symbol")`), and substantially faster on large panels. Chapter 5 covers the polars way in detail.

## Bottom line

- For moving averages and standard deviations of fixed lookback → `rolling(W).mean()/std()`.
- For decaying influence → `ewm(halflife=H).mean()`.
- For "from-the-beginning" statistics → `expanding()`.
- Always **shift signal results forward by 1** before using them as positions.
- Always **use `groupby(level=symbol).rolling()`** on panels, never a flat rolling.
- For custom logic in `apply`, **use `raw=True, engine="numba"`** when speed matters.

Continue to **[GroupBy: `apply` vs `agg` vs `transform`](04-groupby.md)**.
