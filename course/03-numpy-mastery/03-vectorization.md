# Vectorisation patterns

This chapter is a pattern catalogue. Each one replaces a Python loop with a NumPy expression that runs 50–500× faster. Reach for these reflexively.

## 1. Conditional assignment — `np.where`

```python
import numpy as np

prices = np.array([100, 105, 95, 110, 90])
signal = np.where(prices > 100, 1, -1)
# array([-1,  1, -1,  1, -1])
```

Three-argument `where`: condition, value-if-true, value-if-false. All three are broadcast-compatible.

For multi-branch cascades, `np.select`:

```python
conds = [prices < 95, prices < 105, prices < 115]
values = [-1, 0, 1]
signal = np.select(conds, values, default=2)
```

## 2. Masked aggregation — `np.where` + `.sum`

To answer "average return on days the signal fired":

```python
signal = np.array([1, 0, 1, -1, 0])
returns = np.array([0.01, 0.005, -0.002, -0.003, 0.0])
fired = signal != 0
print(returns[fired].mean())
```

Or with `np.where`:

```python
print(np.where(fired, returns, 0).sum() / fired.sum())
```

The first form is usually clearer; the second is what to reach for when you want to keep array shape.

## 3. Differencing — `np.diff` and `prepend`

```python
prices = np.array([100, 102, 101, 105])
changes = np.diff(prices)               # array([ 2, -1,  4])
changes = np.diff(prices, prepend=prices[0])  # array([ 0,  2, -1,  4])
```

For returns:

```python
simple = np.diff(prices) / prices[:-1]
log = np.diff(np.log(prices))
```

## 4. Cumulative operations — `cumsum`, `cumprod`, `cummax`

```python
returns = np.array([0.01, -0.005, 0.02, -0.01])
equity_curve = np.cumprod(1 + returns)       # array([1.01, 1.005, 1.025, 1.015])
peak = np.maximum.accumulate(equity_curve)   # running max
drawdown = equity_curve / peak - 1
print(drawdown.min())                         # worst drawdown
```

`np.maximum.accumulate` (and `np.minimum.accumulate`) compute the running max/min in O(n). Way faster than a Python loop, way more readable than recasting it as a fold.

## 5. Where-arg — `np.argmax`, `np.argmin`, `np.argsort`

```python
returns = np.array([0.01, -0.005, 0.02, -0.01])
best_day = np.argmax(returns)        # 2
worst_day = np.argmin(returns)       # 3
order = np.argsort(returns)          # ascending: array([3, 1, 0, 2])
top3 = order[::-1][:3]               # 3 best days
```

For top-K without full sort, `np.argpartition` is O(n):

```python
top3 = np.argpartition(returns, -3)[-3:]   # the 3 largest, unordered
top3 = top3[np.argsort(returns[top3])[::-1]]  # ordered if you need it
```

For huge arrays where you only want the top 10, `argpartition` is far cheaper than `argsort`.

## 6. Boolean indexing for filtering

```python
above_average = prices > prices.mean()
strong_days = prices[above_average]
```

For multi-condition filtering, combine with `&`, `|`, `~` (note the parens are required because of operator precedence):

```python
big_up = (returns > 0.01) & (volumes > 1_000_000)
print(prices[big_up])
```

## 7. Fancy indexing for gather/scatter

```python
idx = np.array([0, 5, 2, 7, 0])         # repeats allowed
prices[idx]                              # gather: shape matches idx
```

Scatter:

```python
out = np.zeros(10)
np.add.at(out, idx, [1, 1, 1, 1, 1])    # increment at idx, handling repeats correctly
```

A plain `out[idx] += 1` only increments each unique index once (because the right-hand side is evaluated then assigned, and only the last write survives). `np.add.at` is the buffered version.

## 8. Aggregations with weights — `np.average`

```python
prices = np.array([100, 101, 102, 100])
weights = np.array([1, 2, 5, 1])
vwap = np.average(prices, weights=weights)
```

Cleaner than `(prices * weights).sum() / weights.sum()`.

## 9. Histograms in vectorised form

```python
returns = np.random.randn(10000) * 0.02
hist, edges = np.histogram(returns, bins=50)
```

For multidimensional, `np.histogram2d` and `np.histogramdd`.

## 10. Sorting and `searchsorted`

`np.searchsorted` is the vectorised binary search. If you have a sorted array of bucket edges and want to assign incoming values to buckets:

```python
edges = np.array([0, 100, 500, 1000, 5000])
sizes = np.array([50, 300, 800, 2000, 10000])
buckets = np.searchsorted(edges, sizes, side="right")
# array([1, 2, 3, 4, 5])
```

Vastly faster than a Python loop, especially for million-element arrays.

## 11. Aggregating without `groupby` — `np.add.reduceat`

For the "per-group sum" pattern when the groups are runs of consecutive equal values, `reduceat` is the right tool:

```python
xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
indices = np.array([0, 3, 5])           # group starts
sums = np.add.reduceat(xs, indices)     # sum of each segment
# array([ 6,  9, 30])    (1+2+3,  4+5,  6+7+8+9)
```

For arbitrary group keys, pandas's `groupby` (or `numpy_groupies`) is usually cleaner.

## 12. Replacing nans without filtering

```python
returns = np.array([0.01, np.nan, -0.005, np.nan, 0.02])
clean = np.nan_to_num(returns, nan=0.0)
# array([ 0.01 ,  0.   , -0.005,  0.   ,  0.02 ])

mean = np.nanmean(returns)        # 0.00833...   — ignores NaNs
std = np.nanstd(returns)
```

Every reduction has a `nan*` variant: `nansum`, `nanmean`, `nanmax`, `nanmedian`, `nanargmax`, ...

## 13. `np.clip` for bounding

```python
positions = np.array([0.8, -1.2, 0.3, 2.0])
bounded = np.clip(positions, -1, 1)
# array([ 0.8, -1. ,  0.3,  1. ])
```

A Kelly-fraction limiter in one line.

## 14. `np.diff` of dates

`np.datetime64` supports arithmetic:

```python
dates = np.array(["2024-01-02", "2024-01-03", "2024-01-05"], dtype="datetime64[D]")
gaps = np.diff(dates)
# array([1, 2], dtype='timedelta64[D]')
```

You can convert to integer days with `gaps.astype(int)`.

## 15. The "vectorise a piecewise function" trick

You have a function with three branches: f(x) = 0 if x<a, linear from a to b, 1 if x>b. The non-vectorised form is a `for` loop. The vectorised form:

```python
def smooth_threshold(x, a, b):
    x = np.asarray(x)
    out = np.zeros_like(x)
    mid = (x > a) & (x < b)
    out[mid] = (x[mid] - a) / (b - a)
    out[x >= b] = 1.0
    return out
```

Two boolean masks, two indexed assignments. C speed. Reads almost like prose.

## 16. The vectorised cross-sectional rank

We saw this in the broadcasting chapter; it shows up so often it earns a place here too:

```python
def xsec_rank(X: np.ndarray) -> np.ndarray:
    """Rank each row of X to centered [-0.5, +0.5]."""
    order = X.argsort(axis=1).argsort(axis=1)
    return order / (X.shape[1] - 1) - 0.5
```

## 17. Replacing `iterrows` over pandas

If you're walking a pandas DataFrame to compute something row-wise:

```python
# slow
out = []
for i, row in df.iterrows():
    out.append(row["close"] / row["open"] - 1)

# fast
out = df["close"].values / df["open"].values - 1
```

`.values` (or `.to_numpy()`) gives you a NumPy array; the rest is element-wise math. Module 4 covers when you can stay in pandas (`df["close"] / df["open"] - 1` also works directly).

## 18. `np.vectorize` is NOT vectorisation

`np.vectorize(f)` runs your Python function once per element. It's a convenience wrapper, not a speedup. It's the wrong choice 95% of the time.

The exception: when `f` truly cannot be expressed as element-wise array operations (e.g. a Brent's-method root finder for implied volatility per option). In those cases, `np.vectorize` at least cleans up the bookkeeping. But understand: it is not vectorisation; it's a Python loop with a nicer face.

## The mental model

When you see a `for` loop over array elements, ask:

1. Can this be expressed as `arr1 op arr2`? → element-wise.
2. Can this be expressed with broadcasting? → `arr[..., None] op other[None, ...]`.
3. Is it a reduction? → `np.sum/mean/.../axis=`.
4. Is it a sort/rank? → `argsort/argpartition/searchsorted`.
5. Is it a stateful sweep (cumulative)? → `np.cumsum/cumprod/maximum.accumulate`.
6. Is it a histogram / bucket? → `np.histogram/searchsorted/digitize`.

If none of the above applies, you have a genuinely loop-shaped problem. **That's when Numba (`@njit`) enters**. Module 11 covers it in the context of feature engineering.

Continue to **[einsum and tensor contractions](04-einsum.md)**.
