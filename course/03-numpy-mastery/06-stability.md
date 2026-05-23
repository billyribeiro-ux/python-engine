# Numerical stability

Floating-point arithmetic is a leaky abstraction. The good news: a handful of patterns will keep you out of trouble for the rest of your career. The bad news: if you don't know them, your model occasionally returns `NaN` and you can't figure out why.

This chapter covers the patterns. It is short on purpose — none of them are hard once you've seen them.

## Floats are not real numbers

Three facts to internalise:

1. **Most decimal numbers can't be represented exactly.** `0.1 + 0.2 == 0.30000000000000004`. Welcome.
2. **`float64` has ~16 decimal digits of precision.** Beyond that, you lose information.
3. **Subtracting two nearly-equal numbers cancels precision.** This is the root of most stability bugs.

```python
>>> 1e16 + 1 - 1e16
0.0
>>> 1e16 - 1e16 + 1
1.0
```

Same algebra, different precision. The first form lost the `+ 1` because `1e16 + 1 == 1e16` in `float64`.

## The log-sum-exp trick

You have $\log \sum_i \exp(x_i)$ — common in softmax, in Bayesian likelihoods, in mixture models. If any $x_i$ is, say, 800, `exp(800)` overflows to `inf`. If they're all very negative, the sum underflows to 0 and the `log` returns `-inf`.

The fix is purely algebraic. Let $M = \max_i x_i$. Then:

$$
\log \sum_i e^{x_i} = M + \log \sum_i e^{x_i - M}
$$

After the subtraction, the largest term in the sum is `exp(0) = 1`; all others are between 0 and 1. No overflow.

```python
import numpy as np

def log_sum_exp(x: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))

# or just use the SciPy version
from scipy.special import logsumexp
```

`scipy.special.logsumexp` is the correct, vectorised, axis-aware implementation. Use it. The hand-rolled version is here so you know what it's doing.

## Stable softmax

Same trick. Naive softmax:

```python
def softmax_unstable(x):
    e = np.exp(x)
    return e / e.sum()
```

Stable softmax:

```python
def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

PyTorch and JAX use this internally. So should you, if you're rolling your own.

## Catastrophic cancellation

Take the classical "variance, the easy formula":

$$
\operatorname{Var}(x) = \mathbb{E}[x^2] - \mathbb{E}[x]^2
$$

In code:

```python
def variance_unstable(x):
    return (x * x).mean() - x.mean() ** 2
```

Now compute the variance of $x = 10^7 + r$, where $r$ is small. Both terms are around $10^{14}$ — the subtraction wipes out the tiny variance.

```python
x = 1e7 + np.random.randn(1000)
print(variance_unstable(x))     # often 0.0 or negative — wrong
print(np.var(x))                # ~1.0 — correct
```

The two-pass algorithm computes the mean first, then the centred sum of squares:

```python
def variance_stable(x):
    m = x.mean()
    return ((x - m) ** 2).mean()
```

`np.var` uses Welford's online algorithm, which is even better (numerically stable AND single-pass). Just call `np.var` and don't write your own.

## When to use `np.float32` vs `np.float64`

- **Storage**: `float32` halves memory, often doubles cache throughput. For multi-billion-element arrays, this matters.
- **Default precision**: NumPy uses `float64`. PyTorch defaults to `float32`. JAX defaults to `float32` (and `float64` only with explicit enable). GPUs are far faster at `float32` than `float64`.
- **For accumulators**: even when the input is `float32`, sum into a `float64`. Otherwise the sum drifts.

```python
arr_f32 = np.random.randn(10_000_000).astype(np.float32)
print(arr_f32.sum())                                # float32 — drifts
print(arr_f32.sum(dtype=np.float64))                # accumulated in float64
print(np.sum(arr_f32, dtype=np.float64))            # equivalent
```

This is the same reason real BLAS implementations have separate single- and double-precision routines.

## When `nan` is the answer (and when it isn't)

`np.nan` is the right value for "missing data". It propagates through arithmetic (`nan + 1 == nan`), and reductions like `mean` poison the whole result. That's a feature: it forces you to be explicit.

```python
arr = np.array([1.0, 2.0, np.nan, 4.0])
print(arr.mean())       # nan
print(np.nanmean(arr))  # 2.333... — explicit about ignoring nans
```

`nan` is also what `0/0` and `inf - inf` produce. **`inf` is a different beast** — it propagates differently (`inf - 1 == inf`, `inf - inf == nan`).

!!! warning "`x == nan` is always False"
    Use `np.isnan(x)`. Same for `inf`: `np.isinf(x)`.

## Condition numbers — when matrix problems explode

If a linear system $Ax = b$ has a high **condition number** $\kappa(A)$, small errors in $b$ get amplified by up to $\kappa$ when you solve for $x$. For float64, $\kappa > 10^{15}$ means the answer is meaningless.

```python
A = np.random.randn(5, 5)
print(np.linalg.cond(A))     # something like 10–100, fine
```

A typical bad-conditioning sign: a covariance matrix close to singular because columns are nearly collinear. The fix is **regularisation** (add a small multiple of the identity) or **shrinkage** (Ledoit-Wolf, which we use in Module 11) or **decomposition-based solving** (use `scipy.linalg.lstsq`, `np.linalg.solve`, never invert the matrix directly).

!!! danger "Never invert a matrix to solve a linear system"
    `x = np.linalg.inv(A) @ b` is both slower and less stable than `x = np.linalg.solve(A, b)`. The first computes the inverse (more error-amplifying steps). The second factorises and solves directly. They're algebraically equivalent and numerically very different.

## A worked example: stable financial calculations

Computing log returns from prices:

```python
log_returns = np.diff(np.log(prices))      # stable, even at huge price levels
```

Computing implied volatility from option prices: use a Newton-Raphson with vega; check for vega → 0 (deep ITM/OTM) and fall back to Brent's method. The course's `engine.options` module has the full implementation in Module 14.

Computing portfolio P&L over many days:

```python
# unstable for long horizons because of repeated rounding
equity = np.cumprod(1 + returns)

# stable — work in log space, exponentiate at the end
log_equity = np.cumsum(np.log1p(returns))
equity = np.exp(log_equity)
```

`np.log1p(x) = log(1 + x)` is the stable version for small $x$. Same for `np.expm1(x) = exp(x) - 1`.

## End of Module 3

You now have the language of NumPy. The next module is **pandas + polars** — where this knowledge becomes the bedrock for a real data layer, and where we start touching market data in earnest.

Continue to **Module 4 — pandas + polars** (Phase 2).

!!! note "Phase 2 in progress"
    The pandas + polars module ships in the next phase of the course. Module 3 is the last chapter of Phase 1.

Until then, you have enough Python, hacks, and NumPy to go and do real damage on your own time series problems. Bookmark this site, run the examples, and bring questions to the Phase 2 review.
