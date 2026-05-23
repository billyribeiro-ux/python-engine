# The bootstrap, properly

The bootstrap is the most under-used inferential tool in retail quant work. It gives you a confidence interval for *any* statistic you can compute, with no distributional assumptions. For finance — where distributions are rarely Gaussian and almost never IID — that's enormous.

But the textbook IID bootstrap is *wrong* for time series. The fix (block bootstrap) is one line of code and night-and-day for honesty.

## The IID bootstrap

Given a sample $x_1, ..., x_n$, an estimate $\hat\theta$:

1. Draw $n$ values *with replacement* from the sample. That's a **bootstrap resample**.
2. Compute the statistic on the resample. That's a **bootstrap estimate**.
3. Repeat $B$ times to get a distribution of bootstrap estimates.
4. Take the 2.5th and 97.5th percentiles for a 95% confidence interval.

In code:

```python
import numpy as np
rng = np.random.default_rng(0)

def bootstrap_ci(x: np.ndarray, statistic, B: int = 10_000, alpha: float = 0.05):
    n = len(x)
    samples = rng.choice(x, size=(B, n), replace=True)
    estimates = np.apply_along_axis(statistic, axis=1, arr=samples)
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

returns = np.random.randn(1000) * 0.01
ci = bootstrap_ci(returns, np.mean)
print(f"95% CI for mean return: ({ci[0]:.4f}, {ci[1]:.4f})")
```

That's it. No formula, no normality assumption, works for *any* statistic.

## What the IID bootstrap gets wrong for time series

It assumes the observations are independent. They aren't. Returns have:

- **Autocorrelation** in absolute value (the volatility cluster).
- **Volatility regime** changes that span many days.
- **Cross-asset dependence** that varies over time.

If you bootstrap returns IID, you destroy this structure. The resulting confidence intervals are over-optimistic: they assume the *true* volatility of your statistic is smaller than it really is, because the resamples don't preserve the dependence.

## The moving-block bootstrap

The fix: resample **blocks** of consecutive observations, not individual observations.

```python
def block_bootstrap(x: np.ndarray, block_len: int, B: int = 10_000) -> np.ndarray:
    n = len(x)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(B, n_blocks))
    out = np.empty((B, n_blocks * block_len))
    for b in range(B):
        chunks = [x[s : s + block_len] for s in starts[b]]
        out[b] = np.concatenate(chunks)
    return out[:, :n]                          # trim to original length
```

Each resample is built from random starting points; each block preserves the local dependence; the boundaries between blocks lose dependence but the bulk of structure is retained.

```python
samples = block_bootstrap(returns, block_len=20, B=10_000)
mean_estimates = samples.mean(axis=1)
ci = np.percentile(mean_estimates, [2.5, 97.5])
print(f"Block-bootstrap 95% CI for mean: {ci}")
```

For a typical daily return series, **block lengths of 10–60** are reasonable. The right choice depends on the autocorrelation horizon — measure with the ACF if you want to be precise (Module 7).

## Stationary bootstrap (Politis-Romano)

A refinement: instead of fixed block length, use **random** block lengths drawn from a geometric distribution. This makes the bootstrap resample itself a stationary process.

```python
def stationary_bootstrap(x: np.ndarray, mean_block_len: float, B: int = 10_000) -> np.ndarray:
    n = len(x)
    p = 1 / mean_block_len
    out = np.empty((B, n))
    for b in range(B):
        i = 0
        idx = rng.integers(0, n)
        while i < n:
            out[b, i] = x[idx]
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
            i += 1
    return out
```

Same idea, smoother in spirit. For honest CIs on Sharpe ratios and similar derived statistics, this is what to reach for. The `arch` package has a production implementation: `arch.bootstrap.StationaryBootstrap`.

## A worked example: confidence interval on a backtest Sharpe

```python
from arch.bootstrap import StationaryBootstrap

def sharpe(r):
    return r.mean() / r.std(ddof=1) * np.sqrt(252)

bs = StationaryBootstrap(20, returns)        # mean block length 20
ci = bs.conf_int(sharpe, reps=5000, method="bca").flatten()
print(f"Bootstrap Sharpe 95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

Two things to notice:

- `BCa` (bias-corrected accelerated) intervals adjust for skew in the bootstrap distribution. They're usually a small but real improvement over plain percentile intervals.
- The CI is going to be **wider than you expect**. That's the point — the IID approximation has been making everyone over-confident for decades.

## What the bootstrap can and can't tell you

**Can:**
- A confidence interval for any statistic you can compute.
- A p-value for "is this statistic > zero?" (via the fraction of bootstrap estimates below zero).
- A standard error for downstream uncertainty propagation.

**Can't:**
- Make data you don't have. If your sample is too short or your event of interest happened twice, no bootstrap will rescue you.
- Tell you about regimes you've never seen. The Sharpe CI from a quiet-market sample says nothing about the Sharpe in a crisis.

For tail events specifically, **extreme value theory** (chapter 5) does what the bootstrap can't.

## A worked example: significance of a pairs spread half-life

You computed a half-life of 22 days for an OU-process spread. Is it significantly different from "no mean reversion" (which would be infinite half-life)?

```python
def half_life(spread):
    diff = np.diff(spread)
    lag = spread[:-1]
    beta = np.cov(lag, diff, ddof=1)[0, 1] / np.var(lag, ddof=1)
    return -np.log(2) / np.log1p(beta) if beta < 0 else np.inf

bs = StationaryBootstrap(15, spread)
ci = bs.conf_int(half_life, reps=2000)
print(f"Half-life 95% CI: {ci}")
```

If the upper bound of the CI is, say, 80 days, you have weak evidence of mean reversion. If it's 30 days, you're confident.

## Pitfalls

!!! warning "Bootstrapping the parameter, not the data"
    For statistics like the Sharpe ratio, you bootstrap the *returns* and recompute. You do *not* bootstrap the per-day Sharpe estimates — that's a different problem.

!!! warning "Block length too small kills dependence"
    A block length of 1 is the IID bootstrap. For volatility-cluster data, that's a serious bias toward narrow CIs.

!!! warning "Block length too large kills the variance gain"
    If the block is your whole sample, you just resample the same thing every time. The variance estimate collapses to zero.

!!! warning "Bootstrap of stationary statistics on non-stationary data"
    The bootstrap assumes the underlying process is stationary. Prices aren't; returns roughly are. Always bootstrap returns, not prices.

## Bottom line

For confidence intervals on financial statistics:

- IID returns? **`StationaryBootstrap`** from the `arch` package. Use the BCa method.
- Cross-sectional statistics? Plain IID bootstrap is fine — the dependence is across time, not across symbols.
- Tail statistics (VaR, max drawdown)? Bootstrap is OK for the body but use EVT (chapter 5) for the tail.

Reach for the bootstrap any time someone hands you a single backtest result. The first question to ask is "what's the CI?" — and the bootstrap is how you answer it.

Continue to **[Bayesian thinking for traders](03-bayesian.md)**.
