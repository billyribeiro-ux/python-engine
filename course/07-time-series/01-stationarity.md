# Stationarity, ADF, KPSS, and friends

Almost every time-series model you'll learn assumes the data is **stationary**. Most financial data isn't. Returns are *roughly* stationary. Prices and volume aren't. This chapter is how you check, and what to do when the answer is "no".

## What stationarity actually means

**Strict stationarity** — the joint distribution of $(X_t, X_{t+1}, ..., X_{t+k})$ doesn't depend on $t$. Too strong; nobody checks this.

**Weak (or covariance) stationarity** — three properties:

1. $\mathbb{E}[X_t]$ is constant in $t$.
2. $\text{Var}(X_t)$ is constant in $t$.
3. $\text{Cov}(X_t, X_{t+h})$ depends only on $h$, not $t$.

This is the working definition. ARIMA, GARCH, AR, all the standard models assume it.

A return series is approximately weakly stationary. A price series is not (its mean drifts). A vol series is not (regimes). Always work on **returns** or **differences**, not levels.

## The ADF test

The Augmented Dickey-Fuller test. Null hypothesis: the series has a *unit root* (i.e. it's non-stationary, like a random walk). Reject the null → the series is stationary.

```python
from statsmodels.tsa.stattools import adfuller

stat, pvalue, *_ = adfuller(returns)
print(f"ADF stat={stat:.3f}, p={pvalue:.4f}")
# typical for returns: p < 0.001 → reject null → stationary

stat, pvalue, *_ = adfuller(prices)
print(f"ADF on prices: p={pvalue:.4f}")
# typical: p ≈ 0.4 → can't reject random walk → treat prices as non-stationary
```

ADF takes a `regression` argument: `"c"` (constant), `"ct"` (constant + trend), `"ctt"` (constant + trend + trend²), `"nc"` (none). Choose based on whether the series visibly has a trend.

## The KPSS test

The KPSS test is the **mirror image** of ADF. Null hypothesis: the series *is* stationary. Reject the null → it's not.

```python
from statsmodels.tsa.stattools import kpss

stat, pvalue, *_ = kpss(returns, regression="c")
print(f"KPSS stat={stat:.3f}, p={pvalue:.4f}")
# typical for returns: p > 0.1 → fail to reject → stationary
```

The intent is to combine both tests for higher confidence:

| ADF result | KPSS result | Verdict |
|---|---|---|
| reject (stationary) | fail to reject (stationary) | confidently stationary |
| fail to reject (non-stationary) | reject (non-stationary) | confidently non-stationary |
| reject (stationary) | reject (non-stationary) | **fractional integration** — somewhere in between |
| fail to reject | fail to reject | inconclusive, more data needed |

The fractional case (case 3) is the interesting one for finance — many series sit between I(0) and I(1) and respond to fractional differentiation rather than first differencing. López de Prado pushes this point hard in *Advances in Financial Machine Learning*.

## Phillips-Perron

Like ADF but with non-parametric correction for serial correlation. Use it when ADF gives borderline results:

```python
from statsmodels.tsa.stattools import zivot_andrews         # for structural breaks
from arch.unitroot import PhillipsPerron

pp = PhillipsPerron(returns)
print(pp)
```

For typical work, ADF + KPSS is plenty.

## Fractional differentiation

Standard differencing (`diff(1)`) makes prices stationary but **destroys long-memory structure** — exactly the structure you want for ML features. Fractional differentiation finds the *minimum* differencing order $d \in [0, 1]$ that achieves stationarity, preserving as much memory as possible.

```python
import numpy as np

def frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Coefficients of the fractional differencing series, truncated to `size`."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])

def frac_diff(series: np.ndarray, d: float, thresh: float = 1e-4) -> np.ndarray:
    """López de Prado's fixed-width window version."""
    w = frac_diff_weights(d, len(series))
    # find truncation where weights become negligible
    w_cum = np.cumsum(np.abs(w[::-1]))
    width = int(np.searchsorted(w_cum, thresh * w_cum[-1]))
    w = w[-width:]
    out = np.full(len(series), np.nan)
    for t in range(width, len(series)):
        out[t] = (w * series[t - width + 1 : t + 1]).sum()
    return out
```

Sweep $d$ from 0 to 1; for each, test stationarity with ADF; pick the smallest $d$ that gets you to stationarity. For SPY prices, typical answers are $d \approx 0.35-0.45$.

This is one of the highest-value tricks in financial ML — features computed on a fractionally differenced price series outperform features on returns *and* on raw prices for many tasks. Module 10 returns to this.

## ACF and PACF

The **autocorrelation function** (ACF) measures correlation between $X_t$ and $X_{t-h}$ across lags $h$. The **partial autocorrelation function** (PACF) does the same controlling for intermediate lags.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(returns, lags=40)            # raw returns: tiny spike at lag 0, near-zero elsewhere
plot_acf(returns ** 2, lags=40)       # absolute returns: large persistent decay → vol clustering!
```

Three patterns to recognise:

- **Returns ACF**: nearly zero at all lags. Markets are efficient at the daily frequency.
- **Squared (or absolute) returns ACF**: slowly decaying. **This is volatility clustering** — the empirical fact that motivates the entire GARCH family.
- **Pairs spread ACF**: spike at lag 1, decay over a few weeks → mean-reverting.

## Hurst exponent

A single-number summary of the long-memory behaviour of a series:

- $H = 0.5$: random walk, no memory.
- $H > 0.5$: trending / persistent.
- $H < 0.5$: mean-reverting / anti-persistent.

```python
import numpy as np

def hurst(series: np.ndarray, lags: np.ndarray = None) -> float:
    """Estimate via the R/S statistic. Works for series of length > ~200."""
    series = np.asarray(series, dtype=float)
    if lags is None:
        lags = np.unique(np.logspace(0.7, np.log10(len(series) // 4), 30).astype(int))
    tau = []
    for lag in lags:
        x = series[lag:] - series[:-lag]
        tau.append(np.sqrt(np.std(x)))
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)
```

For SPY's daily returns, $H \approx 0.5$ (random walk). For volatility, $H \approx 0.7-0.8$ (strongly persistent). For a mean-reverting pair spread, $H \approx 0.3-0.4$.

The Hurst exponent is most useful as a regime indicator: when $H$ on a rolling window crosses 0.5, the character of the series is changing.

## Cointegration — when individually non-stationary series move together

Two non-stationary series $X_t, Y_t$ are **cointegrated** if some linear combination $Y_t - \beta X_t$ is stationary. This is the mathematical basis of pairs trading.

The **Engle-Granger** two-step test:

```python
from statsmodels.tsa.stattools import coint

stat, pvalue, _ = coint(asset_a_prices, asset_b_prices)
print(f"Cointegration p-value: {pvalue:.4f}")
```

For more than two series, the **Johansen** test:

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

result = coint_johansen(prices_matrix, det_order=0, k_ar_diff=1)
print(result.lr1)            # trace test statistics
print(result.cvt[:, 1])      # 95% critical values
# When lr1[i] > cvt[i, 1], reject "rank ≤ i" → at least i+1 cointegrating relationships
```

Module 8 builds a complete pairs trading strategy on top of this.

## Pitfalls

!!! warning "Stationarity tests on a non-stationary series with a strong trend can falsely 'reject'"
    ADF's null assumes the alternative is stationary around either a constant or a constant+trend. Pick the right alternative; otherwise interpret the p-value carefully.

!!! warning "Sample-size dependence"
    All these tests are noisier on shorter samples. A test that fails on 100 observations may pass on 1,000.

!!! warning "Stationarity is local"
    A series can be stationary on one regime and non-stationary across regimes. Always also check rolling-window stationarity, not just the full sample.

!!! warning "Cointegration is even more sample-dependent"
    Two random walks are spuriously cointegrated about 5% of the time at p<0.05 (because that's what 5% means). Use multiple-testing corrections when scanning many pairs.

## Bottom line

For most quant work:

- Work on **returns** or **fractionally differenced** prices, not raw prices.
- Run **ADF + KPSS** as a pair; trust the agreeing case, investigate the disagreeing one.
- Plot **ACF on squared returns** to convince yourself volatility clustering is real and motivates GARCH.
- For pairs: **Engle-Granger** for two assets, **Johansen** for portfolios.

Continue to **[ARIMA and the GARCH family](02-arima-garch.md)**.
