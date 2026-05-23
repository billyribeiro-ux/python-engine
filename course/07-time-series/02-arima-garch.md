# ARIMA and the GARCH family

ARIMA is the textbook model for time series; GARCH is the textbook model for volatility. For returns, ARIMA models are usually weak because returns are barely autocorrelated. GARCH models are anything but weak — volatility clustering is the strongest stylised fact in finance, and GARCH is the natural framework.

This chapter is the practical guide to fitting these models in Python and not falling for the common traps.

## ARIMA in 60 seconds

ARIMA(p, d, q) — AutoRegressive(p), Integrated(d), Moving Average(q):

- **AR(p)**: $x_t = \phi_1 x_{t-1} + ... + \phi_p x_{t-p} + \varepsilon_t$
- **MA(q)**: $x_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}$
- **I(d)**: differenced $d$ times before fitting.

For returns, $d = 0$. For prices, $d = 1$ (or fractional, per chapter 1).

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(returns, order=(1, 0, 1))
fit = model.fit()
print(fit.summary())
forecast = fit.forecast(steps=5)
```

Three things to know:

1. **Pick (p, q) using AIC** — refit for several orders, take the smallest AIC. `pmdarima.auto_arima` automates this if you can install it.
2. **Forecast horizon matters**: ARIMA gives reasonable 1-step forecasts and degenerate (mean-reverting) forecasts at long horizons. It is *not* a forecasting machine.
3. **For finance the gain over a constant mean is usually negligible** for daily returns. ARIMA on intraday returns or on macro series (rates, FX) can have an edge.

## The GARCH family — why volatility models matter

Volatility *clusters*: large moves are followed by large moves, calm by calm. ACF of squared returns shows this empirically (Module 7, chapter 1). GARCH formalises it:

$$
r_t = \mu + \sigma_t z_t, \quad z_t \sim \text{i.i.d.}
$$
$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Today's vol is a weighted combination of yesterday's surprise (the $r_{t-1}^2$ term) and yesterday's vol (the $\sigma_{t-1}^2$ term). Two parameters; one of the most powerful three-line models in all of statistics.

Constraints: $\omega > 0$, $\alpha, \beta \geq 0$, $\alpha + \beta < 1$ for stationarity.

```python
from arch import arch_model

am = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
res = am.fit(disp="off")
print(res.summary())
```

`returns * 100` because `arch` works better in percentage units numerically. `dist="t"` because residual returns are fatter-tailed than Gaussian.

Forecast one-step-ahead vol:

```python
forecast = res.forecast(horizon=1)
sigma_tomorrow = float(forecast.variance.iloc[-1, 0] ** 0.5) / 100
print(f"GARCH(1,1) σ for tomorrow: {sigma_tomorrow:.4f}  (or {sigma_tomorrow*np.sqrt(252):.2%} annualised)")
```

## Variants you should know about

### EGARCH — exponential GARCH

Models $\log \sigma_t^2$, which:

- automatically enforces positivity (no constraint on coefficients),
- allows **asymmetric** response to good vs bad news (the "leverage effect" — bad news raises vol more than good).

```python
am = arch_model(returns * 100, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t")
```

The `o=1` argument enables the asymmetry term. For equity indices, EGARCH typically fits noticeably better than GARCH.

### GJR-GARCH

A different way to capture asymmetry: add an extra term that fires only on negative returns.

$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \gamma r_{t-1}^2 \mathbb{1}_{r_{t-1} < 0} + \beta \sigma_{t-1}^2
$$

```python
am = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
```

In practice GJR and EGARCH give similar forecasts; pick the one with the lower AIC on your data.

### TARCH, APARCH, FIGARCH

Diminishing returns. The variant you fit to your data is rarely the largest source of edge — get a working GARCH or EGARCH and move on.

## Stochastic volatility models

GARCH treats vol as a *deterministic* function of past returns. Stochastic volatility (SV) models treat vol as a *latent process* with its own innovation:

$$
\log \sigma_t^2 = \alpha + \beta \log \sigma_{t-1}^2 + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

SV fits the data marginally better than GARCH but is harder to estimate (MCMC or particle filters; see chapter 4). For most practical purposes, EGARCH gets you 90% of the way at 10% of the cost.

## The realised volatility shortcut

If you have intraday data, you don't need GARCH for measurement (you still might want it for forecasting):

```python
# 5-minute returns aggregated to daily realised vol
intraday_returns = price_5min.pct_change()
daily_rv = (intraday_returns ** 2).groupby(intraday_returns.index.date).sum()
rv_annualised = np.sqrt(daily_rv * 252)
```

This is the **realised volatility** estimator — typically far more accurate than any daily-bar-only estimator. The literature on realised vol (Andersen, Bollerslev, Diebold; Barndorff-Nielsen, Shephard) is one of the most successful corners of empirical finance.

For tick-level data, naive realised vol is **biased** by market microstructure noise. The fixes — **realised kernels** and **pre-averaging** — are covered in Module 19 (Execution + Microstructure).

## A worked example: a vol-targeting overlay

Use a GARCH forecast to size positions to a target volatility:

```python
from arch import arch_model

def vol_targeted_position(returns, target_vol=0.10):
    am = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off", show_warning=False)
    sigma_tomorrow = float(res.forecast(horizon=1).variance.iloc[-1, 0] ** 0.5) / 100
    annual_vol = sigma_tomorrow * np.sqrt(252)
    return min(target_vol / annual_vol, 3.0)
```

The `min(..., 3.0)` cap prevents the overlay from going crazy when realised vol gets very low. This is the same vol-targeting pattern from Module 4, with the GARCH forecast replacing the rolling-std estimator. The forecast is better in volatile periods (it responds to vol-of-vol).

## Things to check after fitting

1. **Residuals are white noise**. The Ljung-Box test on `res.resid / res.conditional_volatility`:

   ```python
   from scipy.stats import jarque_bera
   from statsmodels.stats.diagnostic import acorr_ljungbox

   std_resid = res.resid / res.conditional_volatility
   lb = acorr_ljungbox(std_resid, lags=[10], return_df=True)
   print(lb)                                # p > 0.05 → no remaining structure (good)
   ```

2. **The shape parameter of the t-distribution**. For equity index returns, expect 6-10 degrees of freedom. If it's <4, you have extreme tails and should sanity-check the data.

3. **Persistence** $\alpha + \beta$. Should be < 1, typically around 0.95-0.99 for daily equity data. Higher = more persistent vol; lower = mean-reverts faster.

## Multivariate GARCH

For portfolio vol forecasting, the **DCC-GARCH** (Dynamic Conditional Correlation) is the workhorse:

```python
# arch package doesn't ship DCC; use the `mgarch` package or implement directly
# Conceptually:
# 1. Fit univariate GARCH to each return series
# 2. Standardise to get residuals z_i,t
# 3. Model the correlation matrix of z_i,t as time-varying via DCC
```

For most workflows, **DCC is overkill** vs. either an exponentially weighted covariance (Module 4) or shrinkage (Module 11/20). Save DCC for when you genuinely care about a time-varying joint distribution.

## Pitfalls

!!! warning "GARCH on a series with regimes"
    A single GARCH model fit across a 2008-crisis-included sample will average across regimes. Use a regime-switching GARCH (`MS-GARCH`) or fit per-regime.

!!! warning "Daily-bar GARCH with intraday data available"
    If you have minute data, your intraday-realised-vol estimate is much sharper than what GARCH can produce from daily bars. Use both — GARCH for the smoothing, realised vol for the level.

!!! warning "Don't use the Gaussian residual distribution for equities"
    `dist="t"` is the default for a reason. Gaussian residuals under-fit the tails and bias forecasts during stress.

!!! warning "Forecasting horizons > a week"
    GARCH mean-reverts to the unconditional vol in maybe 30-60 days. Don't extrapolate.

## Bottom line

- For daily equity vol forecasting, **EGARCH(1, 1) with t-distributed residuals** is the right default.
- Forecast one step ahead, refit weekly or monthly.
- Use **realised vol** if you have intraday data — better measurement, not just better forecasting.
- For multivariate vol, **exponentially weighted covariance** or **shrunk sample covariance** beat DCC in practice.

Continue to **[Kalman filters from scratch](03-kalman.md)**.
