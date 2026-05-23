# Extreme value theory

The body of a return distribution is interesting; the tail is the part that decides whether you survive. EVT is the branch of statistics that models the tail directly, instead of fitting a distribution to the whole thing and hoping the tail behaviour falls out right (which it never does).

This chapter is short and practical. The two theorems you need to know give you two methods you'll actually use.

## The setup

You have a long return series. The 99.9% worst day has happened a handful of times. Maybe never in your sample. Yet your strategy will live or die by those events.

Standard distributional fits (Gaussian, t, kernel) are pulled toward the bulk of the data. They under-estimate the tail because the tail is sparse and the bulk is dense. EVT inverts the problem: throw out the bulk, model only the extremes.

## Block maxima — the GEV theorem

**Fisher-Tippett-Gnedenko**: if you divide your data into blocks of size $n$ and take the maximum of each block, the *normalised* distribution of these maxima converges (as $n \to \infty$) to a **Generalised Extreme Value (GEV)** distribution:

$$
G(x) = \exp\left( -[1 + \xi (x - \mu)/\sigma]^{-1/\xi} \right)
$$

with three parameters: location $\mu$, scale $\sigma$, shape $\xi$.

- $\xi > 0$: heavy tail (Fréchet — what you see in finance).
- $\xi = 0$: light tail (Gumbel — exponential decay).
- $\xi < 0$: bounded tail (Weibull — has an upper limit).

In code:

```python
import numpy as np
from scipy.stats import genextreme

# 30 years of daily returns → 30 yearly maxima
yearly_max = -returns.resample("Y").min()       # losses, taken as positive
shape, loc, scale = genextreme.fit(yearly_max)
print(f"GEV: xi={-shape:.3f}, mu={loc:.4f}, sigma={scale:.4f}")
```

(Note `scipy.stats.genextreme`'s `c` argument is the *negative* of the standard $\xi$.)

You now have a distribution over yearly maximum loss. The 1-in-50-year loss:

```python
return_50yr = genextreme.ppf(1 - 1/50, shape, loc=loc, scale=scale)
```

This is the tail quantile your sample is too short to estimate empirically.

The GEV approach is statistically clean but **wasteful** — you discarded all but the maxima of each block. The next approach uses more of the data.

## Peaks-over-threshold — the GPD theorem

**Pickands-Balkema-de Haan**: pick a high threshold $u$. The distribution of *exceedances* $(X - u)$, conditional on $X > u$, converges to a **Generalised Pareto Distribution (GPD)**:

$$
H(y) = 1 - \left(1 + \xi y / \sigma\right)^{-1/\xi}
$$

with shape $\xi$ and scale $\sigma$.

```python
from scipy.stats import genpareto

# Threshold u: the 95th percentile of losses
losses = -returns
u = np.quantile(losses, 0.95)
exceedances = losses[losses > u] - u

shape, loc, scale = genpareto.fit(exceedances, floc=0)   # force loc=0
print(f"GPD: xi={shape:.3f}, sigma={scale:.4f}")
```

The 99.9% VaR, expressed as a tail probability $p$ for the exceedance level:

```python
p_exceed = (losses > u).mean()                       # probability of exceeding u
target_quantile = (1 - 0.999) / p_exceed             # rescale to conditional tail
var_999 = u + genpareto.ppf(1 - target_quantile, shape, scale=scale)
print(f"99.9% daily VaR: {var_999:.2%}")
```

Compare to the empirical quantile — for a sample of 5,000 days, the empirical 99.9% is the 5th worst loss, which is wildly noisy. The GPD estimate is much more stable.

## Picking the threshold

Too low → the GPD theorem's asymptotic isn't kicked in yet, and you'll have bias.
Too high → too few exceedances, large variance.

The standard practice: a **mean residual life** plot.

```python
import matplotlib.pyplot as plt

thresholds = np.quantile(losses, np.linspace(0.80, 0.99, 30))
mean_excess = [losses[losses > u].mean() - u for u in thresholds]
plt.plot(thresholds, mean_excess)
plt.xlabel("threshold"); plt.ylabel("mean excess")
plt.show()
```

If the GPD assumption holds, mean excess is linear in $u$ above some threshold. Pick the $u$ where the plot becomes approximately linear. Honest practice: there's judgment involved — pick a range and check sensitivity.

## Tail index estimation

The shape parameter $\xi$ is the **tail index**. A few rules of thumb for daily equity returns:

- Index returns (SPY, SPX): $\xi \approx 0.2 - 0.3$ for losses.
- Individual stocks: typically $\xi \approx 0.3 - 0.5$.
- Cryptocurrencies: $\xi$ often > 0.5 — genuinely heavy tails.
- Strategy P&L: depends heavily on the strategy; carry trades and short-vol typically high $\xi$.

A $\xi > 0.5$ means the **theoretical variance is infinite**, and standard mean-variance optimisation is misleading. Use CVaR, not standard deviation, for risk targeting.

## Conditional VaR (CVaR / expected shortfall)

The expected loss *given* that you've exceeded VaR. For the GPD parametrisation:

$$
\text{CVaR}_\alpha = \frac{\text{VaR}_\alpha + \sigma - \xi u}{1 - \xi}
$$

(valid for $\xi < 1$).

```python
def gpd_cvar(alpha, u, shape, scale, p_exceed):
    var = u + scale / shape * (((1 - alpha) / p_exceed) ** (-shape) - 1)
    return (var + scale - shape * u) / (1 - shape)

print(f"99% CVaR: {gpd_cvar(0.99, u, shape, scale, p_exceed):.2%}")
```

CVaR is the right risk number to set position sizes against. VaR tells you "this much or less, 99% of the time"; CVaR tells you "*when* it gets bad, this is the expected loss" — which is what actually blows up accounts.

## A worked example: tail-risk-targeted position sizing

```python
def position_size(strategy_returns, capital, cvar_budget, alpha=0.99):
    """Size the strategy so its 99% CVaR equals the budget."""
    losses = -strategy_returns
    u = np.quantile(losses, 0.95)
    excess = losses[losses > u] - u
    shape, _, scale = genpareto.fit(excess, floc=0)
    p_exceed = (losses > u).mean()
    cvar = gpd_cvar(alpha, u, shape, scale, p_exceed)
    return cvar_budget / cvar * capital
```

This sizes the strategy so a 1-in-100-day loss equals your tolerance. Far more robust than vol-targeting when the underlying distribution is fat-tailed.

## The crisis-of-2020 problem

In March 2020, daily SPY losses hit -10% (vs. a historical 99% VaR of around -3.5%). Your in-sample EVT fit didn't include that day. After it, your fit gets larger — but it never *predicted* the specific event.

EVT does **not** predict crises. It quantifies the tail you can already see and extrapolates a bit. It does not invent new ones. The honest interpretation: "the model gives an honest tail given what's happened; the tail can be heavier than what's happened."

For genuinely worst-case scenarios, **stress test with constructed shocks** — what if returns are 1.5× the worst historical week? — and size such that you survive.

## Pitfalls

!!! warning "Non-stationary tail"
    Vol regimes change. A GPD fit on 2017's quiet returns badly underestimates 2020's tail. Refit periodically and use the higher of recent vs all-history.

!!! warning "Independence assumption"
    The EVT theorems assume independent observations. Returns have weak autocorrelation but strong volatility clustering. Pre-filter with a GARCH (Module 7) and fit EVT to standardised residuals if you need precision.

!!! warning "Fitting to a single side"
    Losses and gains can have different tails. Always fit them separately. For risk, only losses matter.

!!! warning "Sample-size dependence of $\xi$"
    Tail estimates have huge variance from sample size alone. Bootstrap CIs on $\xi$ are sobering.

## Bottom line

For tail risk:

- **Use GPD for VaR and CVaR**, with threshold selected from a mean-residual-life plot.
- **Report CVaR, not VaR**, for sizing decisions.
- **Refit periodically** — every 6–12 months for long-horizon strategies.
- **Stress-test on constructed shocks** that go *beyond* the fitted tail. The world is bigger than your sample.

## End of Module 6

You now have the statistical machinery to evaluate strategies honestly. The next module — **Time series** — gives you the temporal-structure tools (stationarity tests, GARCH, Kalman filters, particle filters, change-point detection) that show up in every quant model.

Continue to **[Module 7 — Time Series](../07-time-series/index.md)**.
