# VaR and CVaR — honestly

Value-at-Risk (VaR) is the loss level you'd expect to be exceeded at some probability (say, 5%). Conditional Value-at-Risk (CVaR, also "Expected Shortfall") is the *average* loss given that you exceed VaR. Both are foundational; both are commonly misused.

This chapter is the practical guide — what to use, what to avoid, how to compute them robustly with the tools from Module 6 chapter 5.

## The definitions

For loss random variable $L$ (positive = loss):

$$
\text{VaR}_\alpha(L) = \inf\{l : P(L \leq l) \geq \alpha\}
$$

$$
\text{CVaR}_\alpha(L) = \mathbb{E}[L | L > \text{VaR}_\alpha(L)]
$$

VaR is the $\alpha$-quantile of the loss distribution. CVaR is the average of losses in the worst $1 - \alpha$ tail.

## Three ways to compute VaR

### Historical simulation

Take your loss series; report the $\alpha$-quantile.

```python
import numpy as np
import pandas as pd

def historical_var(pnl: pd.Series, alpha: float = 0.95) -> float:
    losses = -pnl
    return float(np.quantile(losses, alpha))


def historical_cvar(pnl: pd.Series, alpha: float = 0.95) -> float:
    losses = -pnl
    var = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())
```

**Pros**: assumption-free, intuitive.
**Cons**: only as good as the history; misses tail events that haven't happened yet.

### Parametric (Gaussian)

Assume losses are Gaussian with mean $\mu$ and standard deviation $\sigma$:

$$
\text{VaR}_\alpha = \mu + \sigma \cdot \Phi^{-1}(\alpha)
$$

```python
from scipy.stats import norm

def gaussian_var(pnl: pd.Series, alpha: float = 0.95) -> float:
    losses = -pnl
    return float(losses.mean() + losses.std() * norm.ppf(alpha))
```

**Pros**: closed-form, fast.
**Cons**: **wrong for financial data**. Real losses are fat-tailed; Gaussian VaR systematically under-estimates the tail.

### EVT (GPD on exceedances)

From Module 6 chapter 5: fit a Generalised Pareto Distribution to exceedances over a high threshold; compute VaR / CVaR from the fitted GPD.

```python
from scipy.stats import genpareto

def evt_var_cvar(pnl: pd.Series, alpha: float = 0.99, threshold_pct: float = 0.95) -> dict:
    losses = -pnl
    u = np.quantile(losses, threshold_pct)
    excess = losses[losses > u] - u
    p_exceed = (losses > u).mean()
    shape, _, scale = genpareto.fit(excess, floc=0)
    var = u + (scale / shape) * (((1 - alpha) / p_exceed) ** (-shape) - 1)
    cvar = (var + scale - shape * u) / (1 - shape)
    return {"var": float(var), "cvar": float(cvar), "shape": float(shape),
            "scale": float(scale), "threshold": float(u)}
```

**Pros**: honest tail; handles fat tails correctly.
**Cons**: requires enough exceedances (~30+) for stable fit.

## Compare for a typical strategy

```python
pnl = ...    # your strategy's daily returns

print(f"Historical 99% VaR: {historical_var(pnl, 0.99):.4f}")
print(f"Gaussian 99% VaR:   {gaussian_var(pnl, 0.99):.4f}")
print(f"EVT 99% VaR:        {evt_var_cvar(pnl, 0.99)['var']:.4f}")
print(f"EVT 99% CVaR:       {evt_var_cvar(pnl, 0.99)['cvar']:.4f}")
```

Typical pattern: Gaussian << Historical < EVT. The Gaussian assumption hides the tail; EVT honestly extrapolates it.

## Why CVaR is the right risk metric for sizing

VaR has a flaw: it's not **coherent**. Adding two positions can produce a portfolio whose VaR is *higher* than the sum of the parts' VaRs. That's mathematically pathological.

CVaR is coherent and subadditive: the CVaR of a portfolio is at most the sum of its components' CVaRs. **Diversification reduces CVaR** — exactly what you'd want from a risk metric.

For position sizing:

```python
def cvar_sized(strategy_returns: pd.Series, capital: float, cvar_budget: float,
                alpha: float = 0.99) -> float:
    """Size so that the strategy's α-CVaR equals the budget fraction of capital."""
    cvar = evt_var_cvar(strategy_returns, alpha)["cvar"]
    return capital * cvar_budget / cvar
```

## Backtesting VaR forecasts

Once you have a VaR model, backtest it. The standard test: **Kupiec's POF (Proportion of Failures)** test.

For an $\alpha$-VaR model forecasting daily VaR over $T$ days, count exceedances $K$ (days the loss exceeded the forecast VaR). Under the null $K \sim \text{Binomial}(T, 1 - \alpha)$.

```python
from scipy.stats import binom


def kupiec_pof_test(realised_losses: pd.Series, forecast_vars: pd.Series, alpha: float = 0.99):
    """Test whether observed exceedance rate matches the model's α."""
    exceedances = (realised_losses > forecast_vars).sum()
    T = len(realised_losses)
    expected = T * (1 - alpha)
    # Likelihood ratio test
    ratio_h0 = ((1 - alpha) ** exceedances) * (alpha ** (T - exceedances))
    rate_obs = exceedances / T
    ratio_h1 = (rate_obs ** exceedances) * ((1 - rate_obs) ** (T - exceedances))
    lr = -2 * np.log(ratio_h0 / ratio_h1) if ratio_h1 > 0 else float("inf")
    from scipy.stats import chi2
    p = 1 - chi2.cdf(lr, df=1)
    return {"exceedances": int(exceedances), "expected": float(expected),
            "lr_stat": float(lr), "p_value": float(p)}
```

p < 0.05 → reject the model. Either too many exceedances (model under-forecasts risk) or too few (over-forecasts).

A well-calibrated 99% VaR model on 1000 days should produce ~10 exceedances. Too many (say, 25) means real risk is bigger than the model says; too few (say, 2) means the model is too conservative.

## Conditional coverage

Kupiec tests *marginal* coverage. The stricter **Christoffersen test** also checks that exceedances are temporally independent — a real model should have rare, random exceedances, not clustered.

```python
def christoffersen_test(realised_losses: pd.Series, forecast_vars: pd.Series, alpha: float = 0.99):
    """Test independence of exceedances."""
    exceeds = (realised_losses > forecast_vars).astype(int).values
    # Transition counts
    n00 = sum((exceeds[:-1] == 0) & (exceeds[1:] == 0))
    n01 = sum((exceeds[:-1] == 0) & (exceeds[1:] == 1))
    n10 = sum((exceeds[:-1] == 1) & (exceeds[1:] == 0))
    n11 = sum((exceeds[:-1] == 1) & (exceeds[1:] == 1))
    pi_01 = n01 / (n00 + n01) if (n00 + n01) else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) else 0
    pi = (n01 + n11) / len(exceeds)
    # LR for independence
    if pi == 0 or pi == 1 or pi_01 == 0 or pi_11 == 0:
        return {"stat": float("nan"), "p_value": float("nan")}
    lr = -2 * np.log((1-pi)**(n00+n10) * pi**(n01+n11)) + \
          2 * np.log((1-pi_01)**n00 * pi_01**n01 * (1-pi_11)**n10 * pi_11**n11)
    from scipy.stats import chi2
    p = 1 - chi2.cdf(lr, df=1)
    return {"stat": float(lr), "p_value": float(p)}
```

Failing this means your exceedances cluster — the model isn't capturing vol regimes. Add GARCH-conditional vol forecasting (Module 7 chapter 2).

## Pitfalls

!!! warning "Reporting Gaussian VaR for fat-tailed returns"
    A common practice in retail; produces misleading risk numbers. Banks have long since switched to historical or EVT.

!!! warning "Confusing position-level VaR with portfolio-level VaR"
    Position-level VaRs don't add. Compute portfolio VaR on the joint return distribution, accounting for correlations.

!!! warning "VaR for non-linear positions"
    Options have payoffs non-linear in the underlying. Linear VaR (assume options are equivalent to delta shares) understates risk. Use Monte Carlo or scenario-based VaR.

!!! warning "Stale data + low exceedance counts**
    Recent data has few extreme events. EVT requires meaningful exceedance counts (~30+) for stable fits. Use long windows or pool across symbols.

## Bottom line

For risk measurement:

- **Use CVaR**, not VaR, as the headline metric — it's coherent.
- **Use EVT** (GPD on exceedances), not Gaussian, for tail estimation.
- **Backtest with Kupiec + Christoffersen** — confirm both rate and independence.
- **Refit periodically** — vol regimes change.

For sizing, CVaR-targeting (Module 8 chapter 6) handles fat tails honestly.

Continue to **[The Kelly trap](03-kelly-trap.md)**.
