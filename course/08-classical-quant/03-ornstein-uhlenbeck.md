# Ornstein-Uhlenbeck and stationary spreads

The Ornstein-Uhlenbeck (OU) process is the canonical continuous-time mean-reverting process. It's the right model whenever you have a stationary series that returns to some long-run mean — pairs trade spreads, the term structure of interest rates, VIX, even some commodity prices. Knowing OU calibration and the half-life formula gives you a quick, principled way to size and time trades on any mean-reverting series.

## The model

$$
dX_t = \theta (\mu - X_t)\, dt + \sigma\, dW_t
$$

Three parameters:

- $\mu$ — the long-run mean.
- $\theta > 0$ — the speed of mean reversion. Higher = snaps back faster.
- $\sigma$ — the diffusion (noise).

The conditional distribution given $X_0$:

$$
X_t \mid X_0 \sim \mathcal{N}\!\left( \mu + (X_0 - \mu) e^{-\theta t},\; \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t}) \right)
$$

Two useful quantities:

- **Half-life of deviations from $\mu$**: $\ln 2 / \theta$.
- **Stationary variance**: $\sigma^2 / (2\theta)$.

## Calibration in discrete time

Discretise OU at intervals $\Delta t = 1$ (one trading day, one bar, whatever). The OU dynamics become an AR(1):

$$
X_t = a + b X_{t-1} + \varepsilon_t
$$

with $b = e^{-\theta}$, $a = \mu (1 - b)$, $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$.

Estimate $a, b, \sigma_\varepsilon^2$ by OLS:

```python
import numpy as np

def calibrate_ou(x: np.ndarray) -> dict:
    """Calibrate OU parameters from a discrete series of one-step-spaced observations."""
    x = np.asarray(x, dtype=float)
    x_lag, x_now = x[:-1], x[1:]
    A = np.vstack([np.ones_like(x_lag), x_lag]).T
    coef, *_ = np.linalg.lstsq(A, x_now, rcond=None)
    a, b = coef
    resid = x_now - (a + b * x_lag)
    sigma_eps = resid.std(ddof=2)
    if b >= 1.0:                                           # not mean-reverting
        return dict(theta=0.0, mu=np.nan, sigma=np.nan,
                    half_life=np.inf, stationary=False)
    theta = -np.log(b)
    mu = a / (1 - b)
    sigma = sigma_eps * np.sqrt(2 * theta / (1 - b ** 2))
    return dict(theta=theta, mu=mu, sigma=sigma,
                half_life=np.log(2) / theta, stationary=True)
```

Plug in any series. The `half_life` tells you the natural timescale.

## A worked example: OU on the SPY-IVV spread

SPY and IVV are essentially the same index tracker. Their price difference should mean-revert toward zero.

```python
from engine.data import YFinanceFeed
import pandas as pd

feed = YFinanceFeed()
spy = feed.bars("SPY", "2022-01-01", "2024-12-31")["close"]
ivv = feed.bars("IVV", "2022-01-01", "2024-12-31")["close"]
df = pd.concat({"SPY": spy, "IVV": ivv}, axis=1).dropna()

spread = df["SPY"] - df["IVV"]
params = calibrate_ou(spread.values)
print(params)
# typical: theta ~ 0.1, half_life ~ 7 days, mu ~ 0
```

A half-life of about a week is consistent with arbitrageurs closing the gap over a few sessions.

## Trading the spread

Once you have a calibrated OU, the trading rule writes itself:

```python
def ou_signal(x: pd.Series, lookback: int = 252, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """Rolling OU; long when spread is `entry_z` SD below mean, short when above."""
    out = pd.Series(0.0, index=x.index)
    for i in range(lookback, len(x)):
        window = x.iloc[i - lookback : i].values
        p = calibrate_ou(window)
        if not p["stationary"] or p["half_life"] > lookback:
            continue
        # standardised deviation
        z = (x.iloc[i] - p["mu"]) / (p["sigma"] / np.sqrt(2 * p["theta"]))
        if abs(z) > entry_z:
            out.iloc[i] = -np.sign(z)            # long if below mean, short if above
        elif abs(z) < exit_z and out.iloc[i - 1] != 0:
            out.iloc[i] = 0                       # flat when close to mean
        else:
            out.iloc[i] = out.iloc[i - 1]         # carry position
    return out.shift(1)                            # tradable next day
```

Three rules of thumb:

1. **Entry at 2 sigma**, exit at 0.5 sigma. Anything more aggressive and transaction costs eat the trades.
2. **Hold for at most a few half-lives**, then re-evaluate. A trade that doesn't converge in 2-3 half-lives is probably a regime break.
3. **Size by the expected return per unit risk**, not by the dollar deviation.

## The trader's mental model

When you see a series and someone says "it'll mean-revert", ask three questions:

1. **Is it stationary?** ADF / KPSS (Module 7). If not, your "trade" is just betting on a random walk to do something.
2. **What is the half-life?** $\ln 2 / \theta$. If it's 6 months, that's the natural horizon — you're not "scalping" anything.
3. **What is the conditional one-sigma move?** $\sigma / \sqrt{2\theta}$ from full convergence. That sets the size you need to make a meaningful return.

If the half-life is days and the sigma is 1% of capital, you have a great mean-reversion target. If the half-life is months and the sigma is 0.1%, the expected Sharpe is small relative to costs.

## Generalisations — when single-asset OU breaks

- **Time-varying mean.** Use a rolling-window calibration or a Kalman filter to estimate a slowly-changing $\mu$.
- **Regime switches.** Mean-reversion strength changes — fit per regime with an HMM (Module 16).
- **Long-memory deviations.** If deviations show clustering, fit an OU with stochastic volatility or use the Heston process (Module 15).
- **Multivariate spreads.** A linear combination of $k$ assets — use Johansen cointegration (next chapter) to find the cointegrating vector.

## Pitfalls

!!! warning "OU fit on non-stationary data is meaningless"
    Always check stationarity first. A non-stationary series fit to OU returns $b \approx 1$, which the code above flags with `stationary=False`. But subtler cases (slow drift in the mean) give finite-but-wrong $\theta$.

!!! warning "The half-life shrinks with regime breaks"
    A pair that has historically mean-reverted in 7 days may suddenly take 30 days during a regime change. Re-calibrate often.

!!! warning "Reversal isn't free"
    Trading the spread requires shorting one leg. Costs include borrow, hard-to-borrow specials, and asymmetric execution. The OU calibration tells you the *gross* opportunity; subtract realistic frictions before you size.

## Bottom line

OU calibration in one line gives you the half-life and the conditional stationary scale of any candidate mean-reverting series. That's the right starting point for any reversal trade.

For single-asset reversal: OU on the residual after removing a trend (e.g. SPY's deviation from its 200-day MA).

For pairs: OU on the cointegrated spread (next chapter).

Continue to **[Cointegration and pairs trading](04-cointegration.md)**.
