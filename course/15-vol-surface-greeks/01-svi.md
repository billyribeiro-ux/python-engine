# SVI and SSVI — the workhorse smile parameterisation

The market's implied volatility per strike, for a fixed expiry, traces out a curve (typically a smile or a skew). You can store the curve as a table of points and interpolate, but a *parametric* fit gives you smoother values, robust extrapolation, and the ability to enforce no-arbitrage. SVI ("Stochastic Volatility Inspired", Gatheral 2004) is the industry-standard parameterisation.

## The raw SVI formula

For log-moneyness $k = \log(K / S)$ and total implied variance $w = \sigma^2 T$:

$$
w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)
$$

Five parameters per slice: $a, b, \rho, m, \sigma$.

- $a$ controls the level (minimum variance).
- $b$ controls the wing growth rate.
- $\rho \in (-1, 1)$ controls the slope at the centre.
- $m$ shifts the smile centre.
- $\sigma$ controls the smoothness at the centre.

Constraints for no-static-arbitrage on a single slice:

$$
b \geq 0, \quad |\rho| \leq 1, \quad a + b \sigma \sqrt{1 - \rho^2} \geq 0
$$

## A fitter from scratch

```python
import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_slice(k_obs: np.ndarray, w_obs: np.ndarray, weights: np.ndarray | None = None):
    """Least-squares fit of one slice's SVI parameters.

    k_obs: log-moneyness array.
    w_obs: observed total implied variance (sigma^2 * T) at those strikes.
    """
    if weights is None:
        weights = np.ones_like(w_obs)

    def loss(params):
        a, b, rho, m, sigma = params
        w_fit = svi_total_variance(k_obs, a, b, rho, m, sigma)
        return float(np.sum(weights * (w_fit - w_obs) ** 2))

    # Initial guess (Gatheral's heuristic)
    x0 = [np.mean(w_obs), 0.1, -0.5, 0.0, 0.1]
    bounds = [(0.0, None), (0.0, None), (-0.99, 0.99), (-1.0, 1.0), (1e-4, None)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return dict(zip(["a", "b", "rho", "m", "sigma"], res.x.tolist()))
```

A typical SPY 30-day smile fits to RMSE ~0.001 in implied variance with this loss. For market-making accuracy, weighted least squares (down-weighting illiquid strikes) is better.

## Why SVI works

Three nice properties:

1. **The wings are asymptotically linear** in $|k|$ — matches Lee's no-arbitrage bound on smiles.
2. **The fit is smooth, monotone, and bounded** — meaningful for risk computation.
3. **No-arbitrage conditions are easy to check** as algebraic constraints on the 5 parameters.

The price you pay: 5 parameters per slice means a 12-slice surface has 60 parameters. SSVI (next section) reduces this by linking the slices.

## SSVI — Surface SVI

Gatheral-Jacquier (2013). Constrains the surface so that the slices share structure across $T$:

$$
w(k, \theta_T) = \frac{\theta_T}{2} \left( 1 + \rho \phi(\theta_T) k + \sqrt{(\phi(\theta_T) k + \rho)^2 + (1 - \rho^2)} \right)
$$

where $\theta_T$ is the at-the-money total variance at expiry $T$, and $\phi$ is a function (typically a power-law $\phi(\theta) = \eta / \theta^\gamma$).

A whole surface — many expiries, full smile per expiry — is described by:

- The ATM total-variance curve $\theta_T$ (a function of T; often a few points or a small parametric form).
- A constant $\rho$ across the surface.
- Two parameters $(\eta, \gamma)$ for the $\phi$ power law.

That's it. A complete SPY surface fits to ~10 parameters total. Calendar-arbitrage-free conditions are automatic with the right choice of $\phi$.

## A worked example: fit, plot, sanity-check

```python
import numpy as np
import matplotlib.pyplot as plt

# Pretend we observe a smile (k = log-moneyness, w = total iv variance)
k = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
T = 30 / 365
sigma_obs = np.array([0.22, 0.20, 0.185, 0.18, 0.185, 0.20, 0.22])
w = sigma_obs ** 2 * T

params = fit_svi_slice(k, w)
print(params)

# Densely sample the fitted smile
k_dense = np.linspace(-0.20, 0.20, 200)
w_fit = svi_total_variance(k_dense, **params)
sigma_fit = np.sqrt(w_fit / T)

# Sanity-check: implied density should be non-negative (no butterfly arb)
# d^2 C / dK^2 >= 0 for valid surfaces
```

For production use, fit the parameters with cross-validation on out-of-sample dates to avoid overfitting noise.

## No-arbitrage enforcement

A vol surface has three types of arbitrage to avoid:

1. **Calendar arbitrage** — total variance must be non-decreasing in $T$ for each strike. SSVI with the right $\phi$ enforces this.
2. **Butterfly arbitrage** — implied call price as a function of strike must be convex. Checked numerically by computing the second derivative of $C(K)$.
3. **Static arbitrage on a single slice** — Lee's conditions on the wing slopes. Auto-satisfied by the SVI constraints above.

For production, after fitting, check butterfly arbitrage explicitly:

```python
def check_butterfly(strikes, call_prices):
    """The call price must be convex in strike. Returns the worst violation."""
    pp = np.diff(np.diff(call_prices))
    return pp.min()                    # should be >= 0 for no-arb
```

If the surface fits the data but admits butterfly arbitrage, lower the SVI's `b` parameter (less aggressive wings) and refit.

## Where SVI shines

- **Equity index smiles** — SPY, SPX, NDX. SVI captures the asymmetric skew very well.
- **Vol-arbitrage strategies** — fit SVI to the market; identify strikes where the market price deviates from the fit; trade the dislocation.
- **Risk computation across strike** — interpolate consistent vols for any strike.

## Where it doesn't

- **FX smiles** — often more symmetric and bimodal; SABR is more natural.
- **Rates / swaption surfaces** — SABR is standard.
- **Single-stock with discrete events** — earnings induce abnormal smile shapes that 5 parameters can't fit. Use a mixture model (SVI per regime).

## Pitfalls

!!! warning "Fit to bid-ask mid, weight by inverse spread"
    Trying to fit illiquid wings to last-trade IVs gives garbage. Use mids and down-weight wide-spread strikes.

!!! warning "Per-slice independent fits give non-arbitrage-free surfaces"
    Five-parameter-per-slice SVI on 12 expiries with independent fits often has calendar arbitrage. Either move to SSVI or apply post-fit smoothing across expiries.

!!! warning "The smile and the term structure interact"
    The 7-day smile is dramatically more extreme than the 90-day smile. A single SVI shape doesn't fit; SSVI's $\theta_T$ scaling handles this.

!!! warning "SVI extrapolation past the observed strikes is fragile"
    The wings extrapolate linearly in log-moneyness, which is fine — but the slope is determined by the highest-strike data point. If that's noisy, the entire wing is wrong.

## Bottom line

For options trading:

- **Per-slice SVI** for fast, robust smile fitting.
- **SSVI** for whole-surface fitting with calendar-arbitrage-free guarantees.
- **Always check butterfly arbitrage** post-fit.
- **For frontier-track dislocation scanners** (Module 18), the SVI fit is the residual signal — strikes where the market price departs from the smooth surface are candidates for trades.

Continue to **[SABR — the rates / FX skew model](02-sabr.md)**.
