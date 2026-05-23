# Hawkes processes for order-flow and news clusters

A Poisson process assumes events arrive independently — the chance of the next one is the same regardless of what just happened. Most financial event streams are *not* Poisson. A news headline triggers another news headline; a buy at the market triggers more buys; volatility clusters in time.

The **Hawkes process** (Hawkes 1971) is the simplest model that captures **self-excitation**: each event raises the probability of more events for a while afterward. For modelling order flow clustering, news-event clustering, and the propagation of shocks across markets, Hawkes is the right framework.

## The math, briefly

A Hawkes process has a conditional intensity:

$$
\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}
$$

Three parameters:

- $\mu$ — baseline intensity (events per unit time when no recent events).
- $\alpha$ — jump in intensity caused by each past event.
- $\beta$ — decay rate of that jump.

After an event at $t_i$, the intensity jumps by $\alpha$ and decays exponentially with rate $\beta$. The expected number of "child" events from one parent is $\alpha / \beta$ — the **branching ratio**. If it's < 1, the process is stable; if > 1, it explodes.

For the typical use case, the branching ratio is between 0.3 and 0.8 — events are correlated but not runaway.

## Multivariate Hawkes — cross-excitation

For multiple event types, each event of type $j$ excites future events of type $k$ at a rate $\alpha_{kj}$. The matrix $A = [\alpha_{kj}]$ encodes the cross-excitation structure:

$$
\lambda_k(t) = \mu_k + \sum_j \sum_{t_i^j < t} \alpha_{kj} e^{-\beta_{kj} (t - t_i^j)}
$$

For order flow, the natural decomposition: events are (buy, sell). $\alpha_\text{BB}$ = "buys trigger more buys"; $\alpha_\text{SS}$ = "sells trigger more sells"; $\alpha_\text{BS}, \alpha_\text{SB}$ = cross-side triggering.

For news flow, events are (positive, negative, neutral) and the cross-matrix captures whether negative news triggers more negative news, etc.

## Fitting Hawkes

For an observed event sequence $t_1, ..., t_n$ in $[0, T]$, the log-likelihood is:

$$
\log L = \sum_i \log \lambda(t_i) - \int_0^T \lambda(s) \, ds
$$

The integral has a closed form for exponential kernels:

```python
import numpy as np
from scipy.optimize import minimize


def hawkes_loglik(params, events, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10
    n = len(events)
    # Intensity at each event time
    intensities = np.zeros(n)
    excitation = 0.0
    last_t = 0.0
    for i, t in enumerate(events):
        excitation = excitation * np.exp(-beta * (t - last_t)) + alpha
        intensities[i] = mu + (excitation - alpha)        # before counting this event
        last_t = t
    # Compensator (the integral)
    comp_baseline = mu * T
    comp_excitation = (alpha / beta) * sum(1 - np.exp(-beta * (T - t)) for t in events)
    return -(np.sum(np.log(mu + np.cumsum([0] + [alpha * np.exp(-beta * (events[i] - events[i-1]))
                                                  for i in range(1, n)])))
             - comp_baseline - comp_excitation)


def fit_hawkes(events: np.ndarray, T: float):
    """Maximum-likelihood Hawkes fit. Events are sorted timestamps."""
    x0 = [len(events) / T * 0.5, 1.0, 2.0]    # crude initial guess
    bounds = [(1e-6, None), (1e-6, None), (1e-3, None)]
    res = minimize(lambda p: hawkes_loglik(p, events, T), x0,
                    method="L-BFGS-B", bounds=bounds)
    return dict(zip(["mu", "alpha", "beta"], res.x.tolist()))
```

The implementation above is a pedagogical version; for production use the `tick` library (Bacry et al.) which has C++-backed multivariate Hawkes fitters orders of magnitude faster.

## Where to use Hawkes in trading

### Order flow microstructure

Fit a 2D Hawkes (buy + sell) to a few hours of trade data. The branching ratio tells you how clustered the flow is. **High branching ratio = trending; low = noisy reverting**. Trades within high-cluster periods are more likely to have followers (favourable for momentum entries); low-cluster periods are mean-reverting.

### News event clustering

Fit a Hawkes on news arrival times (per asset). When you see a news event, the model predicts the probability of more news in the next hour. **High predicted intensity = elevated short-term realised vol**. Useful for sizing the next hour's positions.

### Cross-asset shock propagation

A multivariate Hawkes on market opens / closes across regions. **The off-diagonal entries** ($\alpha_\text{Asia → US}$ etc.) tell you how shocks propagate. High Asia-to-US implies that a big Asian close-to-open move will excite a bigger US morning move than usual.

## A worked example: simulating + fitting

```python
import numpy as np

def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed: int = 0):
    """Ogata's thinning algorithm."""
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    intensity = mu
    while t < T:
        # Upper bound for intensity in next interval
        m = mu + sum(alpha * np.exp(-beta * (t - ti)) for ti in events) + alpha
        dt = rng.exponential(1 / m)
        t += dt
        if t > T:
            break
        # Acceptance probability
        lam_t = mu + sum(alpha * np.exp(-beta * (t - ti)) for ti in events)
        if rng.random() < lam_t / m:
            events.append(t)
    return np.array(events)


# True parameters
true_mu, true_alpha, true_beta = 0.5, 0.8, 2.0
events = simulate_hawkes(true_mu, true_alpha, true_beta, T=1000)
print(f"Number of events: {len(events)}")
fit = fit_hawkes(events, T=1000)
print(f"Recovered: {fit}")
print(f"True:      mu={true_mu}, alpha={true_alpha}, beta={true_beta}")
```

Recovery is typically within 10% of true. For real fitting on order-flow data, much faster algorithms (EM, ADM4) are preferred.

## Hawkes-based scanner: "is this stock about to spike?"

```python
import pandas as pd

def hawkes_intensity_now(recent_events: list[float], asof: float,
                          mu: float, alpha: float, beta: float) -> float:
    """Compute current Hawkes intensity given fitted params and recent events."""
    return mu + sum(alpha * np.exp(-beta * (asof - t)) for t in recent_events)


def vol_spike_scanner(symbols: list[str], recent_news_by_symbol: dict, asof: pd.Timestamp,
                       fitted_params_by_symbol: dict, threshold: float = 3.0) -> pd.DataFrame:
    """Flag symbols where Hawkes intensity is unusually high — implying near-term vol."""
    rows = []
    for sym in symbols:
        if sym not in fitted_params_by_symbol:
            continue
        params = fitted_params_by_symbol[sym]
        events = recent_news_by_symbol.get(sym, [])
        intensity = hawkes_intensity_now(events, asof.timestamp(), **params)
        baseline = params["mu"]
        if intensity / baseline > threshold:
            rows.append({"symbol": sym, "intensity": intensity, "baseline": baseline,
                         "ratio": intensity / baseline})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)
```

A symbol whose current Hawkes intensity is 3-10× its baseline is in a self-exciting cluster — short-term realised vol is likely to be high. Use as a sizing input or as a "skip" filter for mean-reversion strategies.

## Pitfalls

!!! warning "Branching ratio > 1"
    If your fitted $\alpha / \beta > 1$, the process is explosive — events compound forever. This often indicates either bad data (duplicate timestamps, recording artefacts) or that the model class is too simple for the data.

!!! warning "Heavy-tailed kernels"
    Exponential decay is the simplest case but real markets often show power-law decay (events excite more events for hours, not seconds). Use sum-of-exponentials or non-parametric kernel estimators.

!!! warning "Non-stationary Hawkes**
    Hawkes assumes the parameters don't change with regime. They do. Refit on rolling windows or use time-varying-baseline Hawkes models.

!!! warning "Fitting on small samples"
    With <100 events, Hawkes fits are very noisy. The branching ratio is the most stable; baseline rate the least.

## Bottom line

Hawkes processes capture **clustering and self-excitation** in event streams. They're the right tool for:

- Order-flow microstructure analysis (clustered trading regimes).
- News-arrival rates and short-term vol prediction.
- Cross-asset / cross-region shock propagation.

For production use, use the `tick` library. Hawkes won't single-handedly produce a strategy, but a Hawkes-derived feature ("current intensity / baseline") slotted into a larger ML model often adds 0.05-0.1 Sharpe to short-horizon predictors.

Continue to **[Topological data analysis on rolling correlation matrices](02-tda.md)**.
