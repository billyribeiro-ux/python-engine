# Change-point detection

Markets change regime. The strategy that worked in low-vol grind dies in a vol spike; the carry trade that was bulletproof for three years unwinds in a week. Change-point detection is the family of statistical methods that ask "has the distribution generating my data just changed?" and answers with both a yes/no and a *where*.

This chapter covers the three you'll actually use: CUSUM (the workhorse), Bayesian online change-point detection (the principled real-time tool), and the `ruptures` package for offline multi-point segmentation.

## CUSUM — cumulative sum

A change-point detector you can implement in 10 lines. Tracks the cumulative deviation from a reference mean and signals when it exceeds a threshold.

```python
import numpy as np

def cusum(x: np.ndarray, target: float, k: float, h: float) -> np.ndarray:
    """Return an array of {-1, 0, +1} where +1/-1 mark an upper/lower change-point."""
    s_pos = s_neg = 0.0
    out = np.zeros(len(x), dtype=int)
    for t, xt in enumerate(x):
        s_pos = max(0.0, s_pos + xt - target - k)
        s_neg = min(0.0, s_neg + xt - target + k)
        if s_pos > h:
            out[t] = +1; s_pos = 0
        elif s_neg < -h:
            out[t] = -1; s_neg = 0
    return out
```

Three parameters:

- `target` — the reference mean (often zero for centred returns).
- `k` — the "slack" — small deviations are absorbed before they accumulate.
- `h` — the alert threshold.

Tune `k` and `h` to control the trade-off between sensitivity and false positives. Typical setup for return series: `k = sigma`, `h = 3 * sigma`.

Practical use: a CUSUM on the running squared-return mean detects vol regime shifts; a CUSUM on the residual of your model detects when the model has stopped fitting.

## Bayesian Online Change-Point Detection (BOCPD)

Adams & MacKay (2007). The principled approach: maintain a posterior over the *run length* — how many bars have elapsed since the last change-point — and update it at each new observation.

The key quantity is $P(r_t = r | x_{1:t})$ — the probability that the current segment is of length $r$. When a new observation is poorly explained by the current segment's predictive, mass shifts toward $r = 0$ (i.e., "we're at a new change-point").

```python
import numpy as np

def bocpd(data, hazard, observation_log_pdf, observation_update):
    """
    hazard: callable r -> P(change at run length r) — often constant 1/lambda
    observation_log_pdf(x, params) -> log p(x | params)
    observation_update(params, x) -> updated params
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    params = [None]                # list of parameters per run-length
    init_params = ...
    params[0] = init_params

    for t, x in enumerate(data):
        # predictive probabilities for each surviving run length
        pred = np.array([observation_log_pdf(x, params[r]) for r in range(t + 1)])
        pred = np.exp(pred - pred.max())
        # growth (run length increases by 1)
        growth = R[t, :t+1] * pred * (1 - hazard(np.arange(t + 1)))
        # change-point mass
        cp = (R[t, :t+1] * pred * hazard(np.arange(t + 1))).sum()
        # next-step posterior
        R[t+1, 1:t+2] = growth
        R[t+1, 0] = cp
        R[t+1] /= R[t+1].sum()
        # update parameters
        params = [init_params] + [observation_update(params[r], x) for r in range(t + 1)]
    return R
```

This is the algorithm in skeleton form. For a Gaussian model with unknown mean and known variance, the parameters are sufficient statistics (running sum + count) and the predictive is closed-form. The `bayesian_changepoint_detection` package implements the standard cases.

Read the posterior `R[t]` to see "the current segment is likely $r$ bars old." When mass piles up near $r = 0$, you're in a change-point.

The killer feature: BOCPD is **online**. You don't need the whole series. It updates as data arrives. For live regime detection, this is the right tool.

## `ruptures` for offline segmentation

For research / EDA on a known historical series, `ruptures` is the practical workhorse:

```python
import ruptures as rpt
import numpy as np

# Returns or log-vol
signal = np.log(rolling_vol.dropna().values)

# Pelt algorithm with an RBF cost — flexible
algo = rpt.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=10)            # returns indices of detected breaks

# Or with a known number of breaks
breakpoints = rpt.Binseg(model="l2").fit(signal).predict(n_bkps=5)
```

Three algorithms worth knowing:

- **Pelt** — optimal exact segmentation; cost grows linearly in series length. The default.
- **Binseg** — binary segmentation; greedy, fast, approximate.
- **Window** — sliding window; very fast, good for rough scans.

Five cost models worth knowing:

- `l1`, `l2` — least absolute / squared deviation. Detect mean shifts.
- `rbf` — kernel-based, detects shifts in distribution shape.
- `normal` — Gaussian likelihood; needs both mean and variance.
- `ar` — autoregressive coefficient changes.

For vol-regime detection on equities, `rpt.Pelt(model="rbf").fit(log_vol).predict(pen=…)` with a penalty tuned to give roughly the number of breaks you expect is a great starting point.

## A worked example: regime overlay

Use BOCPD on rolling realised vol to detect regime shifts; switch strategy allocation when one occurs.

```python
import numpy as np

# Simple Gaussian BOCPD on log-vol with hazard 1/200 (mean run length ~200 bars)
def detect_regime_shifts(log_vol, hazard_rate=1/200):
    # Implementation in `bayesian_changepoint_detection` package; pseudocode here.
    R = bocpd_gaussian(log_vol, hazard=hazard_rate)
    # For each t, the maximum-a-posteriori run length
    map_run = R.argmax(axis=1)
    # A change-point is when map_run[t] < map_run[t-1] - 5 (a drop in segment age)
    shifts = np.where(np.diff(map_run) < -5)[0]
    return shifts
```

Detected shifts feed an allocator: when a shift fires, raise the cash buffer for $N$ days while you re-evaluate the live strategies.

## A worked example: model stale detection

A common live-trading task: your ML model's predictions stop matching realised outcomes. CUSUM on the prediction residual is the cleanest answer:

```python
def model_stale_detector(residuals, sigma):
    return cusum(residuals, target=0.0, k=0.5*sigma, h=4*sigma)

flags = model_stale_detector(model_residuals, sigma=model_residuals.std())
if flags[-1] != 0:
    alert("Model residuals broke CUSUM threshold — refit or pause")
```

This catches the "you're using yesterday's model on a different world" problem within a few bars.

## When change-point detection misfires

!!! warning "Confusing change-points with outliers"
    A single huge return isn't a regime change — it's a fat-tail. Smooth first (e.g. fit on rolling vol, not raw returns) or use robust statistics.

!!! warning "Picking a too-low hazard"
    BOCPD with a too-frequent prior (`hazard = 1/10`) declares change-points constantly. Pick the hazard to match your real belief about regime durations.

!!! warning "Multiple-testing"
    Scan 500 symbols for change-points; ~5% will trigger by chance under any sensitivity. Apply a multiple-testing correction (Module 6) or only act when *many* symbols flag simultaneously.

!!! warning "Stationary process with a slow trend = no change-point"
    A series with a gentle linear drift will *always* fail CUSUM as it accumulates. Detrend first or use the right model (e.g. `ar` cost for `ruptures`).

## Bottom line

For three different jobs:

- **Live, online regime / model staleness detection** → CUSUM or BOCPD.
- **Offline segmentation of a historical series** → `ruptures` with `Pelt` + `rbf`.
- **Multi-symbol regime scanner** → BOCPD per symbol + a counting rule on simultaneous triggers.

## End of Module 7

You now have the time-series toolkit a Principal Engineer would actually use: stationarity tests, ARIMA/GARCH for vol forecasting, Kalman + particle filters for state estimation, wavelets for multi-scale feature work, and change-point detection for regime shifts.

The next module — **Classical Quant** — finally puts this machinery to work building strategies: factor models, momentum, mean reversion, pairs, risk parity, Black-Litterman, Kelly sizing. The fun starts now.

Continue to **[Module 8 — Classical Quant](../08-classical-quant/index.md)** (Phase 3).

!!! note "End of Phase 2"
    Phase 2 covered pandas/polars, data engineering, statistics, and time series — about 40 chapters of content total across Phases 1 and 2. Phase 3 starts the strategy modules. Bookmark the site and bring questions to the next review.
