# Conformal prediction with MAPIE

A point prediction is a guess. A prediction with an honest confidence interval is a decision aid. Conformal prediction is a remarkably simple framework that, given any black-box predictor, produces intervals with **a finite-sample guarantee on coverage** — you ask for 90% coverage and you get at least 90% coverage on truly held-out data.

For trading, conformal intervals on the next-bar return give you exactly what you need to size: an interval-bounded forecast you can multiply against your risk budget.

## The setup

Given:

- A predictor $\hat f$ (any black box: gradient boosting, neural net, even a constant).
- A calibration set the predictor has never seen, of size $n$.

For each calibration point compute the **nonconformity score**:

$$
s_i = |y_i - \hat f(x_i)|
$$

(other choices of $s$ exist; absolute residual is the standard for regression). Sort the $s_i$. For a target coverage $1 - \alpha$, pick the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile, call it $\hat q$. Then for a new point $x$:

$$
[\hat f(x) - \hat q, \; \hat f(x) + \hat q]
$$

is a **valid** $(1-\alpha)$ interval — meaning, *across many such intervals on truly fresh data, at least $1-\alpha$ contain the truth*. No distributional assumptions needed.

This is **split conformal**. It's the simplest variant and the right starting point.

## MAPIE — the standard Python library

MAPIE (Model Agnostic Prediction Interval Estimator) implements split conformal, cross-conformal, jackknife+, and adaptive variants:

```python
from mapie.regression import MapieRegressor
from sklearn.ensemble import GradientBoostingRegressor

base = GradientBoostingRegressor(n_estimators=100, max_depth=3)
mapie = MapieRegressor(base, method="plus", cv=5)
mapie.fit(X_train, y_train)

# Predict point + intervals at multiple confidence levels
y_pred, y_intervals = mapie.predict(X_test, alpha=[0.1, 0.05])
# y_intervals[i, 0, j] = lower bound of (1 - alpha_j) interval for sample i
# y_intervals[i, 1, j] = upper bound
```

For binary classification, MAPIE's `MapieClassifier` gives you **prediction sets** instead of intervals — the smallest set of labels that contains the true label with at least $(1-\alpha)$ probability.

## Why conformal beats normal-based intervals

Standard "$\hat y \pm 1.96 \sigma$" intervals assume Gaussian errors and a correctly-specified model. Both are heroic for finance. Conformal makes no such assumption — it builds intervals from the actual residual distribution.

For fat-tailed financial returns, conformal intervals are typically:

- **Wider** than Gaussian intervals (which is honest).
- **Asymmetric** when you use adaptive variants that fit a quantile predictor (more on this below).
- **Valid out-of-sample** even when the model is badly miscalibrated.

## Adaptive conformal — when uncertainty varies

The basic split conformal gives constant-width intervals. But the *true* uncertainty varies — quiet periods should give tight intervals, volatile periods wide ones.

**Conformalised quantile regression** (CQR) fits a quantile regressor and adjusts the quantile boundaries by the conformal residual:

```python
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Fit lower and upper quantile predictors at alpha/2 and 1 - alpha/2
lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=100)
hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=100)
pt = GradientBoostingRegressor(n_estimators=100)

mapie = MapieQuantileRegressor(estimator=[lo, hi, pt])
mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
y_pred, y_intervals = mapie.predict(X_test)
```

Now the interval width depends on $x$ — adaptive to local volatility.

## A worked example: conformal sizing

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor

# Pretend X is features, y is forward 5-day return
def conformal_sized_positions(X, y, X_test, alpha=0.20, kelly_factor=0.25):
    base = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    mapie = MapieRegressor(base, method="plus", cv=5)
    mapie.fit(X, y)
    y_pred, y_int = mapie.predict(X_test, alpha=[alpha])
    lower = y_int[:, 0, 0]
    upper = y_int[:, 1, 0]
    # Trade only when the interval is strictly on one side of zero
    pos = np.zeros_like(y_pred)
    pos[lower > 0] = +1
    pos[upper < 0] = -1
    # Size proportional to the midpoint over the half-width (proxy Sharpe)
    half_width = (upper - lower) / 2
    edge = np.abs(y_pred) / np.maximum(half_width, 1e-6)
    return pos * kelly_factor * edge.clip(0, 4)
```

The pattern: **only trade when the conformal interval is strictly on one side of zero**, and **size by the ratio of point prediction to interval width**. This automatically:

- Stays flat when uncertainty exceeds the signal.
- Up-sizes when the model is confident and the prediction is meaningful.
- Reduces size when conformal widths increase (volatile regimes).

It's the cleanest combination of model output and uncertainty for sizing.

## Online / streaming conformal

For live trading, you want the conformal threshold to update as data arrives. **Adaptive Conformal Inference (ACI)** does this:

```python
# Conceptual sketch
class OnlineConformal:
    def __init__(self, alpha=0.1, gamma=0.005):
        self.alpha = alpha
        self.gamma = gamma            # adaptation rate
        self.q = 0.0                  # current threshold
        self.history = []
    def predict_interval(self, point):
        return (point - self.q, point + self.q)
    def update(self, point, truth):
        residual = abs(truth - point)
        # If we missed (truth outside interval), raise q; if we covered, lower q
        miss = int(residual > self.q)
        self.q += self.gamma * (miss - self.alpha)
        self.history.append(residual)
```

Each update nudges $q$ toward maintaining the target coverage. After enough data, $q$ stabilises at the right level for the current regime. When the regime shifts, $q$ adapts.

## Pitfalls

!!! warning "Conformal validity requires exchangeable data"
    Strictly speaking, the coverage guarantee assumes the calibration and test samples are exchangeable. Financial data isn't (it's autocorrelated and non-stationary). In practice, conformal still gives roughly the right coverage if you respect time ordering and refit periodically.

!!! warning "Constant-width intervals on heteroskedastic data"
    Basic conformal gives the same width everywhere. For finance, switch to CQR or normalized conformal (where you also fit a residual-magnitude predictor and divide).

!!! warning "Calibration set too small"
    With $n < 100$ calibration points, the quantile is noisy and the intervals are wider than they need to be. Use jackknife+ instead of split for small datasets.

!!! warning "Coverage is marginal, not conditional"
    "90% coverage" means *averaged over all conditions*. The model can have 100% coverage in calm markets and 50% in crashes, and still satisfy the marginal guarantee. For trading, you usually care more about the worst-case condition — test coverage *per regime*.

## Bottom line

For honest forecasts:

- **Wrap your predictor in MAPIE** for instant prediction intervals.
- **Use `method="plus"`** (cross-conformal) for the best stability.
- **Use CQR (`MapieQuantileRegressor`)** when uncertainty varies (most financial data).
- **Trade only when intervals strictly cover one side of zero**; size by point/width ratio.
- **Refit periodically** to handle regime shifts; for live systems, consider online conformal (ACI).

Conformal is the single most under-used tool in retail ML-for-trading. Adopt it and your sizing immediately becomes more honest.

Continue to **[An end-to-end ML signal pipeline](06-pipeline.md)**.
