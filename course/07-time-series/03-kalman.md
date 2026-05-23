# Kalman filters from scratch

The Kalman filter is the optimal estimator for a linear dynamical system observed with Gaussian noise. It is also one of the most useful tools in quantitative finance, full stop. Once you've seen what it does, you'll find applications constantly: time-varying betas, pairs trading hedge ratios, smoothing noisy signals, online recursive regression, state-of-the-world inference under uncertainty.

This chapter implements it from scratch so the mechanics are unmistakable, then shows you the applications.

## The state-space framework

Two equations, no more no less:

**State equation** — how the hidden state evolves:
$$
x_t = F_t x_{t-1} + B_t u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)
$$

**Observation equation** — how the state generates what you see:
$$
z_t = H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)
$$

The filter recursively estimates the posterior $p(x_t | z_{1:t})$ — your best guess of the hidden state given everything you've seen so far. Because everything is linear and Gaussian, the posterior is itself Gaussian: it has a mean $\hat x_t$ and a covariance $P_t$.

## The recursion in five lines

Predict:
$$
\hat x_t^- = F_t \hat x_{t-1} + B_t u_t
$$
$$
P_t^- = F_t P_{t-1} F_t^\top + Q_t
$$

Update:
$$
K_t = P_t^- H_t^\top (H_t P_t^- H_t^\top + R_t)^{-1}
$$
$$
\hat x_t = \hat x_t^- + K_t (z_t - H_t \hat x_t^-)
$$
$$
P_t = (I - K_t H_t) P_t^-
$$

That's it. $K_t$ is the **Kalman gain** — how much to trust the new observation. When the observation variance $R$ is large, $K$ is small (don't trust it). When the prediction variance is large, $K$ is large (trust it more).

## Implementation, no library

```python
import numpy as np

class Kalman:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R           # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        y = z - self.H @ self.x                           # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return float(y), float(S)                         # return for logging
```

Sixteen lines. That's the entire optimal-linear-Gaussian estimator. Production-grade code uses Joseph form for the covariance update (numerically more stable) and avoids the matrix inverse in `update` — but the algebra is the same.

## A worked example: smoothing a noisy price into a hidden "fair value"

Model:
- State $x_t$ — the "true" fair value (1D), evolves as a random walk: $x_t = x_{t-1} + w_t$.
- Observation $z_t$ — the noisy traded price: $z_t = x_t + v_t$.

```python
import numpy as np
np.random.seed(0)

# Simulated truth and observations
T = 500
true = np.cumsum(np.random.randn(T) * 0.5)
obs  = true + np.random.randn(T) * 2.0

kf = Kalman(
    F=np.array([[1.0]]),
    H=np.array([[1.0]]),
    Q=np.array([[0.25]]),      # tells the filter the truth wanders this much
    R=np.array([[4.0]]),       # tells the filter the obs noise is this big
    x0=[[obs[0]]], P0=[[1.0]],
)

filtered = np.empty(T)
for t in range(T):
    kf.predict()
    kf.update(obs[t])
    filtered[t] = float(kf.x)
```

The `filtered` series is much smoother than `obs` and tracks `true` closely. The smoothness comes free from the math — no tuning, no rolling window.

The two knobs are $Q$ (process noise) and $R$ (observation noise). Their ratio controls how aggressively the filter follows new observations vs. trusts its prediction. In practice you tune $Q/R$ empirically (or estimate with EM via `pykalman`).

## A worked example: time-varying hedge ratio for pairs trading

State: the hedge ratio $\beta_t$ between two assets, modelled as a random walk:
$$
\beta_t = \beta_{t-1} + w_t
$$

Observation: $A_t = \alpha + \beta_t B_t + v_t$ — the linear relationship between the two prices, with the $A$ price as the observation.

This is a linear regression where the slope can **change over time**. The Kalman filter gives you the time-varying slope online.

```python
class KalmanPairs:
    def __init__(self, delta=1e-4, R=1.0):
        # State: [intercept, slope]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        # Process noise: Q = delta/(1-delta) * I, the "random walk" prior on betas
        self.Q = np.eye(2) * (delta / (1 - delta))
        self.R = np.array([[R]])

    def update(self, a, b):
        H = np.array([[1.0, b]])
        z = np.array([[a]])
        # predict
        self.P = self.P + self.Q
        # update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        return float(y), float(S)
```

```python
import pandas as pd
from engine.data import YFinanceFeed
feed = YFinanceFeed()
ko  = feed.bars("KO",  "2018-01-01", "2024-12-31")["close"]
pep = feed.bars("PEP", "2018-01-01", "2024-12-31")["close"]
df = pd.concat({"KO": ko, "PEP": pep}, axis=1).dropna()

kp = KalmanPairs(delta=1e-4, R=1.0)
out = []
for a, b in zip(df["KO"], df["PEP"]):
    spread, var = kp.update(a, b)
    out.append((float(kp.x[0]), float(kp.x[1]), spread, var ** 0.5))
df_out = pd.DataFrame(out, index=df.index, columns=["alpha", "beta", "spread", "z_sd"])
df_out["z"] = df_out["spread"] / df_out["z_sd"]      # standardised innovation
```

`df_out["beta"]` is the slowly-changing hedge ratio. `df_out["z"]` is the standardised spread — your trading signal. Trade when $|z|$ exceeds a threshold; exit when it crosses zero. Module 16 wraps this into a full strategy with risk management.

The `delta` parameter controls how fast the slope is allowed to change. Smaller = more stable but slower to adapt. Typical range: $10^{-5}$ to $10^{-3}$ for daily data.

## A worked example: estimating the latent equity premium

You believe the expected return of an asset is a slowly-varying state. Each daily return is a noisy observation of "today's expected return + noise". A 1D Kalman filter on the returns gives you an online estimate of the time-varying expected return:

```python
class KalmanMu:
    def __init__(self, q, r, x0=0.0):
        self.x, self.P, self.q, self.r = x0, 1.0, q, r
    def update(self, z):
        self.P += self.q              # predict
        K = self.P / (self.P + self.r)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
```

```python
km = KalmanMu(q=1e-7, r=1e-4)          # daily returns: tiny noise on mu, small observation noise
mu_path = [km.update(r) for r in daily_returns]
```

`mu_path` is the filter's estimate of the expected daily return over time. Combined with a vol estimate and a Sharpe-targeted position sizer, you have a complete momentum strategy in 30 lines.

## The smoother (RTS)

If you have an offline dataset and want the **best** estimate of every past state given the *whole* history (not just past), use the **Rauch-Tung-Striebel** smoother. It's a backward pass after the forward filter. `pykalman`'s `KalmanFilter.smooth(...)` does it; the math is in any Kalman textbook.

For research (offline analysis), always smooth. For live trading, you can't — you only have the past.

## When the Kalman filter doesn't apply

- The system is **non-linear** (e.g. options pricing as a function of vol). Use the Extended (EKF) or Unscented (UKF) Kalman Filter — both implemented in `filterpy`.
- The noise is **non-Gaussian**. Use a particle filter (next chapter).
- You have **regime switches**. Use a mixture of Kalman filters (interacting multiple model, IMM).

## Pitfalls

!!! warning "Tuning Q and R via 'eyeballing'"
    For a few critical apps, fit them via the EM algorithm (`pykalman.KalmanFilter` with `em_vars=["transition_covariance", "observation_covariance"]`). Hand-tuned $Q$ and $R$ are a recipe for "the filter looks right but doesn't trade well."

!!! warning "Forgetting to update P, only x"
    A common bug in hand-rolled filters: tracking the mean but not the covariance. The covariance is the filter's notion of uncertainty; without it, the Kalman gain is wrong.

!!! warning "Numerical instability on long runs"
    The covariance update is numerically delicate. Use the Joseph form $(I-KH)P(I-KH)^\top + KRK^\top$ in production — it's symmetric-and-positive-definite by construction.

## Bottom line

The Kalman filter is the right tool when:

- You have a **linear-Gaussian** system (or you can approximate as one).
- You want **online** estimates that update as data arrives.
- You want a principled **uncertainty** estimate, not just a point.

For pairs trading, beta estimation, smoothing noisy signals — reach for it by default. Build the from-scratch version once; from then on use `filterpy` or `pykalman`. The math is the same.

Continue to **[Particle filters](04-particle.md)**.
