# Almgren-Chriss optimal execution in detail

Module 13 chapter 5 introduced Almgren-Chriss (AC) as the RL baseline. This chapter is the full implementation with the realistic refinements that ship in production execution algos.

## The model recap

You must execute $X$ shares over $T$ time steps. Optimal schedule minimises expected total cost + λ × variance of cost.

Two impact terms:

- **Temporary impact**: $\eta n_k$ per share traded in step $k$.
- **Permanent impact**: $\gamma \sum_k n_k$ per share (cumulative).

Timing risk: variance from price drift over the schedule.

The closed-form schedule with risk aversion $\lambda > 0$:

$$
n_k \propto \sinh(\kappa(T - k)) - \sinh(\kappa(T - k - 1))
$$

where $\kappa = \sqrt{\lambda \sigma^2 / \eta}$.

## The clean implementation

```python
import numpy as np


def almgren_chriss_schedule(X: float, N: int, sigma: float, eta: float,
                              lambda_risk: float = 0.0, gamma_perm: float = 0.0) -> np.ndarray:
    """
    Optimal schedule: shares to trade in each of N discrete steps.

    X: total shares.
    N: number of execution steps.
    sigma: per-step price volatility (in price units, not %).
    eta: temporary impact per share.
    lambda_risk: mean-variance risk aversion (0 = risk-neutral → equal slices).
    gamma_perm: permanent impact per share (informational).
    """
    if lambda_risk <= 0:
        return np.full(N, X / N)
    kappa = np.sqrt(lambda_risk * sigma ** 2 / eta)
    t = np.arange(N + 1)
    # Holdings trajectory (shares remaining)
    holdings = X * np.sinh(kappa * (N - t)) / np.sinh(kappa * N)
    # Shares traded per step
    return -np.diff(holdings)
```

For typical equity execution (1000 shares of SPY over 10 minutes), the schedule is heavily front-loaded — about 30% in the first step, then tapering.

## Expected cost and risk

The expected total cost (vs arrival price) and its variance:

$$
\mathbb{E}[\text{cost}] = \eta \sum_k n_k^2 + \gamma X \sum_k n_k
$$

$$
\text{Var}[\text{cost}] = \sigma^2 \sum_k T_k^2 \cdot \left(\sum_{j \geq k} n_j\right)^2
$$

where $T_k$ is the step size (typically 1).

```python
def expected_cost_and_variance(schedule, sigma, eta, gamma_perm=0.0, step=1.0):
    X = schedule.sum()
    n = schedule
    expected_cost = eta * np.sum(n ** 2) + gamma_perm * X * np.sum(n)
    holdings = np.concatenate([[X], X - np.cumsum(n)])
    variance = sigma ** 2 * step * np.sum(holdings[:-1] ** 2)
    return float(expected_cost), float(variance)
```

Plot expected cost vs variance over a sweep of $\lambda$ values — the **efficient frontier** of execution schedules. Choose $\lambda$ based on your real cost-vs-risk preference.

## Adding realistic constraints

Production execution adds:

### 1. Minimum and maximum step sizes

Many venues have minimum order sizes. Maximum step sizes prevent any single step from being too aggressive.

```python
def constrained_schedule(X, N, sigma, eta, lambda_risk, min_step=1, max_step=None) -> np.ndarray:
    """Clip the AC schedule to per-step bounds, redistributing the slack."""
    sched = almgren_chriss_schedule(X, N, sigma, eta, lambda_risk)
    if max_step is not None:
        sched = np.minimum(sched, max_step)
    sched = np.maximum(sched, min_step)
    # Rescale to total X
    return sched * (X / sched.sum())
```

### 2. Volume-aware throttling

Don't trade more than some fraction of the displayed volume per minute:

```python
def volume_aware_schedule(X, sigma, eta, lambda_risk, volume_forecast: np.ndarray,
                           max_pct_volume: float = 0.10) -> np.ndarray:
    """volume_forecast: per-step expected total trading volume."""
    N = len(volume_forecast)
    sched = almgren_chriss_schedule(X, N, sigma, eta, lambda_risk)
    cap = volume_forecast * max_pct_volume
    sched = np.minimum(sched, cap)
    # Carry forward any "missed" shares
    deficit = X - sched.sum()
    if deficit > 0:
        # Add to back-loaded later steps if room
        for i in range(N - 1, -1, -1):
            available = cap[i] - sched[i]
            add = min(available, deficit)
            sched[i] += add
            deficit -= add
            if deficit <= 0: break
    return sched
```

### 3. Square-root impact

Linear impact is wrong for large orders. Replace $\eta n$ with $\eta \sqrt{n / V}$ where $V$ is daily volume. The closed-form AC no longer applies; solve numerically.

```python
from scipy.optimize import minimize


def numerical_optimal_schedule(X, N, sigma, adv, eta_sqrt, lambda_risk):
    """Numerical solve for square-root impact. n_k must be non-negative and sum to X."""
    def cost(n):
        cost_term = np.sum(eta_sqrt * np.sqrt(n / adv) * n)         # SQRT impact times qty
        holdings = X - np.cumsum(n)
        var_term = sigma ** 2 * np.sum(holdings ** 2)
        return cost_term + lambda_risk * var_term
    # Constraint: sum to X, all non-negative
    x0 = np.full(N, X / N)
    constraints = [{"type": "eq", "fun": lambda n: n.sum() - X}]
    bounds = [(0, None)] * N
    result = minimize(cost, x0, constraints=constraints, bounds=bounds, method="SLSQP")
    return result.x
```

## Comparing schedules

```python
import numpy as np

X = 100_000           # shares to execute
N = 20                # steps
sigma = 0.5
eta = 0.001
adv = 50_000_000

for lam in [0, 1e-7, 1e-6, 1e-5]:
    s = almgren_chriss_schedule(X, N, sigma, eta, lam)
    cost, var = expected_cost_and_variance(s, sigma, eta)
    print(f"λ={lam:.0e}: avg shares/step={X/N:.0f}, "
           f"first-step={s[0]:.0f}, last-step={s[-1]:.0f}, "
           f"E[cost]={cost:.0f}, Var={var:.0f}")
```

You'll see the trade-off: higher $\lambda$ → more front-loading → lower variance, higher expected cost.

## Adaptive AC

The fixed AC schedule doesn't react to actual price moves. **Adaptive AC** re-solves the optimal schedule at each step using the current market state. With realistic costs and randomness, adaptive AC outperforms fixed AC by 5-15% on realised cost variance.

```python
def adaptive_step(X_remaining, N_remaining, sigma, eta, lambda_risk):
    """Re-solve AC for the remaining work."""
    if N_remaining <= 0: return X_remaining
    sched = almgren_chriss_schedule(X_remaining, N_remaining, sigma, eta, lambda_risk)
    return sched[0]                            # trade only the next step
```

At each tick, call `adaptive_step` with `(X_remaining, N_remaining)` and execute the recommended amount. Updates $\sigma$ from realised vol; updates $\eta$ if you have a real-time impact model.

## When to use POV vs AC vs VWAP

- **AC / IS (Implementation Shortfall)**: minimise cost vs the arrival price. Best when your decision came at the arrival and the alpha is decaying.
- **VWAP**: track the day's volume-weighted average price. Best for benchmark-tracking and orderly execution.
- **POV (Percent of Volume)**: trade a fixed % of current volume. Best when you want to stay opportunistic and not lead the tape.

For most execution under cost-and-time constraints: adaptive AC (with realistic impact) is the right starting point.

## Pitfalls

!!! warning "Permanent vs temporary impact estimation"
    Hard to identify separately from data. A common approximation: permanent ≈ half of total impact.

!!! warning "AC for very small orders is overkill"
    For orders < 0.5% of ADV, just use VWAP or TWAP. AC's value scales with size.

!!! warning "AC under intraday volume seasonality"
    The standard AC assumes constant rates throughout the schedule. Real intraday volume has U-shape (high open, low midday, high close). Use volume-aware variants.

!!! warning "Implementation latency**
    Re-computing AC every tick is computationally cheap; sending child orders that fast is broker-rate-limited. Tune the schedule to match the broker's order-rate cap.

## Bottom line

Almgren-Chriss is the **structural backbone** of optimal execution. Production refinements:

- **Adaptive re-solve** at each step.
- **Square-root impact** for large orders.
- **Volume-aware constraints** (max % of ADV per step).
- **Layer RL on top** (Module 13 chapter 5) for residual learning.

For a quantitative shop executing 0.1-5% of daily volume, AC-based algos save 5-20 bps per trade vs naive TWAP.

Continue to **[Child-order placement with reinforcement learning](06-rl-execution.md)**.
