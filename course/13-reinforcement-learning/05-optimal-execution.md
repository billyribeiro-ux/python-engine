# Almgren-Chriss optimal execution

The Almgren-Chriss (AC) model (1999/2000) is the canonical analytic answer to "you must execute X shares over T minutes — how do you schedule the trades to minimise the sum of expected cost and risk?" Like AS for market-making, AC is the analytic baseline you build on top of with RL.

## The AC model

Total shares to execute: $X$. Execution interval: $[0, T]$. Discretise into $N$ steps; let $n_k$ be the shares traded in step $k$. Constraints: $\sum n_k = X$, $n_k \geq 0$.

Two cost terms:

1. **Temporary impact** (linear): $\eta n_k$ per share — proportional to the size of each child order.
2. **Permanent impact** (linear): $\gamma \sum_k n_k$ — accumulates with cumulative trading.

Plus **timing risk** — variance of the cost from price drift over the schedule, $\sigma^2 \sum_k T_k^2 \cdot n_k^2$ (Frobenius).

Mean-variance objective:

$$
\min_{n_k} \mathbb{E}[\text{cost}] + \lambda_\text{risk} \text{Var}[\text{cost}]
$$

The closed-form optimal schedule when $\lambda_\text{risk} > 0$ is **exponentially-decreasing**:

$$
n_k \propto \sinh(\kappa (N - k))
$$

where $\kappa$ depends on $\lambda_\text{risk}, \sigma, \eta$. For $\lambda_\text{risk} = 0$ (risk-neutral), $\kappa \to 0$ and the schedule becomes **linear** (equal slices).

In code:

```python
import numpy as np

def almgren_chriss_schedule(X, N, sigma, eta, lambda_risk=0.0):
    """Optimal AC schedule: shares to trade per discrete step (k = 0, ..., N-1)."""
    if lambda_risk <= 0:
        return np.full(N, X / N)            # equal slices
    kappa = np.sqrt(lambda_risk * sigma ** 2 / eta)
    t = np.arange(N + 1)
    holding = X * np.sinh(kappa * (N - t)) / np.sinh(kappa * N)
    return -np.diff(holding)                # how much to trade in each interval

schedule = almgren_chriss_schedule(X=100_000, N=20, sigma=0.5, eta=0.001, lambda_risk=1e-6)
print(schedule)                              # heavier at the start, tapers
```

The intuition: trade more aggressively early when you don't know what the price will do; less aggressively as you approach the deadline (you've already locked in most of the cost).

## Where AC breaks

1. **Constant impact** — real impact is concave (square-root, not linear).
2. **Constant risk aversion** — your real risk aversion depends on inventory remaining and time left.
3. **No reaction to price moves** — AC schedules the trades ex-ante and doesn't change them.
4. **No queue / venue dynamics** — assumes you can always trade at the prevailing price.
5. **No adverse selection** — assumes the market doesn't react to your trades.

Most of these are why real execution algos use AC as a starting point, then layer adaptive logic on top. RL is the natural framework for the adaptive part.

## A worked example: comparing AC variants

```python
import numpy as np

def cost_under_path(schedule: np.ndarray, prices: np.ndarray, eta: float = 0.001) -> float:
    """Compute the realised cost = sum(n_k * (price + temp_impact))."""
    assert len(schedule) == len(prices)
    fills = prices + eta * schedule
    return float((schedule * fills).sum())


def simulate_paths(N: int = 20, sigma: float = 0.5, n_paths: int = 1000):
    paths = np.cumsum(np.random.default_rng(0).normal(0, sigma, (n_paths, N)), axis=1) + 100.0
    return paths


# Compare equal-slice (TWAP) vs AC vs front-loaded vs back-loaded
paths = simulate_paths()
X, N = 100_000, 20
scheds = {
    "TWAP": np.full(N, X / N),
    "AC (risky)": almgren_chriss_schedule(X, N, sigma=0.5, eta=0.001, lambda_risk=1e-5),
    "Front-loaded": X * np.exp(-np.arange(N) / 5) / np.exp(-np.arange(N) / 5).sum(),
    "Back-loaded": X * np.exp(np.arange(N) / 5) / np.exp(np.arange(N) / 5).sum(),
}

for name, sched in scheds.items():
    costs = [cost_under_path(sched, paths[i] - paths[i, 0]) for i in range(len(paths))]
    print(f"{name}: mean={np.mean(costs):.0f}, std={np.std(costs):.0f}")
```

TWAP has higher variance (one bad path costs a lot). AC has lower variance (heavier early reduces timing risk). Front-loaded is even lower variance but higher mean cost. The trade-off matches AC's mean-variance theory.

## RL on top of AC

The pattern: start with the AC schedule, allow the agent to **deviate** based on real-time state (current price relative to arrival, remaining inventory, observed flow):

```python
# Pseudocode
ac_schedule = almgren_chriss_schedule(X, N, sigma, eta, lambda_risk)
for k in range(N):
    state = (k, inventory_remaining, current_price - arrival_price, recent_flow)
    deviation = rl_policy.predict(state)
    qty_this_step = max(0, ac_schedule[k] + deviation)
    fill_at_market(qty_this_step)
    inventory_remaining -= qty_this_step
```

The RL learns to **accelerate** when the price has moved against you (lock in remaining cost), **decelerate** when it has moved in your favour (let the favourable drift continue), and **resize** based on observed flow imbalance.

For training, the env simulates intraday paths with a realistic impact model and lets the agent practice millions of executions.

## Production patterns

Real execution algos go beyond plain AC + RL adaptation:

- **VWAP** (volume-weighted average price) — trade in proportion to historical volume profile. Often the right baseline for predictable intraday volume patterns.
- **POV** (percent-of-volume) — trade at a fixed percentage of current volume. Adaptive but not optimal.
- **Implementation Shortfall** — minimise the gap to arrival price (which is what AC and RL minimise).
- **Liquidity-seeking** — opportunistic; only trade when liquidity is abundant. Variable horizon.

For a complete production execution stack, see Cartea-Jaimungal-Penalva *Algorithmic and High-Frequency Trading* (2015) — still the best academic reference.

## Pitfalls

!!! warning "Square-root impact for big orders"
    Linear impact assumed by AC is fine for small orders. For orders > 5% of ADV, impact scales as $\sqrt{q/V}$. Use the square-root cost model (Module 9 ch6) in your simulator.

!!! warning "Permanent impact estimation"
    Almgren-Chriss treats permanent impact as a fixed parameter. In practice it depends on market conditions and is hard to identify. Conservative assumption: permanent impact = half the temporary impact.

!!! warning "Reactive execution on a slow signal"
    If your RL agent reacts to "current price minus arrival" and the price is autocorrelated, you can end up chasing the trend (worst case for execution). Smooth or filter the signal carefully.

!!! warning "Out-of-sample drift on execution algos"
    Markets change. An execution algo trained on 2020 high-vol data may be over-eager in 2024 calm regimes. Periodically refit on recent data.

## Bottom line

For optimal execution:

- **AC analytical schedule** as the baseline.
- **RL deviation on top** for adaptive behaviour based on real-time state.
- **Use a square-root impact model** in simulation for orders > 1% ADV.
- **Always paper-trade** before deploying any execution algo live.

Continue to **[Offline RL and the sim-to-real gap](06-offline-rl.md)**.
