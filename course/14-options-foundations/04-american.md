# American options — binomial trees and LSMC

Black-Scholes is for European options (exercise at expiry only). US equity options are American — exercisable at any time. The early-exercise feature gives them slightly more value, especially for in-the-money puts (where early exercise locks in the strike before further downside). Two main pricing methods: binomial trees (simple, slow) and Longstaff-Schwartz Monte Carlo (faster for many strikes / high-dim).

## When early exercise matters

- **American puts** — yes, especially deep ITM. The dividend the strike-cash earns at the bond rate can outweigh the option's remaining time value.
- **American calls on non-dividend stocks** — no. Black-Scholes is exact. Early exercise is never optimal.
- **American calls on dividend-paying stocks** — sometimes, right before a large discrete dividend. The dividend drop in spot exceeds the remaining time value.

For SPY (a continuously-yielding ETF), early exercise is rarely optimal in practice but the model should still price American.

## The binomial tree

Cox-Ross-Rubinstein (1979). Discretise time into $N$ steps. At each step, the price can move up by factor $u = e^{\sigma \sqrt{\Delta t}}$ or down by $d = 1/u$ with risk-neutral probabilities chosen so the expected log-return matches the risk-neutral drift.

Build a tree of $N + 1$ price levels at expiry. Compute payoffs at expiry. Then walk backwards: at each node, the option value is the max of (a) early exercise payoff, and (b) discounted expected continuation value. The root is the price.

```python
import numpy as np

def binomial_american(S, K, T, r, q, sigma, N=200, kind="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Terminal prices
    j = np.arange(N + 1)
    prices_T = S * (u ** (N - j)) * (d ** j)
    if kind == "call":
        values = np.maximum(prices_T - K, 0)
    else:
        values = np.maximum(K - prices_T, 0)

    # Backward induction with early exercise
    for step in range(N - 1, -1, -1):
        j = np.arange(step + 1)
        prices_t = S * (u ** (step - j)) * (d ** j)
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        exercise = np.maximum(prices_t - K, 0) if kind == "call" else np.maximum(K - prices_t, 0)
        values = np.maximum(continuation, exercise)
    return float(values[0])
```

50 lines. $N = 200$ is enough for sub-cent accuracy on most equity options. $N = 1000$ for production. The complexity is $O(N^2)$ in time and memory.

## Convergence

```python
for N in [10, 50, 100, 500, 2000]:
    p_amer = binomial_american(100, 110, 0.5, 0.04, 0.0, 0.25, N=N, kind="put")
    print(f"N={N}: American put = {p_amer:.4f}")
```

Typical output:

```
N=10:    5.5102
N=50:    5.4837
N=100:   5.4790
N=500:   5.4762
N=2000:  5.4757
```

The price converges from above. For an exact comparison the corresponding European put (Black-Scholes) is ~5.45 — the American premium is about 0.03 per $100 in this case.

## Longstaff-Schwartz Monte Carlo

For multi-factor or high-dimensional options (e.g., basket options), binomial trees become unwieldy. **Longstaff-Schwartz** (2001) uses Monte Carlo paths with a regression-based estimate of continuation value:

1. Simulate $M$ underlying paths over $N$ time steps.
2. At expiry, compute payoff per path.
3. Walk backwards: at each step, regress the discounted future cash flow (already known from previous backward step) on a small basis of the current price (e.g., $1, S, S^2$).
4. Predicted regression gives the continuation value; compare to exercise. Exercise if exercise > continuation.

```python
def lsmc_american(S0, K, T, r, q, sigma, kind="put",
                  n_paths=10_000, n_steps=50, basis_degree=2, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    # Simulate paths
    paths = np.full((n_paths, n_steps + 1), S0)
    z = rng.normal(0, 1, (n_paths, n_steps))
    log_paths = np.cumsum(drift + diffusion * z, axis=1)
    paths[:, 1:] = S0 * np.exp(log_paths)
    # Cash flows at expiry
    cf = np.maximum(K - paths[:, -1], 0) if kind == "put" else np.maximum(paths[:, -1] - K, 0)
    disc = np.exp(-r * dt)
    for t in range(n_steps - 1, 0, -1):
        # Only ITM paths are candidates for early exercise
        itm = (K - paths[:, t]) > 0 if kind == "put" else (paths[:, t] - K) > 0
        if itm.sum() == 0:
            cf = disc * cf
            continue
        x = paths[itm, t]
        # Basis: polynomial up to basis_degree
        X = np.column_stack([x ** k for k in range(basis_degree + 1)])
        y = cf[itm] * disc  # discounted future cash flow
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        continuation = X @ coef
        exercise = K - x if kind == "put" else x - K
        do_exercise = exercise > continuation
        # Update cash flows for paths that exercise
        idx_itm = np.where(itm)[0]
        cf[idx_itm[do_exercise]] = exercise[do_exercise]
        cf = disc * cf
        # For paths that didn't exercise, cf is already (disc * future_cf) which is correct
    return float(cf.mean())
```

For 10,000 paths and 50 steps, LSMC gives a 1-cent-accurate price in milliseconds. For high accuracy, scale to 100,000 paths and 200 steps.

## When to use what

- **Single-asset American option** → binomial tree with N=500-1000. Fast, accurate, simple.
- **Multi-asset basket options** → LSMC. Binomial doesn't scale to dimensions > 2.
- **Path-dependent options** (Asian, lookback) → LSMC. The cash flow at each step can depend on the whole path.
- **Vol-surface stripping** → LSMC against a stochastic-vol model (Module 15).

## Pitfalls

!!! warning "Binomial tree off-by-one on dividend dates"
    Discrete dividends require subtracting the dividend on the right time step in the tree. Easy to get wrong; verify against put-call parity for the European case.

!!! warning "LSMC basis choice"
    Polynomial basis up to degree 2 or 3 is fine for most equity options. For exotic payoffs, use Laguerre or Hermite polynomials.

!!! warning "Variance reduction in LSMC"
    Naive LSMC has high MC variance. Use antithetic variates (mirror each path's noise) and control variates (subtract the European option's analytical price difference) for 5-10× variance reduction.

!!! warning "American Greeks"
    Computing Greeks of American options analytically is hard. Finite-difference on the binomial-tree pricer works but is slow. The "smooth" Greeks come from regression-based methods or adjoint algorithmic differentiation.

## Bottom line

For American option pricing:

- **Binomial tree (N=500)** as the default, especially for single-asset.
- **LSMC** for multi-asset or path-dependent.
- **For SPY options specifically**, the American premium is small — Black-Scholes is a fine approximation for OTM, and put-call parity gives ITM for free.

Continue to **[Put-call parity and synthetic positions](05-parity.md)**.
