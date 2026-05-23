# Implied volatility — robust solvers

The implied volatility (IV) is the value of $\sigma$ that makes the Black-Scholes price equal the observed market price. Conceptually simple; numerically full of traps. This chapter walks the working solver in `engine.options.implied_vol` and the edge cases that break naive approaches.

## The setup

Given market price $C^*$ for a European call with $(S, K, T, r, q)$, find $\sigma$ such that:

$$
\text{BS}(S, K, T, r, q, \sigma) = C^*
$$

The function $\text{BS}(\sigma)$ is monotonically increasing in $\sigma$. So a 1D root-finder works — if you can frame the problem correctly.

## Newton-Raphson with vega

The classical approach: Newton-Raphson uses vega as the derivative:

$$
\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - C^*}{\nu(\sigma_n)}
$$

Quadratic convergence near the root. Fast when it works.

**When it fails**: deep ITM or OTM options have $\nu \to 0$. Dividing by tiny vega blows up. Newton-Raphson swings wildly and diverges.

## Brent's method — the robust default

Brent's method combines bisection, secant, and inverse quadratic interpolation. It's guaranteed to converge if you provide a bracket where the function changes sign. For implied vol, that's easy: try `(1e-6, 5.0)` — the function evaluates to "much too low" at the small end and "much too high" at the large end.

```python
from scipy.optimize import brentq

def implied_vol(target_price, S, K, T, r, q=0.0, kind="call", tol=1e-7):
    # Check no-arbitrage bounds
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if kind == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    upper = S * np.exp(-q * T) if kind == "call" else K * np.exp(-r * T)
    if not (intrinsic - tol <= target_price <= upper + tol):
        return float("nan")
    def diff(sigma):
        return price(S, K, T, r, q, sigma, kind=kind) - target_price
    return float(brentq(diff, 1e-6, 5.0, xtol=tol))
```

That's `engine.options.implied_vol`. Brent's method handles the deep-ITM/OTM case robustly because it doesn't depend on the derivative — only on sign changes.

## Using it

```python
from engine.options import implied_vol, price

# Round-trip: price an option at a known vol, then back out the IV
S, K, T, r, q = 100.0, 105.0, 0.25, 0.03, 0.0
target_sigma = 0.30
target_price = float(price(S, K, T, r, q, target_sigma, kind="call"))
iv = implied_vol(target_price, S, K, T, r, q, kind="call")
print(f"Round-trip IV: {iv:.6f}")          # 0.300000
```

For a real chain:

```python
import pandas as pd
chain = pd.DataFrame({
    "strike": [95, 100, 105, 110, 115],
    "mid": [8.50, 5.20, 2.85, 1.20, 0.35],
    "T": [30/365] * 5,
})
chain["iv"] = chain.apply(
    lambda r: implied_vol(r["mid"], S=100.0, K=r["strike"], T=r["T"],
                          r=0.045, q=0.0, kind="call"),
    axis=1,
)
print(chain)
```

For a vectorised version (faster on big chains), wrap in `np.vectorize` or roll a vectorised Brent.

## Vectorised Newton with safety

If you must go faster, vectorised Newton with a Brent fallback:

```python
import numpy as np
from scipy.stats import norm

def implied_vol_vec(targets, S, K, T, r, q=0.0, kind="call", max_iter=20, tol=1e-7):
    sigma = np.full_like(np.asarray(K, dtype=float), 0.2)
    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if kind == "call":
            p = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            p = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        diff = p - targets
        sigma = sigma - diff / np.where(vega > 1e-8, vega, 1e-8)
        if np.max(np.abs(diff)) < tol:
            break
    # Fall back to Brent for any that didn't converge
    converged = np.abs(diff) < tol
    # ... handle non-converged elements with scalar Brent ...
    return sigma
```

For a 1000-row chain, vectorised Newton is ~100× faster than per-row Brent. Use it for backtests and large-scale surface fitting.

## The deep-ITM / OTM trap

For a deep ITM call (say, $S = 200, K = 100$), the call is essentially "buy the underlying minus a bond." Its vega is tiny. Its market price has only one or two decimal places of precision. The implied vol back-solve has very high standard error.

Don't quote IVs for deeply ITM options as if they were precise. The market does this too — quote a deep ITM option's IV at 22% and the deep OTM put at 22% and you're saying the same thing (by put-call parity).

**Best practice**: for IVs, work from the *out-of-the-money* options on each side and use put-call parity for the ITM side.

## A worked example: smile recovery from a chain

```python
import pandas as pd
from engine.options import implied_vol

# Pretend chain
chain = pd.DataFrame({
    "strike": [450, 460, 470, 475, 480, 490, 500],
    "right": ["P", "P", "P", "P", "C", "C", "C"],
    "mid": [3.50, 4.80, 6.50, 7.90, 9.10, 5.20, 2.40],
})
S, T, r, q = 475.0, 30/365, 0.045, 0.013

chain["iv"] = chain.apply(
    lambda row: implied_vol(row["mid"], S, row["strike"], T, r, q,
                            kind="call" if row["right"] == "C" else "put"),
    axis=1,
)
print(chain)
# The smile: IV typically higher at the wings than ATM
```

The IVs you get back should form a smile (or smirk for equity indices). If they don't, you have stale quotes, mid-price errors, or a model misspecification.

## Pitfalls

!!! warning "Asking for IV outside no-arbitrage bounds"
    A call price below intrinsic (= max(0, S - K e^{-rT})) has no valid IV. Our `implied_vol` returns `NaN` in this case rather than failing or returning a meaningless number.

!!! warning "American options"
    Black-Scholes IV is meaningless for American options that are likely to be early-exercised. For US equity options that are ITM and have months left, use a binomial-tree-based IV solver. For OTM equity options, Black-Scholes IV is a fine approximation.

!!! warning "Bid-ask mid IV vs market IV"
    Vendor-reported IVs are usually from last trade; you usually want IVs from the bid-ask mid. Recompute.

!!! warning "Near-expiry instability"
    For options expiring in hours, IV becomes noisy because price granularity dominates. Filter to T > 1 day for meaningful smiles.

## Bottom line

For implied volatility:

- **Use `engine.options.implied_vol`** for one-off calls — robust, Brent-based, returns `NaN` for out-of-bounds prices.
- **Use vectorised Newton with Brent fallback** for big chains in backtests.
- **Always recompute from bid-ask mid**, never the last trade.
- **Always check no-arbitrage bounds** before fitting a surface.

Continue to **[American options — binomial trees and LSMC](04-american.md)**.
