# The Kelly trap

Kelly's formula (1956) gives the bet fraction that maximises long-run growth: $f^* = \mu / \sigma^2$ for a continuous-return strategy. It is **mathematically optimal under the assumptions** — known $\mu$ and $\sigma$, infinite horizon, no drawdown constraints, no agency risk.

It is **systematically too aggressive** for real trading. This chapter is why, and what to use instead.

## Three reasons full Kelly is too much

### 1. You don't know μ

Kelly assumes you know the true expected return. You don't — you have a sample estimate $\hat\mu$ that's noisy. **Sizing on $\hat\mu$ instead of $\mu$ means you sometimes over-bet**.

The error is most painful on **high-confidence-looking strategies**: a strategy with a 0.8 in-sample Sharpe but a true 0.4 Sharpe gets sized at *4×* the right Kelly fraction (because sample Sharpe enters squared).

### 2. Drawdown distribution at full Kelly is brutal

At full Kelly, the **expected maximum drawdown** over a long horizon approaches 100%. Empirically, Kelly portfolios spend significant time at 30-50% drawdowns. **Most investors can't tolerate this** — they redeem.

### 3. The growth-rate curve is asymmetric

Kelly's growth rate $g(f)$ is concave, peaking at $f^*$. The downside (over-betting) is much steeper than the upside (under-betting):

- 50% Kelly → ~75% of optimal growth.
- 100% Kelly → 100% of optimal growth (in theory).
- 150% Kelly → ~75% of optimal growth.
- 200% Kelly → 0% growth (you've crossed the "doubling rate is zero" line).

Estimation error pushes you upward; the asymmetry punishes that direction. The safe side is to bet less.

## Fractional Kelly

The standard production response: **fractional Kelly**, typically 25-50% of full Kelly.

```python
import numpy as np


def fractional_kelly_sizing(expected_return: float, variance: float, fraction: float = 0.5) -> float:
    """Returns the position fraction, scaled by the Kelly fraction."""
    if variance <= 0:
        return 0.0
    return fraction * expected_return / variance
```

At 50% Kelly:
- ~75% of full-Kelly growth.
- Roughly half the volatility.
- Roughly half the expected max drawdown.
- Far less sensitive to estimation error.

## A worked example

```python
# Strategy: estimated mean = 0.05, std = 0.10 (Sharpe 0.5)
mu, sigma = 0.05, 0.10
variance = sigma ** 2

for f_kelly in [0.25, 0.5, 1.0, 1.5]:
    position = f_kelly * mu / variance
    expected_g = f_kelly * mu * (mu / variance) - 0.5 * (f_kelly * mu / variance) ** 2 * variance
    print(f"f={f_kelly:.2f}: position={position:.2f}, expected growth={expected_g:.4f}")
```

Output shows the concavity: f=0.5 already captures most of the growth; f=1.5 is past peak and on the way down.

## What I actually use

For most strategies:

1. **Half-Kelly** as the sizing prior.
2. Cap the position at **3× the strategy's vol target** (so a 10% vol strategy can't be sized to 30% notional).
3. Apply a **drawdown overlay** that scales down as drawdown approaches max tolerance.

```python
def production_size(expected_return, variance, vol_target_pct=0.10,
                     max_size_multiplier=3.0, current_drawdown=0.0, max_dd=0.20):
    kelly_half = 0.5 * expected_return / variance
    sigma = np.sqrt(variance)
    max_size = vol_target_pct / sigma * max_size_multiplier
    base = min(abs(kelly_half), max_size) * np.sign(kelly_half)
    dd_factor = (1 - max(0, -current_drawdown / max_dd)) ** 2
    return base * dd_factor
```

Three layers of caution: half-Kelly, hard cap, drawdown overlay. Combined, this is structurally robust to the failure modes of pure Kelly.

## Multi-strategy Kelly

For a portfolio of $K$ strategies with mean vector $\mu$ and covariance matrix $\Sigma$, full Kelly is:

$$
f^* = \Sigma^{-1} \mu
$$

Same problem on steroids — now the noisy covariance matrix gets inverted, amplifying errors. **Multivariate Kelly is essentially never used at full size.**

Production:

```python
def multivariate_fractional_kelly(mu: np.ndarray, Sigma: np.ndarray,
                                    fraction: float = 0.25, gross_cap: float = 1.0) -> np.ndarray:
    """Fractional multivariate Kelly with shrinkage and gross cap."""
    from sklearn.covariance import LedoitWolf
    # Shrink the covariance for stability
    sigma_lw = LedoitWolf().fit(sample_returns).covariance_
    # Full Kelly direction
    raw = np.linalg.solve(sigma_lw, mu)
    # Scale to fraction
    weights = fraction * raw
    # Cap gross
    gross = abs(weights).sum()
    if gross > gross_cap:
        weights *= gross_cap / gross
    return weights
```

Cap, shrink, fractional — three layers protecting against the inversion's instability.

## Pitfalls

!!! warning "Sizing on a backtest Sharpe"
    Even before Kelly, the backtest Sharpe is upward-biased. Deflate it (Module 9 chapter 5) before plugging into Kelly.

!!! warning "Ignoring transaction costs"
    Kelly's formula doesn't account for transaction costs. The right modification: $\mu$ becomes net-of-costs, not gross.

!!! warning "Discretisation effects"
    For positions in whole shares, the optimal continuous Kelly might be 0.337 shares — you'd hold 0 or 1. The discretisation usually doesn't matter, but for tiny strategies on illiquid names, it can.

!!! warning "Volatility regime shifts"
    Kelly assumes stationary $\sigma$. In a vol regime change, your "correct" Kelly fraction changes dramatically. Vol-target instead of using a fixed Kelly fraction (next chapter).

## Bottom line

For sizing:

- **Half-Kelly or less** for any sample-estimated edge.
- **Multi-strategy Kelly with strong shrinkage + gross cap**.
- **Drawdown overlay** as the outer safety net.

The cost of leaving some growth on the table is real; the cost of going broke trying to capture it is final.

Continue to **[Vol targeting at portfolio level](04-vol-targeting.md)**.
