# Portfolio construction: Markowitz, Black-Litterman, risk parity

You have a basket of signals, each with an expected return and a covariance with all the others. How do you combine them into a single portfolio? This is the portfolio-construction problem. There are three classical answers — Markowitz, Black-Litterman, and risk parity — each with strengths and pathologies. A modern shop uses pieces of all three.

## Markowitz mean-variance

For weights $w$ on $N$ assets with expected returns $\mu$ and covariance $\Sigma$:

$$
\min_w \quad \tfrac{1}{2} w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mu = r_{\text{target}}, \quad w^\top \mathbf{1} = 1
$$

Closed-form solution exists. In code with `cvxpy` (which generalises to constraints):

```python
import cvxpy as cp
import numpy as np

def mean_variance(mu: np.ndarray, Sigma: np.ndarray, gamma: float = 1.0,
                  bounds: tuple = (-1.0, 1.0), gross: float = 1.0) -> np.ndarray:
    """Return weights that maximise mu - gamma/2 * w' Sigma w, with gross-leverage cap."""
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))
    cons = [cp.sum(w) == 0,              # market-neutral
            cp.norm(w, 1) <= gross,
            w >= bounds[0], w <= bounds[1]]
    cp.Problem(obj, cons).solve()
    return w.value
```

The killer problem with Markowitz: it is **wildly sensitive to the inputs**, especially $\mu$. Tiny changes in expected returns produce enormously different portfolios. Since $\mu$ is the noisiest thing in finance, the result is portfolios that look optimised but are actually fit to estimation error.

## The estimation-error trap

> "Markowitz portfolios are optimal for an investor with perfect knowledge of the future." — Bob Litterman, possibly apocryphal

A demonstration: simulate 5 assets with no real edge ($\mu = 0$). Estimate $\hat\mu$ from a short sample. Plug into the optimiser. You get extreme weights (often 100% long one asset, 100% short another). Run on a fresh sample → portfolio collapses.

The fix is **shrinkage**: pull the sample $\hat\mu$ and $\hat\Sigma$ toward sensible priors.

## Shrinking the covariance — Ledoit-Wolf

Replace the sample covariance $\hat\Sigma$ with a convex combination of the sample and a shrinkage target (often the identity scaled by mean variance):

$$
\Sigma_{\text{LW}} = \delta F + (1 - \delta) \hat\Sigma
$$

where $\delta$ is the analytically-derived optimal shrinkage intensity.

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf().fit(returns)
sigma_lw = lw.covariance_
```

For a portfolio with $N \approx T$ (e.g. 200 assets with 252 days of data), Ledoit-Wolf shrinkage is **essential**. Without it the inverse covariance is ill-conditioned and the optimisation blows up.

## Black-Litterman

The B-L framework starts from market-cap-implied returns (the prior — "what the market believes") and lets you blend in *views* with explicit confidence levels. The output is a posterior $\bar\mu$ that you feed into Markowitz.

The math is the multivariate Bayesian update. In code:

```python
def black_litterman(market_weights, Sigma, P, Q, Omega, tau=0.05, risk_aversion=2.5):
    """
    market_weights: prior weights (e.g. market cap weights)
    Sigma: asset covariance
    P: K x N pick matrix; row k picks the asset(s) view k is about
    Q: K x 1 expected returns for the views
    Omega: K x K view-confidence diagonal (smaller = more confident)
    """
    pi = risk_aversion * Sigma @ market_weights                   # implied prior returns
    A = np.linalg.inv(tau * Sigma)
    B = P.T @ np.linalg.inv(Omega) @ P
    posterior = np.linalg.solve(A + B, A @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    return posterior
```

If you have **no views**, the B-L posterior reduces to the prior — i.e. you'd just hold the market portfolio. If you have one view, it tilts proportionally toward the view-implied weights.

For trading strategies, the analogue is: the prior is "no edge" (zero mean returns), and your alpha signals are the views with confidence levels given by your historical out-of-sample track record. This is the principled way to combine signals: each signal's tilt is sized by its confidence.

## Risk parity

The premise: instead of equalising *dollar weights*, equalise *risk contributions*. Each asset contributes the same share of portfolio variance.

For asset $i$ with weight $w_i$:
- Marginal risk = $(\Sigma w)_i$
- Risk contribution = $w_i (\Sigma w)_i / \sigma_p$

Risk parity sets $w_i (\Sigma w)_i$ equal across all $i$. No closed form in the general case; it's an iterative solve.

```python
import numpy as np

def risk_parity(Sigma: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    n = Sigma.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ Sigma @ w)
        mrc = Sigma @ w / sigma_p                          # marginal risk contributions
        rc = w * mrc                                        # risk contributions
        target = sigma_p / n                                # equal target
        # Newton-like step
        w = w * target / rc
        w = w / w.sum()
        if np.max(np.abs(rc - target)) < tol:
            break
    return w
```

For long-only portfolios with $\mu$ unknown / unreliable, risk parity often outperforms mean-variance because it doesn't require estimating returns at all — only the covariance.

The famous example: equal-risk allocation between stocks, bonds, and commodities. Levered to the same vol as 100% stocks, it has historically delivered higher Sharpe than 100% stocks. (The Bridgewater All-Weather story.)

## Hierarchical Risk Parity (HRP)

López de Prado's HRP avoids the inverse-covariance instability of mean-variance entirely by **clustering** assets first, then allocating top-down.

The recipe (simplified):

1. Cluster assets by distance $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ (correlation-based) using a hierarchical-clustering algorithm.
2. Quasi-diagonalise: reorder assets so similar ones are adjacent.
3. Recursive bisection: split the sorted list in half, allocate inversely to the variance of each half, recurse.

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    corr = np.zeros_like(Sigma)
    std = np.sqrt(np.diag(Sigma))
    for i in range(n):
        for j in range(n):
            corr[i, j] = Sigma[i, j] / (std[i] * std[j])
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link)

    def bisect(items):
        if len(items) == 1:
            return {items[0]: 1.0}
        half = len(items) // 2
        left, right = items[:half], items[half:]
        wl = bisect(left)
        wr = bisect(right)
        vl = sum(Sigma[i, j] * w_i * w_j for i, w_i in wl.items() for j, w_j in wl.items())
        vr = sum(Sigma[i, j] * w_i * w_j for i, w_i in wr.items() for j, w_j in wr.items())
        alpha = vr / (vl + vr)
        return {**{i: w_i * alpha for i, w_i in wl.items()},
                **{i: w_i * (1 - alpha) for i, w_i in wr.items()}}

    weights_dict = bisect(list(order))
    w = np.zeros(n)
    for i, wt in weights_dict.items():
        w[i] = wt
    return w
```

In practice HRP often outperforms both Markowitz and naive risk parity on out-of-sample portfolios, because the clustering implicitly regularises the inverse-covariance problem.

## CVaR-constrained optimisation

For strategies whose returns are fat-tailed, mean-variance is the wrong objective. Use **Conditional Value-at-Risk** (CVaR — Module 6) as the risk measure:

```python
import cvxpy as cp
import numpy as np

def cvar_optimise(returns: np.ndarray, mu: np.ndarray, alpha: float = 0.95, gross: float = 1.0):
    """Maximise mu @ w subject to CVaR_alpha(R w) <= budget."""
    T, n = returns.shape
    w = cp.Variable(n)
    eta = cp.Variable()
    z = cp.Variable(T, nonneg=True)
    obj = cp.Maximize(mu @ w)
    cons = [
        cp.sum(w) == 0,
        cp.norm(w, 1) <= gross,
        z >= -returns @ w - eta,
        eta + (1 / (T * (1 - alpha))) * cp.sum(z) <= 0.02,
    ]
    cp.Problem(obj, cons).solve()
    return w.value
```

This is the linear-programming formulation of Rockafellar-Uryasev. It scales to thousands of scenarios (you can plug in bootstrapped or simulated return paths) and gives portfolios with much better tail behaviour than mean-variance.

## Which to use when

| Scenario | Recommended construction |
|---|---|
| Long-only, multi-asset class, no strong return views | **Risk parity** or **HRP** |
| Long-short, with quantitative signals | **Black-Litterman** with signals as views, then mean-variance with shrunk covariance |
| Tail-risk-sensitive (carry, short-vol) | **CVaR-constrained** |
| Single-strategy sizing | **Kelly** with vol target (next chapter) |
| Portfolio of strategies | Mean-variance on strategy returns with strong shrinkage |

## Pitfalls

!!! warning "Sample covariance with N>T"
    When you have more assets than observations, the sample covariance is singular. The inverse blows up. Always shrink, or use a factor model for the covariance.

!!! warning "Daily covariance for a weekly-frequency strategy"
    Estimating from daily and using for a weekly strategy double-counts noise. Use the frequency of your strategy.

!!! warning "Static portfolios in regime-changing markets"
    Re-optimise periodically (monthly or quarterly), and use an EWMA or rolling-window covariance so the recent regime is up-weighted.

!!! warning "Asymmetric costs and constraints get ignored"
    Long-only constraints, position-size caps, sector limits, turnover budgets — all of these change the optimal portfolio. `cvxpy` lets you encode them; use it.

## Bottom line

For a working portfolio construction layer:

- **Shrink the covariance** with Ledoit-Wolf or move to HRP entirely.
- **Get expected returns from a Black-Litterman blend** of "no view" and your alpha signals.
- **Optimise with CVaR constraint** for tail-risk-sensitive strategies, mean-variance otherwise.
- **Constrain turnover** in the objective so the optimiser doesn't propose impossibly active portfolios.

Continue to **[Sizing — Kelly, fractional Kelly, CVaR](06-sizing.md)**.
