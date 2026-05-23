# HRP and NCO — modern hierarchical methods

Module 8 chapter 5 introduced Hierarchical Risk Parity (HRP) as an alternative to mean-variance. This chapter shows the full algorithm and its more sophisticated cousin, Nested Clustered Optimization (NCO), both due to López de Prado. Both avoid the instability of inverting noisy covariance matrices — a notorious problem in mean-variance portfolio construction.

## Why mean-variance breaks

Markowitz's formula requires inverting the covariance matrix $\Sigma$. When $\Sigma$ is estimated from finite samples, it's poorly conditioned — small errors in the input produce large changes in the inverse and thus in the optimal weights.

For $N$ assets and $T$ observations, the sample covariance has condition number $\geq T / N$. When $T \approx N$, you're at the edge of singularity; when $T < N$ (more assets than data), it's actually singular.

Shrinkage (Ledoit-Wolf) helps a lot but doesn't eliminate the problem. HRP and NCO go further by avoiding the matrix inversion entirely.

## HRP — the algorithm

1. **Build distance matrix** from correlation: $d_{ij} = \sqrt{2(1 - \rho_{ij})}$.
2. **Cluster** assets hierarchically (single, complete, or Ward linkage).
3. **Quasi-diagonalise**: reorder assets so similar ones are adjacent. The reordered covariance has dense diagonal blocks.
4. **Recursive bisection**: split the ordered list in half. Allocate inversely to the variance of each half. Recurse within each half.

```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0, 2 * (1 - corr)))


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link)

    def cluster_var(weights, indices):
        sub = cov[np.ix_(indices, indices)]
        return float(weights @ sub @ weights)

    def bisect(items):
        """Recursive: return dict {item_idx: weight}."""
        if len(items) == 1:
            return {items[0]: 1.0}
        half = len(items) // 2
        left, right = list(items[:half]), list(items[half:])
        wl = bisect(left)
        wr = bisect(right)
        w_left = np.array([wl[i] for i in left])
        w_right = np.array([wr[i] for i in right])
        var_left = cluster_var(w_left, left)
        var_right = cluster_var(w_right, right)
        alpha = 1 - var_left / (var_left + var_right)
        return {**{i: w * alpha for i, w in wl.items()},
                **{i: w * (1 - alpha) for i, w in wr.items()}}

    weights_dict = bisect(list(order))
    w = np.zeros(n)
    for i, v in weights_dict.items():
        w[i] = v
    return w / w.sum()
```

## A worked example

```python
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# Generate noisy returns
rng = np.random.default_rng(0)
n_assets = 20
n_days = 252
true_loadings = rng.normal(0, 1, (n_assets, 3))
factor_returns = rng.normal(0, 0.02, (n_days, 3))
idiosyncratic = rng.normal(0, 0.01, (n_days, n_assets))
returns = factor_returns @ true_loadings.T + idiosyncratic

# Estimate covariance, then compute HRP weights
cov = LedoitWolf().fit(returns).covariance_
w_hrp = hrp_weights(cov)
print(f"HRP weights sum: {w_hrp.sum():.4f}")
print(f"Min weight: {w_hrp.min():.4f}, max: {w_hrp.max():.4f}")
print(f"Portfolio vol: {np.sqrt(w_hrp @ cov @ w_hrp):.4f}")

# Compare to inverse-variance weighting
inv_var = 1 / np.diag(cov)
w_iv = inv_var / inv_var.sum()
print(f"IV portfolio vol: {np.sqrt(w_iv @ cov @ w_iv):.4f}")
```

HRP weights tend to be more concentrated than naive inverse-variance because the bisection step intelligently weights clusters by their variance contribution.

## NCO — Nested Clustered Optimization

NCO is a sophisticated extension. The idea:

1. **Cluster** the assets (same as HRP).
2. **Within each cluster**, run mean-variance optimisation on that cluster only. Since within-cluster assets are correlated, the inverse is more stable.
3. **Across clusters**, treat each cluster as a single "synthetic asset" and run mean-variance on those.
4. **Combine**: each asset's final weight = within-cluster weight × cluster-level weight.

The conceptual win: mean-variance done on smaller, better-conditioned sub-problems.

```python
from scipy.cluster.hierarchy import fcluster


def nco_weights(cov: np.ndarray, mu: np.ndarray, n_clusters: int = 5):
    """Nested Clustered Optimization."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")

    n = cov.shape[0]
    w_intra = np.zeros(n)
    cluster_returns = []
    cluster_covs = []
    for c in range(1, n_clusters + 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        sub_cov = cov[np.ix_(idx, idx)]
        sub_mu = mu[idx]
        # Within-cluster mean-variance
        ones = np.ones(len(idx))
        try:
            inv = np.linalg.inv(sub_cov + 1e-6 * np.eye(len(idx)))
            w_c = inv @ ones
            w_c = w_c / w_c.sum()
        except np.linalg.LinAlgError:
            w_c = ones / len(idx)
        w_intra[idx] = w_c
        # Synthetic cluster
        cluster_returns.append(float(w_c @ sub_mu))
        cluster_covs.append(float(w_c @ sub_cov @ w_c))

    # Across-cluster mean-variance
    cluster_mu = np.array(cluster_returns)
    cluster_var = np.array(cluster_covs)
    inv_var = 1 / cluster_var
    cluster_w = inv_var / inv_var.sum()         # simple inverse-vol for clusters

    # Combine
    final = np.zeros(n)
    for c in range(1, n_clusters + 1):
        idx = np.where(labels == c)[0]
        final[idx] = w_intra[idx] * cluster_w[c - 1]
    return final / final.sum()
```

NCO with mean-variance at each level is more aggressive than HRP. It uses return forecasts ($\mu$) where HRP uses none — more potential alpha capture, more sensitivity to forecast errors.

## When HRP / NCO win vs mean-variance

| Scenario | Winner |
|---|---|
| Few assets ($N < 10$), good data | Mean-variance |
| Many assets ($N > 50$) with noisy covariance | HRP / NCO |
| No return forecasts | HRP |
| Strong return forecasts | NCO |
| $T \approx N$ or $T < N$ | HRP / NCO (mean-variance is singular) |

## Out-of-sample performance

López de Prado's original HRP paper (2016) showed HRP outperforming mean-variance out-of-sample on broad equity panels — by 5-15% Sharpe in many tests. The reason: stability. HRP weights are smoother across time, less driven by noisy covariance estimates.

For long-only multi-asset class allocation (the "all weather" use case), HRP is the modern default.

## Pitfalls

!!! warning "Linkage method matters"
    Single, complete, average, Ward — different linkage methods produce different clusters and different weights. Try them; pick by out-of-sample performance.

!!! warning "Cluster count for NCO"
    The number of clusters is a hyperparameter. For 50 assets, 5-10 clusters is a reasonable range. Cross-validate.

!!! warning "HRP is not optimal"
    HRP is robust, not optimal. With perfect knowledge of $\mu$ and $\Sigma$, mean-variance wins. In real-world conditions, HRP wins more often than not.

!!! warning "Recursive bisection's brittleness**
    The bisection step depends on the ordering from clustering. Different clustering choices can produce noticeably different weights. The bisection is also non-differentiable, so HRP doesn't compose well into a differentiable optimisation framework.

## Bottom line

For portfolios with many assets and noisy covariance:

- **HRP** for long-only or moderately-leveraged with no strong return views.
- **NCO** when you have return forecasts.
- **Mean-variance with Ledoit-Wolf shrinkage** for small portfolios where you trust the data.
- **Always backtest** the chosen method out-of-sample.

Continue to **[Random matrix theory for correlation cleaning](06-rmt.md)**.
