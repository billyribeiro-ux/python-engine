"""Portfolio construction: Hierarchical Risk Parity (HRP).

Marcos López de Prado's HRP (2016). The algorithm has three steps:

1. **Distance metric.** Convert correlations to a proper distance:
   ``d_ij = sqrt(0.5 * (1 - rho_ij))``.
2. **Hierarchical clustering + quasi-diagonalisation.** Single-linkage
   cluster, then reorder the covariance matrix so similar assets are
   adjacent.
3. **Recursive bisection.** Walk down the cluster tree, splitting weight
   between halves in inverse proportion to each half's cluster variance.

The result is a long-only weight vector that is more *robust* to
noisy covariance estimates than Markowitz's analytic optimum — exactly
the right trade-off when your sample is short and your covariance is
mostly noise (i.e. always, in markets).

Pure NumPy / pandas / SciPy implementation. No mlfinlab dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to López de Prado's distance metric."""
    arr = np.sqrt(0.5 * (1.0 - np.clip(corr.values, -1.0, 1.0)))
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return leaf order from a linkage matrix."""
    link = link.astype(int)
    n_leaves = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    # Iteratively replace any internal-node id with its two children.
    while max(order) >= n_leaves:
        new_order: list[int] = []
        for node in order:
            if node < n_leaves:
                new_order.append(node)
            else:
                row = link[node - n_leaves]
                new_order.extend([int(row[0]), int(row[1])])
        order = new_order
    return order


def _ivp_weights(cov: pd.DataFrame) -> pd.Series:
    """Inverse-variance-portfolio weights for a (sub-)cov matrix."""
    iv = 1.0 / np.diag(cov.values)
    iv /= iv.sum()
    return pd.Series(iv, index=cov.index)


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    """Variance of the IVP-weighted sub-portfolio."""
    sub = cov.loc[items, items]
    w = _ivp_weights(sub).values.reshape(-1, 1)
    return float((w.T @ sub.values @ w)[0, 0])


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP weights from a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Per-period returns; rows are dates, columns are assets.

    Returns
    -------
    pd.Series
        Weights summing to 1.0, indexed by asset.
    """
    if returns.shape[1] < 2:
        raise ValueError("HRP requires at least 2 assets")
    cov = returns.cov()
    corr = returns.corr()
    dist = correlation_distance(corr)
    # Use the condensed-distance form scipy expects.
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    order_idx = _quasi_diag(link)
    sorted_cols = [cov.columns[i] for i in order_idx]

    # Recursive bisection.
    weights = pd.Series(1.0, index=sorted_cols)
    clusters: list[list[str]] = [sorted_cols]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r) if (var_l + var_r) > 0 else 0.5
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights.reindex(returns.columns).fillna(0.0) / weights.sum()
