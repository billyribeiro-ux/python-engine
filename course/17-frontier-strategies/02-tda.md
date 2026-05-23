# Topological data analysis on rolling correlation matrices

The correlation matrix of N asset returns over a rolling window has structure — clusters of co-moving stocks, "core-periphery" patterns, transitions between regimes. **Topological data analysis (TDA)** is a way to *summarise* that structure into a small number of features that respond to regime changes faster and more interpretably than scalar metrics.

This is one of the more interesting frontier applications. The papers that work tend to focus on **persistent homology** of correlation distances — the topology of the asset network, summarised by which holes and connected components appear at which scales.

## The picture

For an $N \times N$ correlation matrix $C$:

1. Convert to a distance matrix $d_{ij} = \sqrt{2(1 - C_{ij})}$. Highly correlated assets are close; uncorrelated are far.
2. Build a **filtration** of point clouds: at scale $r$, edges exist between assets with $d_{ij} \leq r$. As $r$ grows, more edges appear.
3. Compute the **persistent homology** of this filtration — when do holes (cycles) appear and when do they fill in?
4. Summarise with a **persistence diagram** (or barcode): each topological feature gets a (birth, death) pair.

The intuition: in normal regimes, the asset network has many small holes (sub-clusters of low-correlation assets). In a crisis, all correlations rise; holes get filled in quickly; the persistence diagram has different statistics.

## The TDA pipeline in `gudhi`

```python
import numpy as np
import gudhi
import pandas as pd


def correlation_persistence(C: np.ndarray, max_dim: int = 1):
    """Compute persistence diagram from a correlation matrix."""
    # Convert to distance matrix
    d = np.sqrt(np.maximum(0, 2 * (1 - C)))
    # Vietoris-Rips complex
    rips = gudhi.RipsComplex(distance_matrix=d, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
    return simplex_tree.persistence()


def persistence_summary(persistence) -> dict:
    """Reduce a persistence diagram to a few scalar features."""
    h0 = [(b, d) for dim, (b, d) in persistence if dim == 0 and d != float("inf")]
    h1 = [(b, d) for dim, (b, d) in persistence if dim == 1]
    return {
        "n_h0_features": len(h0),
        "total_h0_lifetime": sum(d - b for b, d in h0),
        "n_h1_features": len(h1),
        "total_h1_lifetime": sum(d - b for b, d in h1),
        "max_h1_lifetime": max((d - b for b, d in h1), default=0.0),
    }
```

The `gudhi` library is the standard for persistent homology. The Vietoris-Rips complex above scales to a few hundred points; for thousands of assets, use approximations (sparse VR, Alpha complex).

## Wasserstein distance between persistence diagrams

To detect regime change, compare today's persistence diagram to yesterday's. The right metric is the **Wasserstein distance** between persistence diagrams — like KL divergence but for topological summaries.

```python
def wasserstein_distance_diagrams(pd1, pd2, p: int = 2) -> float:
    """Wasserstein-p distance between two persistence diagrams (H1 features)."""
    h1_1 = np.array([(b, d) for dim, (b, d) in pd1 if dim == 1])
    h1_2 = np.array([(b, d) for dim, (b, d) in pd2 if dim == 1])
    if len(h1_1) == 0 and len(h1_2) == 0:
        return 0.0
    # gudhi has gudhi.wasserstein_distance for this
    return float(gudhi.wasserstein.wasserstein_distance(h1_1, h1_2, order=p))
```

The metric on its own is a regime-change signal: large day-to-day Wasserstein distances indicate that the correlation structure is reorganising.

## A worked example: rolling Wasserstein distance

```python
import numpy as np
import pandas as pd


def rolling_persistence_distance(returns_panel: pd.DataFrame, window: int = 60,
                                  step: int = 5) -> pd.Series:
    """Compute Wasserstein distance between successive persistence diagrams."""
    diagrams = {}
    distances = pd.Series(index=returns_panel.index, dtype=float)
    prev_pd = None
    for end in range(window, len(returns_panel), step):
        sub = returns_panel.iloc[end - window:end].corr().values
        try:
            pd_now = correlation_persistence(sub, max_dim=1)
            if prev_pd is not None:
                distances.iloc[end] = wasserstein_distance_diagrams(prev_pd, pd_now)
            prev_pd = pd_now
        except Exception:
            continue
    return distances.ffill().dropna()
```

A spike in this metric usually precedes (or coincides with) regime changes in the underlying market. Use as a regime-change scanner.

## Why TDA helps

Three concrete uses:

1. **Regime-change detection**. Wasserstein distance between consecutive persistence diagrams is sensitive to changes in the cluster structure that a simple correlation-mean isn't.
2. **Fragility scoring**. Persistence-diagram metrics (total H1 lifetime, max H1 lifetime) decay in crisis regimes — when the correlation network has no "holes" because everything's correlated.
3. **Cluster identification**. Persistent H0 features identify natural asset groupings beyond fixed sector classification.

## A worked example: fragility scanner

```python
def fragility_score(returns_panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """Total H1 lifetime in the rolling-window correlation matrix.
    Low values = fragile (clusters collapsed); high values = robust."""
    out = pd.Series(index=returns_panel.index, dtype=float)
    for end in range(window, len(returns_panel)):
        if end % 5 != 0:    # subsample for speed
            continue
        sub = returns_panel.iloc[end - window:end].corr().values
        try:
            pd_now = correlation_persistence(sub, max_dim=1)
            summary = persistence_summary(pd_now)
            out.iloc[end] = summary["total_h1_lifetime"]
        except Exception:
            continue
    return out.ffill()
```

A fall in this signal often warns of an imminent vol spike or correlation breakdown. Use as a portfolio-de-risking flag: when fragility score drops below its 1-year 20th percentile, reduce gross exposure by 30%.

## A real-world note

TDA-based features have shown up in published research on equity markets, cryptos, and FX. The strongest results come from combining TDA features with traditional ones (vol, correlation, momentum) — not from using TDA alone.

The cost: `gudhi` is slow on big inputs. For a 500-asset rolling correlation matrix updated daily, plan for ~minutes per snapshot. Approximations (sparse VR, Alpha complex) speed this up.

## Pitfalls

!!! warning "Persistence diagram sensitivity to outliers"
    A single outlier asset can dominate the topology. Robust-correlation methods (rank correlation, Spearman) often produce more stable diagrams.

!!! warning "Choice of distance metric"
    $\sqrt{2(1 - C)}$ is conventional but not unique. Different distances give different topologies. Pick one and stick with it.

!!! warning "Stability of summaries"
    A diagram's features depend on the algorithm's implementation details (max edge length, max dimension). Tune carefully.

!!! warning "Backtest leakage**
    Computing today's persistence diagram from today's correlation matrix uses today's return. Lag the input by 1 bar for honest signals.

## Bottom line

TDA on correlation matrices is genuine frontier research, occasionally used at quant funds and rarely at retail. Its strength: **detecting structural changes in the asset network that aren't visible in scalar metrics**. Its weakness: computationally heavy, conceptually demanding, hard to backtest at scale.

For most strategies, the value is in *one* TDA-derived feature added to a larger ML model — not in a pure-TDA strategy.

Continue to **[Transfer entropy networks](03-transfer-entropy.md)**.
