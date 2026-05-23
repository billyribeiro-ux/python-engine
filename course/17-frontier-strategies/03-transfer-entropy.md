# Transfer entropy networks

Correlation measures the linear, contemporaneous relationship between two series. **Transfer entropy** (Schreiber 2000) measures the *directional, non-linear* information flow from one series to another. For asset markets, it's how you build a lead-lag network — which symbols predict which.

Lead-lag relationships at short horizons are real and tradeable. Most algos detect them via simple lag correlation. Transfer entropy is more sensitive (catches non-linear) and gives an explicit direction.

## The definition

Transfer entropy from $X$ to $Y$ at lag 1:

$$
TE_{X \to Y} = \sum_{y_{t+1}, y_t, x_t} p(y_{t+1}, y_t, x_t) \log \frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}
$$

Reading: how much extra information does $X_t$ give you about $Y_{t+1}$, beyond what $Y_t$ already tells you?

If $X$ doesn't help predict future $Y$ given past $Y$, then $TE = 0$. If $X$ adds independent predictive information, $TE > 0$. The asymmetry $TE_{X \to Y} \neq TE_{Y \to X}$ is what makes this directional.

## Computing it in practice

In its raw form, TE requires estimating joint probabilities — slow and sample-hungry. The two practical estimators:

1. **Binned/histogram** — discretise each variable into bins, count joint occurrences. Simple but biased on small samples.
2. **Kernel-density / k-nearest-neighbour** — non-parametric, less biased, scales well in practice. The `pyinform` library implements both.

For trading, the Kraskov-Stögbauer-Grassberger (KSG) estimator using k-NN is the standard.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors


def transfer_entropy_kraskov(X: np.ndarray, Y: np.ndarray, k: int = 4, lag: int = 1) -> float:
    """KSG transfer entropy estimator: TE(X → Y) at the given lag.
    Both X and Y must be 1D arrays of equal length."""
    n = len(X)
    if n <= lag + k:
        return float("nan")
    # Build vectors
    y_next = Y[lag:].reshape(-1, 1)
    y_prev = Y[:-lag].reshape(-1, 1)
    x_prev = X[:-lag].reshape(-1, 1)

    # The three joint spaces
    joint_xyy = np.column_stack([x_prev, y_prev, y_next])
    joint_yy = np.column_stack([y_prev, y_next])
    joint_xy = np.column_stack([x_prev, y_prev])

    # KSG estimator: for each point find epsilon in joint_xyy, then count points in marginals
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(joint_xyy)
    eps = nbrs.kneighbors(joint_xyy)[0][:, k]

    def count_within(data, radius_per_point):
        nbrs_local = NearestNeighbors(metric="chebyshev").fit(data)
        return np.array([len(nbrs_local.radius_neighbors([data[i]], radius=radius_per_point[i],
                                                          return_distance=False)[0]) - 1
                          for i in range(len(data))])

    n_yy = count_within(joint_yy, eps)
    n_xy = count_within(joint_xy, eps)
    n_y = count_within(y_prev, eps)

    from scipy.special import digamma
    te = float(np.mean(digamma(k) - digamma(n_xy + 1) - digamma(n_yy + 1) + digamma(n_y + 1)))
    return max(0.0, te)         # numerical floor
```

This estimator is computationally heavy (~30 seconds per pair for 1000-point series). For production, use the `pyinform` library which has C-backed estimators.

## Building a lead-lag network

For a universe of N assets:

1. Compute TE between all pairs (N² values).
2. Threshold: keep only pairs where TE > some critical value.
3. The resulting directed graph is your lead-lag network.

```python
import pandas as pd
import networkx as nx


def lead_lag_network(returns: pd.DataFrame, threshold: float = 0.05) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(returns.columns)
    for x in returns.columns:
        for y in returns.columns:
            if x == y:
                continue
            te = transfer_entropy_kraskov(returns[x].values, returns[y].values, k=4, lag=1)
            if te > threshold:
                G.add_edge(x, y, weight=te)
    return G
```

In real markets, you typically find:

- A few "hubs" with high out-degree (information producers — usually large-caps, indices, or news-event-driven names).
- Many "sinks" with high in-degree (information consumers — often small-caps and ETFs).
- Sector-based clustering of edges.

## Trading the lead-lag

If asset $X$ has high TE to asset $Y$ at lag 1, the trade is straightforward:

- Watch $X$.
- When $X$ moves substantially, take a position in $Y$ in the same direction.
- Exit after lag 1 (one bar later).

```python
def lead_lag_strategy(returns: pd.DataFrame, network: nx.DiGraph,
                       trigger_z: float = 2.0) -> pd.DataFrame:
    """For each edge (x → y), if x's return is > trigger_z, take position in y."""
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for x, y in network.edges:
        x_z = (returns[x] - returns[x].rolling(60).mean()) / returns[x].rolling(60).std()
        signal = pd.Series(0.0, index=returns.index)
        signal[x_z > trigger_z] = +1
        signal[x_z < -trigger_z] = -1
        positions[y] += network.edges[x, y]["weight"] * signal.shift(1)
    return positions
```

In practice, the lag-1 edge usually carries 0.5-2 bps of edge per trade. At low costs and with many edges, this can produce a meaningful Sharpe.

## A worked example

```python
import numpy as np
import pandas as pd

# Synthetic: x leads y by 1 period
rng = np.random.default_rng(0)
T = 2000
x = rng.normal(0, 1, T)
y = 0.5 * np.roll(x, 1) + 0.5 * rng.normal(0, 1, T)
y[0] = 0

te_xy = transfer_entropy_kraskov(x, y, k=4, lag=1)
te_yx = transfer_entropy_kraskov(y, x, k=4, lag=1)
print(f"TE(X → Y): {te_xy:.4f}")
print(f"TE(Y → X): {te_yx:.4f}")
# TE(X → Y) should be substantially larger.
```

The asymmetry is the lead-lag signature. In real returns, the asymmetry is much weaker and you need careful sample sizes and significance testing.

## Significance via bootstrap

TE estimates have substantial noise. The standard significance test: **shuffle one of the series**, recompute TE, repeat. The shuffled TE distribution gives you a null distribution against which to test the observed value.

```python
def shuffled_te_pvalue(X: np.ndarray, Y: np.ndarray, n_shuffles: int = 100, **kw) -> float:
    observed = transfer_entropy_kraskov(X, Y, **kw)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(n_shuffles):
        Y_perm = rng.permutation(Y)
        null.append(transfer_entropy_kraskov(X, Y_perm, **kw))
    return float(np.mean(np.asarray(null) >= observed))
```

P-values < 0.05 (or after FDR correction across many pairs) are real lead-lag relationships.

## Pitfalls

!!! warning "Massive multiple testing"
    A 500-asset universe has 250,000 directed pairs. Even at p=0.001, expect 250 false positives by chance. Always FDR-correct.

!!! warning "Lag selection"
    TE depends on the chosen lag. Real markets have lead-lag at multiple horizons (microseconds to days). Test multiple lags; the strongest is often at the natural decision horizon of the strategy.

!!! warning "Sample size"
    KSG TE needs hundreds of samples per estimate for low variance. Daily data over a year (~250 points) is marginal; intraday with hundreds of bars per day is more comfortable.

!!! warning "Pre-filtering"
    Filter for high-correlation pairs first (they're more likely to also have high TE). Trying all pairs is computationally infeasible past a few dozen assets.

## Bottom line

Transfer entropy lead-lag networks:

- **Capture directional, non-linear** relationships invisible to correlation.
- **Are computationally expensive** — use `pyinform` for production speed.
- **Require careful significance testing** with bootstrap nulls and FDR correction.
- **Work best at high-frequency** where sample sizes are larger.

Pure-TE strategies tend to have modest Sharpe (0.3-0.5) but very low correlation to traditional strategies. The TE-network's deeper value is often as a *feature* in larger ML pipelines.

Continue to **[Signature methods on rough paths](04-signatures.md)**.
