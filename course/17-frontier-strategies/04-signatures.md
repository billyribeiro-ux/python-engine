# Signature methods on rough paths

The path signature, from rough path theory (Lyons 1998), is a way to turn a time series into a sequence of features that **uniquely characterise the path**. It's like a Fourier transform for paths — but the "frequencies" are iterated integrals that capture order, correlation, and non-linear interactions across channels. For irregularly-sampled, multivariate, possibly noisy time series, signatures are one of the most powerful feature extractors available.

This is genuinely frontier material. The math is non-trivial; the practical use is straightforward thanks to the `signatory` and `iisignature` libraries.

## The idea

For a path $X(t) = (X^1(t), X^2(t), ..., X^d(t))$, the signature at level $k$ is a tensor of iterated integrals:

$$
S^{i_1 i_2 \ldots i_k}(X)_{[s, t]} = \int_{s < t_1 < t_2 < \ldots < t_k < t} dX^{i_1}(t_1) \, dX^{i_2}(t_2) \cdots dX^{i_k}(t_k)
$$

For a $d$-dimensional path, the level-$k$ signature has $d^k$ entries. Concatenate up to some truncation level $M$ and you get a single feature vector of size $1 + d + d^2 + \ldots + d^M$.

The remarkable property: **the signature uniquely determines the path** (up to certain equivalence classes). Two paths with the same signature *are* the same path. Different paths have different signatures.

For ML on time series, this means: **the signature is a universal feature**. Any continuous function of the path can be approximated by a linear function of (a sufficiently truncated) signature. That's the universal approximation theorem for paths.

## Computing signatures in practice

```python
import numpy as np

try:
    import iisignature
except ImportError:
    iisignature = None


def path_signature(path: np.ndarray, level: int = 4) -> np.ndarray:
    """Compute the truncated signature of a multivariate path."""
    if iisignature is None:
        raise ImportError("pip install iisignature")
    return iisignature.sig(path, level)


def signature_size(d: int, M: int) -> int:
    """Number of signature features for d-dim path at truncation level M."""
    return sum(d ** k for k in range(M + 1))
```

For a 5-dimensional path (e.g., OHLCV) at level 4, signature size = 1 + 5 + 25 + 125 + 625 = 781.

## A worked example

```python
# Path: log-returns of (SPY, QQQ, TLT) over the last 60 days, with time as 4th channel
import pandas as pd
import numpy as np
from engine.data import YFinanceFeed

feed = YFinanceFeed()
spy = feed.bars("SPY", "2024-01-01", "2024-06-01")["close"].pct_change().dropna()
qqq = feed.bars("QQQ", "2024-01-01", "2024-06-01")["close"].pct_change().dropna()
tlt = feed.bars("TLT", "2024-01-01", "2024-06-01")["close"].pct_change().dropna()
df = pd.concat({"spy": spy, "qqq": qqq, "tlt": tlt}, axis=1).dropna()
df["time"] = np.arange(len(df)) / len(df)            # time as a channel
df["cum_spy"] = df["spy"].cumsum()                    # cumulative path (signatures want a path, not increments)
df["cum_qqq"] = df["qqq"].cumsum()
df["cum_tlt"] = df["tlt"].cumsum()

path = df[["time", "cum_spy", "cum_qqq", "cum_tlt"]].values    # (T, 4)
sig = path_signature(path, level=3)
print(f"Signature size: {len(sig)}")     # 1 + 4 + 16 + 64 = 85
```

For trading, the typical pipeline:

1. For each rolling window (e.g., past 60 days), compute the path signature of the multivariate path.
2. Use the signature as features in a downstream ML model (logistic regression, GBM, NN).
3. Train to predict the next-period return.

## Why signatures help

Three concrete things signatures buy you:

1. **Order matters** — unlike summary statistics (mean, std), signatures capture the *order* of events. (SPY +1% then TLT +1%) has a different signature from (TLT +1% then SPY +1%).
2. **Non-linear interactions** — cross-terms in the signature ($\int dX^1 \int dX^2$) capture interactions without explicit feature engineering.
3. **Universal** — for any continuous function of the path, there's a linear function of the signature that approximates it arbitrarily well.

For irregularly-sampled or asynchronous data (where standard time-series methods struggle), signatures are especially powerful. They don't care about even spacing.

## Signature-based ML

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def signature_features_for_windows(paths: list[np.ndarray], level: int = 3) -> np.ndarray:
    """Compute signature for each path window; stack into a feature matrix."""
    return np.stack([path_signature(p, level=level) for p in paths])


# Pretend we have:
# windows: list of (60, 4) arrays — one window per training point
# y: array of binary labels (next-period return sign)

X_sig = signature_features_for_windows(windows, level=3)
scaler = StandardScaler().fit(X_sig)
X_sig_scaled = scaler.transform(X_sig)

model = GradientBoostingClassifier().fit(X_sig_scaled, y)
```

The model now has 85 features (for 4-dim, level-3) that *uniquely* represent each window's path. Compare to hand-engineered features — signatures don't require domain expertise.

## Log-signature — a more efficient representation

The full signature has redundancy. The **log-signature** is a smaller, more efficient representation that retains all the same information.

```python
def path_log_signature(path: np.ndarray, level: int = 4) -> np.ndarray:
    return iisignature.logsig(path, level)
```

Log-signature size for level $M$, dimension $d$: about $d \cdot 2^M / M$. For (d=4, M=4): ~16 features instead of 341. Use it when you want fewer features.

## A frontier application: regime detection via signature distances

The signature is a high-dimensional vector. Distance between two windows' signatures measures how "similar" the underlying paths are — including order, cross-correlation, and dynamics.

```python
def signature_regime_distance(window1: np.ndarray, window2: np.ndarray,
                                level: int = 3) -> float:
    """Cosine distance between log-signatures of two paths."""
    s1 = path_log_signature(window1, level)
    s2 = path_log_signature(window2, level)
    return float(1 - np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
```

For regime detection: a large signature distance between today's window and last week's signals a regime change in the multi-asset path dynamics — beyond what single-series metrics catch.

## Pitfalls

!!! warning "Computational scaling"
    Signature size grows as $d^M$. Level 5 with 10 channels is 10^5 = 100,000 features — overkill for most ML. Stick to level 3-4 unless you have lots of data.

!!! warning "Time channel inclusion"
    Always add an explicit time channel to your path. Without it, signatures are invariant to time-reparameterisations (which is sometimes desirable, sometimes not).

!!! warning "Path lift**
    Signatures expect a *path*, not increments. Cumsum your returns before computing signatures, not the raw returns.

!!! warning "Library version pinning**
    `iisignature` and `signatory` have different APIs and slightly different output conventions. Pin a version and test carefully.

## Bottom line

Signatures are:

- **Universal features** for multivariate, possibly irregular time series.
- **Particularly good for high-frequency or asynchronous data**.
- **Computationally heavy** at high truncation levels.
- **Used in serious quant research** (Kidger, Foster, et al. at Oxford-Man and beyond).

For most strategies, signature features added to an existing ML pipeline (XGBoost, Module 11) add 0.05-0.15 of OOS IC — meaningful in a low-SNR setting.

Continue to **[Graph neural networks on cross-asset correlation graphs](05-gnn.md)**.
