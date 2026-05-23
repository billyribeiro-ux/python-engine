# Feature engineering for markets

In low-SNR finance, feature engineering pays more than model choice. A great gradient boosting model on average features generally loses to an average model on great features. This chapter is the catalogue of feature constructions that work, and the two label constructions (fractional differencing and triple-barrier) that need their own discussion.

## The feature taxonomy

For a return-prediction model on a single asset, useful feature families:

- **Return-based** — past returns at multiple horizons (1d, 5d, 21d, 63d, 252d); EWM of returns; momentum and reversal proxies.
- **Volatility** — realised vol at multiple horizons; GARCH-forecast vol; vol-of-vol; jump indicators.
- **Trend strength** — distance from moving averages; Hurst exponent; CUSUM-derived trend signals.
- **Microstructure** — bid-ask spread, depth imbalance, volume signatures, VPIN (Module 19).
- **Cross-asset** — correlation with index, lead-lag with related assets, regime-conditioned betas.
- **Time-of-day / seasonality** — minute-of-day, day-of-week, month-of-year, distance from earnings/FOMC.
- **Fundamental** — point-in-time valuation, earnings revisions, sentiment scores.
- **Alternative** — text sentiment, satellite imagery, web-scraped foot traffic, etc.

For most strategies, 30–80 well-chosen features beat 1,000 mechanically-generated ones. Quality over quantity.

## Fractional differencing for features (revisited)

We met fractional differencing in Module 7. Recap: standard first-differencing (returns) destroys long-memory structure. Fractional differencing finds the smallest order $d$ that achieves stationarity, preserving as much memory as possible.

The course implements it in `engine.features.frac_diff`. Use it like this:

```python
from engine.data import YFinanceFeed
from engine.features import frac_diff
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

feed = YFinanceFeed()
close = feed.bars("SPY", "2014-01-01", "2024-12-31")["close"]

# Sweep d to find the smallest value that passes ADF stationarity
for d in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]:
    series = frac_diff(close, d=d).dropna()
    if len(series) < 250:
        continue
    p = adfuller(series)[1]
    corr_with_raw = np.corrcoef(series.values, close.loc[series.index].values)[0, 1]
    print(f"d={d:.2f}  ADF p={p:.4f}  corr_with_raw={corr_with_raw:.3f}")
```

Typical output: `d ≈ 0.35–0.45` is the smallest value that passes ADF (p < 0.05) while keeping correlation with the raw price around 0.85–0.90. Compare to first differencing (`d = 1.0`) where correlation drops to ~0.05 — you've thrown away almost all the level information.

Use `frac_diff(close, d=0.4)` as a feature alongside returns. It carries level information that returns don't.

## Triple-barrier labels

For classification tasks, the standard label "did the next-day return beat 0?" is noisy and ignores path. **Triple-barrier** labels look at *which barrier was hit first* over a holding window:

- **Profit-take** (positive return threshold) hit first → label = +1.
- **Stop-loss** (negative threshold) hit first → label = -1.
- **Vertical** (time limit) hit first → label = 0.

The course implements it in `engine.features.triple_barrier_labels`:

```python
from engine.features import triple_barrier_labels

# Events: every 5 trading days
events = close.index[::5]
labels = triple_barrier_labels(
    close=close,
    events=events,
    pt_sl=(0.02, 0.02),         # +2% / -2% barriers
    horizon_bars=10,
)
print(labels.head())
print(labels["label"].value_counts(normalize=True))
```

The output is a DataFrame indexed by event time with columns `t1` (when the barrier was hit), `ret` (the realised return at hit), `label` (+1 / -1 / 0).

Two big advantages over fixed-horizon labels:

1. **Path-sensitive.** A trade that hits +2% in 3 days isn't treated the same as one that hits -1% then +2%.
2. **Risk-aware.** The labels reflect what would actually happen with a stop-loss in place.

## Meta-labeling

A López de Prado pattern: instead of one model that decides direction *and* size, use **two stages**:

1. **Primary model** decides direction (long/flat/short).
2. **Secondary "meta" model** decides whether to act on the primary signal — and with what size — based on additional features.

The meta model's label is "did the primary signal's trade hit profit-take (1) or stop-loss (0)?" The secondary model is a binary classifier on this.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Primary signal: simple cross-sectional momentum
primary_side = (returns.rolling(63).sum() > 0).astype(int) * 2 - 1   # +1 / -1

# Events whenever primary signal flips
events = primary_side.diff().fillna(0).abs() > 0
event_idx = events[events].index

# Triple-barrier on signed return (use side to flip barriers for shorts)
side = primary_side.loc[event_idx]
labels = triple_barrier_labels(
    close=close, events=event_idx,
    pt_sl=(0.03, 0.02), horizon_bars=15,
    side=side,
)
# Meta label: did the trade win?
meta_y = (labels["label"] == 1).astype(int)

# Meta features: anything we believe affects probability of success
meta_X = pd.DataFrame({
    "vol_20": returns.rolling(20).std().reindex(event_idx),
    "trend_strength": (close - close.rolling(252).mean()).reindex(event_idx) / close.reindex(event_idx),
    "distance_from_ma200": (close / close.rolling(200).mean() - 1).reindex(event_idx),
}).dropna()
meta_y = meta_y.loc[meta_X.index]

# Train (with purged CV in production; here illustrative)
meta_model = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(meta_X, meta_y)
proba = meta_model.predict_proba(meta_X)[:, 1]

# Size positions by meta probability * side
positions = pd.Series(side.loc[meta_X.index].values * proba, index=meta_X.index)
```

The pattern: **the primary model decides "what direction?"; the meta model decides "how much to bet?"**. Decoupling these two questions usually outperforms a single model trying to do both.

## Useful feature constructions

A handful that come up repeatedly:

```python
def returns_at(close: pd.Series, lookback: int) -> pd.Series:
    return close.pct_change(lookback)

def realised_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    return close.pct_change().rolling(lookback).std() * np.sqrt(252)

def distance_from_ma(close: pd.Series, lookback: int) -> pd.Series:
    return close / close.rolling(lookback).mean() - 1

def zscore(close: pd.Series, lookback: int) -> pd.Series:
    return (close - close.rolling(lookback).mean()) / close.rolling(lookback).std()

def rsi(close: pd.Series, lookback: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.where(delta > 0, 0.0).rolling(lookback).mean()
    down = (-delta.where(delta < 0, 0.0)).rolling(lookback).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def trend_skew(close: pd.Series, lookback: int = 60) -> pd.Series:
    """Asymmetry of recent returns — positive when upside has been larger than downside."""
    r = close.pct_change()
    return r.rolling(lookback).skew()
```

Use them in combinations. For most equity-prediction models, ~20-40 of these plus their EWM-counterparts and a few cross-asset features is plenty.

## A trap: features that contain the label

```python
features["lookback_5d"] = returns.rolling(5).sum()
labels = returns.shift(-5).rolling(5).sum().gt(0).astype(int)
```

If you compute features and labels carelessly, you can leak. Here `lookback_5d` is fine — it sums *past* 5 days. But if you wrote `features["lookback_5d"] = returns.rolling(5).sum().shift(-2)`, the feature contains 3 future bars.

The discipline: **every feature at time $t$ must use only data with timestamp $\leq t$.** Write feature functions as if they're streaming live — that constraint makes the bug impossible.

## A trap: scaling on the full sample

```python
# WRONG
from sklearn.preprocessing import StandardScaler
features_scaled = StandardScaler().fit_transform(features)        # leaks test stats
```

The scaler's mean and std are computed on all data including the test fold. The fix:

```python
# RIGHT — fit on train only, transform train + test
scaler = StandardScaler().fit(features.iloc[train_idx])
features_train = scaler.transform(features.iloc[train_idx])
features_test = scaler.transform(features.iloc[test_idx])
```

Or use a scikit-learn `Pipeline` that handles this for you:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scale", StandardScaler()), ("model", GradientBoostingClassifier())])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
```

## Pitfalls

!!! warning "Looking at importances without honesty"
    Feature importances from a tree model on training data are biased toward features that overfit. Use SHAP values on held-out predictions, or permutation importance on a holdout.

!!! warning "Engineering features into infinity"
    Past a few dozen features, more usually hurts because of variance and curse of dimensionality. Triple your feature count → re-verify CV improves before declaring victory.

!!! warning "Numerical instability from features at very different scales"
    GBMs are scale-invariant; linear models and neural nets are not. Always standardise for linear/NN.

## Bottom line

For features that win:

- **30-80 well-chosen features**, not 10,000.
- **Multi-horizon returns / vol / trend** as the core.
- **Fractional differencing** (Mod 7 + here) for level information without non-stationarity.
- **Triple-barrier labels** for classification, **forward returns** for regression.
- **Meta-labeling** when you have a primary signal and want a calibrated sizing layer.
- **All features causal**; all scaling fit inside the CV fold.

The `engine.features` module exposes `frac_diff` and `triple_barrier_labels`; the rest of the constructions are a few lines each and live alongside your strategy code.

Continue to **[Time-series cross-validation in practice](03-time-series-cv.md)**.
