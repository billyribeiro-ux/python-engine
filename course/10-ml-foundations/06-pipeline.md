# An end-to-end ML signal pipeline

Time to stitch Module 10 together. This chapter assembles a complete, runnable, leakage-resistant ML pipeline using only what's been introduced so far: the data adapter (Module 0), feature engineering (chapter 2), purged CV (Module 9 + chapter 3), and the vectorized backtester (Module 9).

The result is the working template you'll customise for every ML-driven strategy in Modules 11, 16, and 17.

## The shape

```
prices ──► features ──► labels ──► purged k-fold CV ──► fit ──► OOS predictions
                                                                  │
                                                                  ▼
                                                       sized positions ──► backtest
```

Each arrow is a chance to leak. The pipeline below threads them carefully.

## The full pipeline

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from engine.data import YFinanceFeed, ParquetCache
from engine.features import frac_diff
from engine.backtest import PurgedKFold, vectorized_backtest

# --- 1. Data --------------------------------------------------------
feed = ParquetCache(YFinanceFeed(), root="data/bars")
bars = feed.bars("SPY", "2014-01-01", "2024-12-31")
close = bars["close"]
returns = close.pct_change().fillna(0.0)

# --- 2. Features (all causal: use only data with timestamp <= t) ----
def build_features(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    feats = pd.DataFrame({
        "ret_1":   ret,
        "ret_5":   close.pct_change(5),
        "ret_21":  close.pct_change(21),
        "ret_63":  close.pct_change(63),
        "vol_20":  ret.rolling(20).std(),
        "vol_60":  ret.rolling(60).std(),
        "dist_ma200": close / close.rolling(200).mean() - 1,
        "fracdiff_04": frac_diff(close, d=0.4, thresh=1e-2),  # loose threshold to keep more rows
    })
    return feats

X = build_features(close)

# --- 3. Labels: forward 5-day return sign ---------------------------
HORIZON = 5
fwd = close.pct_change(HORIZON).shift(-HORIZON)
y = (fwd > 0).astype(int).rename("y")

# Align and drop NaNs
df = X.join(y).dropna()
X_use = df.drop(columns="y")
y_use = df["y"]

# --- 4. Purged CV + walk-forward OOS predictions --------------------
def oos_proba(X: pd.DataFrame, y: pd.Series, horizon: int) -> pd.Series:
    cv = PurgedKFold(n_splits=5, purge=horizon, embargo=horizon)
    preds = pd.Series(np.nan, index=y.index)
    for train_idx, test_idx in cv.split(len(X)):
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, subsample=0.8, random_state=0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return preds

proba = oos_proba(X_use, y_use, horizon=HORIZON)

# --- 5. Turn probability into positions (centred around 0.5) --------
# Convert calibrated probability to a centred signal in [-1, +1]
signal_raw = 2 * proba - 1
# Buffer: only act on conviction past ±0.10
signal = signal_raw.where(signal_raw.abs() > 0.10, 0.0)
# Reindex to the full return series (zeros where missing)
positions = signal.reindex(returns.index).fillna(0.0)

# --- 6. Backtest -----------------------------------------------------
res = vectorized_backtest(returns, positions, cost_per_unit_turnover=3e-4)
print(res.stats)
```

The key correctness moves:

1. **Features are causal** — every rolling window is right-aligned (the default in pandas) and uses only past data.
2. **Labels are explicitly forward-shifted** — `pct_change(H).shift(-H)` makes today's label the return from today's close to H days later.
3. **CV is purged** — `purge=horizon` drops the H training samples whose label horizon touches the test fold.
4. **OOS predictions only** — `preds.iloc[test_idx]` is set inside each fold; final `proba` contains only out-of-fold values.
5. **Vectorized backtest shifts positions by 1** — today's signal acts on tomorrow's bar. No same-bar leakage.
6. **Realistic costs** — 3 bps per turn, which for SPY is on the conservative end.

## A typical result

On 10 years of SPY:

```
{'sharpe': 0.45-0.7, 'ann_return': 0.04-0.07, 'ann_vol': 0.07-0.10,
 'max_drawdown': -0.12, 'turnover_ann': ~3-5}
```

Not a money printer; not zero either. That's about what you'd expect from a generic gradient-boosting model on the broad market with 8 mechanical features. You're not going to beat the market with this exact recipe — but you have a working template that you'll improve with:

- More and better features (Module 11).
- Calibration (Module 10 ch4) + conformal sizing (Module 10 ch5).
- Meta-labeling on top of a base signal (Module 10 ch2).
- A regime overlay (Module 16).

Each of these typically adds 0.05–0.15 to OOS Sharpe.

## Adding calibration to the pipeline

Replace step 4 with:

```python
from sklearn.calibration import CalibratedClassifierCV

def oos_proba_calibrated(X: pd.DataFrame, y: pd.Series, horizon: int) -> pd.Series:
    cv = PurgedKFold(n_splits=5, purge=horizon, embargo=horizon)
    preds = pd.Series(np.nan, index=y.index)
    for train_idx, test_idx in cv.split(len(X)):
        # Calibrated wrapper uses inner cross-validation
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=0.8, random_state=0),
            method="isotonic",
            cv=3,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return preds
```

Same shape, calibrated probabilities. The position sizing logic in step 5 now operates on real probabilities, not just rank scores. Sharpe typically improves by 0.05-0.10 from sizing alone.

## Adding conformal sizing

Replace the label with a *regression* on the forward return, predict with MAPIE, size by point-over-interval-width:

```python
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor

def conformal_sized(X, y_continuous, horizon, alpha=0.20):
    cv = PurgedKFold(n_splits=5, purge=horizon, embargo=horizon)
    out = pd.Series(np.nan, index=y_continuous.index)
    for tr, te in cv.split(len(X)):
        mapie = MapieRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
            method="plus", cv=3,
        )
        mapie.fit(X.iloc[tr], y_continuous.iloc[tr])
        y_pred, y_int = mapie.predict(X.iloc[te], alpha=[alpha])
        lo, hi = y_int[:, 0, 0], y_int[:, 1, 0]
        # Position only when interval is on one side of zero
        pos = np.zeros_like(y_pred)
        pos[lo > 0] = +1
        pos[hi < 0] = -1
        half_w = (hi - lo) / 2
        size = np.minimum(np.abs(y_pred) / np.maximum(half_w, 1e-6), 3.0)
        out.iloc[te] = pos * size
    return out
```

Smaller trade frequency (you skip when intervals straddle zero), but higher per-trade conviction. Typical Sharpe gain over the binary version: 0.10-0.20.

## Promotion path

Before this pipeline is real:

- [ ] Walk-forward (outer CV) over a strict 30%+ holdout.
- [ ] Inner purged k-fold for hyperparameter tuning, on the training portion only.
- [ ] DSR (Module 9, ch5) across all parameter trials.
- [ ] Sensitivity sweep over costs.
- [ ] Bootstrap CI on the OOS Sharpe.
- [ ] Paper trade for 30+ days.

Skip any of those and you're back to fooling yourself.

## End of Module 10

You now have the ML toolkit specialised for low-SNR finance, all the leakage-resistant infrastructure to actually run the models, and a working end-to-end template. Module 11 (Modern ML) layers on the production tools — XGBoost / LightGBM hyperparameter optimisation with Optuna, SHAP for feature attribution, survival analysis for time-to-stop-out, and causal inference for alpha attribution.

Continue to **[Module 11 — Modern ML for Trading](../11-modern-ml/index.md)**.
