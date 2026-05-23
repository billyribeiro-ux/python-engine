# Stacking, ensembling, meta-labeling

Three closely-related ways to combine models. **Bagging** averages many noisy estimators (we covered it last chapter). **Stacking** trains a meta-model on out-of-fold predictions. **Meta-labeling** uses a secondary model to decide whether to act on a primary signal. All three reduce variance; all three carry leakage traps if done sloppily.

## Stacking — the right way

The pattern:

1. Split training data into $K$ folds.
2. For each fold, train each base model on the *other* $K-1$ folds and predict on the held-out fold.
3. Collect all out-of-fold predictions into a new feature matrix.
4. Train a *meta-model* on those OOF predictions to predict the original target.
5. For the test set, predict with each base model fit on all training data, then run the meta-model on those predictions.

Done correctly, stacking gives the meta-model honest inputs — predictions that none of the base models trained on. Done sloppily (training base models on all data and using their training predictions as features), it overfits massively.

```python
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from engine.backtest import PurgedKFold

def stacked_oof(base_models: dict, X: pd.DataFrame, y: pd.Series, horizon: int):
    """Return OOF prediction matrix (n_samples, n_models)."""
    cv = PurgedKFold(n_splits=5, purge=horizon, embargo=horizon)
    oof = pd.DataFrame(index=X.index, columns=list(base_models.keys()), dtype=float)
    for name, model in base_models.items():
        for tr, te in cv.split(len(X)):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te, oof.columns.get_loc(name)] = m.predict_proba(X.iloc[te])[:, 1]
    return oof.dropna()


def fit_meta(oof: pd.DataFrame, y: pd.Series):
    meta = LogisticRegression(C=1.0).fit(oof, y.loc[oof.index])
    return meta
```

For the test set, retrain each base model on all training data, predict the test, stack into a matrix, and apply the meta-model.

The meta-model is usually a **logistic regression** or a **shallow gradient boost**. Anything complex on top of already-complex base models overfits. The simpler the meta, the better it typically works.

## Diversity matters more than depth

Three GBMs with slightly different hyperparameters stacked together give you essentially no diversity — they all fit the same gradients. The meta-model can't combine them usefully.

Effective stacking comes from **diverse model families**:

- Gradient boosting + Random forest + Logistic regression + (sometimes) a shallow neural net.
- Gradient boosting on different feature subsets.
- The same model on different time-window subsets of training data.

The meta-model then has genuinely different inputs to weight.

## Meta-labeling — the López de Prado pattern

A particular form of stacking that's specifically good for trading.

**Primary model**: outputs a *direction* (long / short / flat). Could be a simple rule (momentum) or a learned model.

**Meta model**: a binary classifier that says "should I take this primary signal's trade?" Trained on `(primary_signal_features, additional_meta_features) → did_the_trade_make_money`.

The meta model's prediction is the **trade size**. If meta says 0.7, you trade 70% of your max position. If meta says 0.2, you skip (or trade tiny).

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from engine.features import triple_barrier_labels


def meta_label_pipeline(prices: pd.Series, primary_signal: pd.Series, meta_features: pd.DataFrame,
                        pt: float = 0.02, sl: float = 0.02, horizon: int = 10):
    # Only look at bars where primary signal is non-zero
    events = primary_signal[primary_signal != 0].index
    sides = primary_signal.loc[events]

    # Triple-barrier labels under the primary's direction
    out = triple_barrier_labels(prices, events=events, pt_sl=(pt, sl),
                                horizon_bars=horizon, side=sides)
    # Meta label: was the trade profitable?
    meta_y = (out["label"] == 1).astype(int)

    # Align meta features to events
    X = meta_features.reindex(events).dropna()
    meta_y = meta_y.loc[X.index]

    # Train meta (in production: use purged CV)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(X, meta_y)
    return model
```

The pattern's strength: **the primary model decides direction, which is usually the easy part. The meta model decides whether the setup is good enough to trade, which is usually the hard part.** A 60% primary-direction accuracy with a meta that triples conviction on the 30% of best setups can dramatically outperform a 60% primary at constant size.

## What meta features to use

For a primary that's a "buy momentum stocks" signal, useful meta features include:

- **Volatility regime** — meta tends to win less in high-vol regimes.
- **Liquidity** — meta works better on widely-traded names.
- **Past primary success** — recent track record for similar setups.
- **Cross-asset confirmation** — is the sector / market moving the same way?
- **Fundamental quality** — for equity, profitability and balance-sheet strength filter out fakes.

The meta features should be *orthogonal* to the primary features as much as possible — otherwise the meta is just re-discovering the primary's signal.

## Voting and weighted averages — the simplest ensemble

For classification, a **majority vote** across models:

```python
def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    P = np.stack(predictions)              # shape (n_models, n_samples)
    return (P.mean(axis=0) > 0.5).astype(int)
```

Or weighted by individual model's OOF accuracy:

```python
def weighted_average(predictions, weights):
    return np.average(np.stack(predictions), axis=0, weights=weights)
```

For regression, just average. Pre-compute weights from each model's OOF score. Astonishingly often, this simple aggregator outperforms stacking — fewer parameters, less to overfit.

## Stacking's killer trap: leakage in OOF construction

Done correctly:

```python
# 1. Base model trained on fold 1-4 → predicts fold 5
# 2. Base model trained on fold 1-3, 5 → predicts fold 4
# ...
# Meta features = OOF predictions
```

Done wrong:

```python
# 1. Base model trained on ALL data → predict ALL data
# 2. Use those predictions as meta features
```

The second leaks catastrophically — the base model has seen the labels it's predicting. The meta-model trained on those predictions appears to do well, fails in production. Use `engine.backtest.PurgedKFold` for the OOF construction, every time.

## A worked example: stacked + meta-labeled pipeline

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from engine.backtest import PurgedKFold

base_models = {
    "xgb": xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss"),
    "rf": RandomForestClassifier(n_estimators=200, max_depth=5),
    "lr": LogisticRegression(C=0.5, max_iter=1000),
}

# 1. OOF predictions from each base model (with purged CV)
oof = stacked_oof(base_models, X, y, horizon=5)

# 2. Meta features: the OOF base predictions plus 'context' features
context = pd.DataFrame({
    "vol_20": returns.rolling(20).std().reindex(oof.index),
    "regime_z": ...,
})
meta_X = pd.concat([oof, context], axis=1).dropna()
meta_y = y.loc[meta_X.index]

# 3. Train meta (with proper held-out CV inside)
meta = LogisticRegression(C=1.0).fit(meta_X, meta_y)
```

For production, both the OOF construction and the meta-training need their own outer walk-forward. The final OOS prediction is what the walk-forward yields.

## Pitfalls

!!! warning "Diversity through hyperparameters only"
    Two XGBoosts with `max_depth=3` vs `max_depth=4` are essentially the same model. For real diversity, change the model *family*, not the hyperparameters.

!!! warning "Meta model that's more complex than the base"
    A deep XGBoost meta on top of LR + GBM bases overfits. Keep the meta simple — usually LR with regularisation.

!!! warning "Stacking with very few samples"
    Stacking adds parameters (the meta model). Below ~1,000 samples, simple bagging or model averaging beats stacking.

!!! warning "Forgetting to retrain base models on all training data"
    For inference, you don't have OOF predictions — you have full-data predictions. Train each base model on *all* training data, predict the test set, and apply the meta. The CV is just for *building the meta's training data*.

## Bottom line

For combining models:

- **Bagging** (10 different seeds of the same model, average) — cheap, reliable variance reduction.
- **Stacking with diverse families** — XGBoost + RF + LR with a logistic meta, OOF via purged CV.
- **Meta-labeling** — primary signal decides direction, meta decides size. Best when you have a strong primary and want a sizing layer.
- **Simple weighted averages** with weights from individual OOF scores — surprisingly often the best.

Continue to **[SHAP for feature attribution](03-shap.md)**.
