# XGBoost, LightGBM, CatBoost — when each wins

Three modern gradient-boosting libraries dominate practical ML. They all do roughly the same thing — boosted trees that minimise a loss — and they all win at different things. For finance, the right answer is "have all three installed and benchmark each on your problem." But the heuristics below get you 80% of the way without benchmarking.

## The picture

| Library | Best at | Notable |
|---|---|---|
| **XGBoost** | Reliable default. Best documented. Mature deployment story. | The original; many years of incremental tuning. |
| **LightGBM** | Wide datasets (thousands of features), large samples. Generally fastest. | Leaf-wise growth; aggressive regularisation matters. |
| **CatBoost** | Categorical features at scale. Ordered boosting reduces leakage. | Slowest to train, often best out-of-box. |

For a panel of S&P 500 daily bars with 80 features, all three will get you within ~0.01 OOS IC of each other. Pick on training speed and ecosystem fit, not on the assumption that one is universally best.

## A working XGBoost setup

```python
import xgboost as xgb

def fit_xgb(X_train, y_train, X_valid, y_valid):
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        gamma=0.5,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=0,
        tree_method="hist",       # the fast default
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model
```

Three knobs that matter most:

1. **`max_depth`** — start at 4. Anything above 6 for low-SNR data is begging to overfit.
2. **`min_child_weight`** — the minimum number of samples per leaf. For noisy data, push it up (10-50). Default 1 is too aggressive.
3. **`subsample` + `colsample_bytree`** — bagging-style randomness. Both around 0.5-0.8 add real variance reduction.

`early_stopping_rounds=50` with a held-out validation set is non-negotiable. Otherwise you train until the in-sample loss bottoms out, which is overfit by construction.

## LightGBM — when speed matters

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,                # the leaf-wise analog of max_depth
    min_data_in_leaf=20,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    random_state=0,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
```

LightGBM grows trees leaf-wise (greedy on best splits) rather than depth-wise. That makes it faster, but also more prone to overfit. Strongly tune `num_leaves` and `min_data_in_leaf`.

For a panel of 5,000 stocks × 8 years × 80 features (~25M rows), LightGBM trains 5-10× faster than XGBoost. For 500 rows, you won't notice the difference.

## CatBoost — when categoricals dominate

```python
import catboost as cb

model = cb.CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    bagging_temperature=0.5,
    cat_features=["sector", "exchange"],   # named or indexed
    random_seed=0,
    od_type="Iter", od_wait=50,            # early stopping
    verbose=False,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

CatBoost's killer feature is **ordered boosting** for categoricals — it uses a target-encoding scheme that avoids leakage on the categorical features themselves. For models with many cross-sectional cats (sector, industry, exchange, country), CatBoost frequently wins by 0.01-0.02 IC.

## Hyperparameter optimisation with Optuna

A clean Optuna setup that respects our PurgedKFold:

```python
import optuna
from sklearn.metrics import log_loss

from engine.backtest import PurgedKFold

def objective(trial, X, y, horizon):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    cv = PurgedKFold(n_splits=4, purge=horizon, embargo=horizon)
    losses = []
    for tr, te in cv.split(len(X)):
        m = xgb.XGBClassifier(n_estimators=500, eval_metric="logloss",
                              early_stopping_rounds=30, **params)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
        p = m.predict_proba(X.iloc[te])[:, 1]
        losses.append(log_loss(y.iloc[te], p))
    return float(np.mean(losses))

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(lambda t: objective(t, X, y, horizon=5), n_trials=50)
print(study.best_params)
```

Three Optuna features worth knowing:

- **TPE sampler** (default) — Bayesian-ish search; better than random and grid for the same trial count.
- **Pruners** — kill unpromising trials early. `MedianPruner` is the easiest.
- **`study.trials_dataframe()`** — full audit trail; useful for sanity-checking the search.

For most quant tasks, 50-100 Optuna trials is the sweet spot. Past that, you're tuning noise.

## A serious anti-pattern

```python
# WRONG — leaking the test set via early stopping
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          early_stopping_rounds=50)
```

The model uses `X_test` to decide when to stop. The test set is no longer truly held out. The fix is a **three-way split** — train, validation (for early stopping), test (untouched).

If you only have CV folds (no held-out test), use the test fold for early stopping but report the score with calibrated awareness: the CV score is slightly optimistic because of this.

## Categorical encoding for XGBoost and LightGBM

Both XGBoost (>1.5) and LightGBM support categorical features natively via the `enable_categorical` flag (XGB) or by passing `categorical_feature=...` (LGB). Use these instead of one-hot encoding:

- One-hot for high-cardinality cats explodes the feature space.
- Target encoding leaks unless done very carefully (CatBoost's ordered-boosting trick).
- Native categorical handling is now both faster and better.

## Bagging-style ensembling

A cheap and powerful trick: train the same model with different seeds and average:

```python
def bagged_predict(X_train, y_train, X_test, n_models=10):
    preds = []
    for seed in range(n_models):
        m = xgb.XGBClassifier(random_state=seed, **best_params).fit(X_train, y_train)
        preds.append(m.predict_proba(X_test)[:, 1])
    return np.mean(preds, axis=0)
```

The variance reduction is real, especially for small samples. For 10 models, expect a 0.01-0.02 Sharpe gain on OOS for free.

## A reproducibility note

GBM libraries are **mostly** deterministic with a fixed seed, but exact reproducibility requires:

- Fixing all `random_state` / `random_seed` / `seed` arguments.
- Setting `nthread=1` (or being okay with tiny floating-point variation from threaded reductions).
- Pinning library versions in your environment.

For research, slight non-determinism is fine. For *publishable* benchmarks, lock everything down.

## Pitfalls

!!! warning "n_estimators with early stopping"
    The reported `best_iteration` in the model is the actual number used. Don't report `n_estimators=2000` if the model stopped at 240.

!!! warning "Default eval_metric"
    Both XGB and LGB default to a loss that's not always what you want for finance. Set it explicitly: `logloss` for binary classification, `rmse` for regression. For ranking, use `ndcg`.

!!! warning "Tree models on standardised features"
    GBMs are scale-invariant. Standardising features is pointless and hides debugging.

!!! warning "Class imbalance via class_weight"
    For very imbalanced labels (e.g., 95% zero-class), use `scale_pos_weight` (XGB) or `class_weight` (LGB) — not undersampling. The latter throws away information.

## Bottom line

A working modern-ML stack for finance:

1. **XGBoost as the baseline** — well-documented, reliable.
2. **LightGBM for speed** — when you sweep many configurations or have big panels.
3. **CatBoost for categorical-heavy panels** — sector, industry, exchange.
4. **Optuna with TPE** for 50-100 trials of hyperparameter search.
5. **Bagged ensemble of 5-10 models** for variance reduction.
6. **Three-way split** (train / val / test) or proper purged CV.

That's the modern production recipe. The next chapter goes deeper into how to combine models — stacking, meta-labeling, and a few patterns that aren't as common as they should be.

Continue to **[Stacking, ensembling, meta-labeling](02-stacking.md)**.
