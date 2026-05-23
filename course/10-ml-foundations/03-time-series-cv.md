# Time-series cross-validation in practice

Module 9, chapter 4 introduced `PurgedKFold` and `WalkForward` for backtesting. For ML model selection — picking hyperparameters, comparing model families — you use the same splitters but with different semantics. This chapter is the working recipe.

## The two CV roles

In a typical ML workflow there are two distinct cross-validation tasks:

1. **Inner CV** — for hyperparameter selection inside the model. "Which `max_depth` for XGBoost?" Run a small CV inside your training set.
2. **Outer CV** — for honest performance estimation. "What Sharpe does this whole pipeline (model + chosen hyperparameters) achieve out-of-sample?" Run walk-forward over a held-out section of history.

Use **`PurgedKFold`** for the inner loop and **`WalkForward`** for the outer.

## A working pattern

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from engine.backtest import PurgedKFold, WalkForward

class _AsSklearn:
    """Adapter so engine.backtest.PurgedKFold works with sklearn's GridSearchCV."""
    def __init__(self, cv): self.cv = cv
    def split(self, X, y=None, groups=None):
        for tr, te in self.cv.split(len(X)):
            yield tr, te
    def get_n_splits(self, X=None, y=None, groups=None): return self.cv.n_splits


def fit_with_cv_hyperparams(X: pd.DataFrame, y: pd.Series, label_horizon: int = 5):
    cv = PurgedKFold(n_splits=5, purge=label_horizon, embargo=label_horizon)
    grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 200]}
    search = GridSearchCV(
        GradientBoostingClassifier(random_state=0),
        param_grid=grid,
        cv=_AsSklearn(cv),
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def walkforward_predictions(X: pd.DataFrame, y: pd.Series, label_horizon: int = 5):
    wf = WalkForward(initial_train=252 * 3, test_size=21, embargo=label_horizon)
    preds = pd.Series(np.nan, index=y.index)
    for train_idx, test_idx in wf.split(len(X)):
        # Inside each outer fold, re-tune hyperparameters on the training data
        train_idx_safe = train_idx[:-label_horizon]   # purge label tail
        model, _ = fit_with_cv_hyperparams(X.iloc[train_idx_safe], y.iloc[train_idx_safe])
        preds.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return preds.dropna()
```

The flow:

1. The **outer walk-forward** simulates monthly retraining in production.
2. **Inside each fold**, an **inner purged k-fold** picks hyperparameters on training data only.
3. The chosen model predicts the next month.
4. After the full walk, `preds` contains out-of-sample probabilities for every test bar.

This is the most-leakage-resistant standard pipeline.

## Why nested CV matters

Without nested CV — i.e., picking hyperparameters on the *same* fold you evaluate on — your "out-of-sample" score is upward biased. The bias is typically small for tightly-tuned models on big datasets and large for loosely-tuned models on small ones (which is exactly the financial setting).

If nested CV is too expensive (you re-fit hyperparameters every outer fold), a reasonable compromise:

- Pick hyperparameters **once** on the first half of history.
- Walk-forward through the second half with those frozen hyperparameters.

This is "blocked" tuning. It's biased less than no-CV-at-all, more than nested CV.

## A sample size rule of thumb

For each PurgedKFold inner fold, you want at least ~1 year (252 bars) of *training* data after the purge. With `n_splits=5` and a 5-day purge, that means the total training set should be at least about 1300 bars (~5 years of daily data).

For shorter samples, drop `n_splits` to 3 or move to expanding-window inner CV.

## Stratification on imbalanced labels

If your labels are imbalanced (e.g. 80% zero-class), the random PurgedKFold folds may end up with very different class distributions. The fix: stratify within the time order.

```python
def stratified_purged_kfold(y: pd.Series, n_splits: int, purge: int, embargo: int):
    """Stratified version: each fold has approximately the same class ratio."""
    # Yield train/test indices, balancing classes within time-ordered folds.
    # Implementation: sort by time, then alternate classes into folds.
    ...
```

For binary classification with strong imbalance, stratification on the inner CV is worth the few lines.

## Cross-sectional panels

For panel data (multiple symbols at each date), there are **two** time axes to worry about:

1. **By date**: standard purged k-fold on the date index, treating each date's full cross-section as one "sample" of N rows.
2. **By symbol-date**: split rows individually; risk of having SPY-2020 in train and QQQ-2020 in test, which the model can correlate on.

For most cross-sectional ML, **split by date** is the right choice. The model is trained on certain dates and tested on others; within a date, all symbols are train or all are test.

```python
def by_date_split(panel: pd.DataFrame, n_splits: int):
    dates = panel.index.get_level_values("date").unique().sort_values()
    cv = PurgedKFold(n_splits=n_splits, purge=5, embargo=5)
    for train_d_idx, test_d_idx in cv.split(len(dates)):
        train_dates = dates[train_d_idx]
        test_dates = dates[test_d_idx]
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        test_mask = panel.index.get_level_values("date").isin(test_dates)
        yield panel.index[train_mask], panel.index[test_mask]
```

## Avoid this anti-pattern

```python
# WRONG — leaks via the scaler
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)         # uses test stats
for train_idx, test_idx in cv.split(len(X)):
    model.fit(X_scaled[train_idx], y[train_idx])
```

The scaler must be fit *inside* the fold:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scale", StandardScaler()), ("model", GradientBoostingClassifier())])
for train_idx, test_idx in cv.split(len(X)):
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = pipe.predict(X.iloc[test_idx])
```

Pipelines aren't just convenience; they're the correctness mechanism.

## Pitfalls

!!! warning "Refitting the entire pipeline each fold is slow"
    For an expensive model (deep nets), nested CV with 5×5 folds is 25 fits. Use early stopping, smaller search spaces, or move to a hyperparameter optimiser like Optuna with TPE that prunes early.

!!! warning "Purge size mismatching label horizon"
    Always set purge ≥ label horizon. For a 5-day forward return label, `purge=5` is the minimum. Larger doesn't hurt much; smaller leaks.

!!! warning "Test fold smaller than the label horizon"
    If your test fold is shorter than your label horizon, you can't even fully evaluate a single sample. Choose `test_size >= 2 * label_horizon` at minimum.

!!! warning "Reusing CV results for the final model"
    The "best" hyperparameters from CV are biased by selection. The model you ship should be re-fit on all training data with those hyperparameters, but its expected live score is the *CV* score, not the in-sample fit score.

## Bottom line

For ML model selection on finance:

- **Inner CV with `PurgedKFold`** (5 splits, purge = label horizon) for hyperparameter choice.
- **Outer walk-forward** for live-like performance estimation.
- **Always inside a `Pipeline`** so scalers, encoders, and feature selectors don't leak.
- **Refit on all train data** for the final ship; **trust the CV score, not the fit score**, for expected live performance.

Continue to **[Calibration — Platt, isotonic, and why you need it](04-calibration.md)**.
