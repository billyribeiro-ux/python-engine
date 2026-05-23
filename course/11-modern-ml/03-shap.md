# SHAP for feature attribution

You trained a gradient-boosted model. It works. Now your PM asks "why is it long Apple today?" or "which features are driving the new Sharpe regime?" SHAP values are the standard answer — they decompose each prediction into per-feature contributions. They're powerful and they have specific traps in time-series finance you need to know.

## What SHAP actually computes

For a prediction $\hat y(x)$ on input $x$, SHAP attributes a value $\phi_i$ to each feature $i$ such that:

$$
\hat y(x) = \phi_0 + \sum_i \phi_i
$$

where $\phi_0$ is the model's baseline (typically the mean prediction). $\phi_i$ is "how much did feature $i$ shift this prediction from baseline?"

The Shapley value formulation comes from cooperative game theory and has three properties:

1. **Local accuracy** — the $\phi_i$ sum to the prediction.
2. **Missingness** — features with no influence get $\phi_i = 0$.
3. **Consistency** — if a model changes such that a feature contributes more, its SHAP value goes up.

`TreeSHAP` computes exact SHAP for tree ensembles in polynomial time. For other models, `KernelSHAP` is a sampling approximation (slow but model-agnostic).

## Using SHAP on an XGBoost model

```python
import shap
import xgboost as xgb

model = xgb.XGBClassifier(...).fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# shap_values shape: (n_samples, n_features)
# For multi-class, list of arrays — one per class.

# Summary plot — features ranked by mean |SHAP| with distribution
shap.summary_plot(shap_values, X_test)

# Single-prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

For a binary classifier the explanation is in log-odds space. For regression it's in the target's units. For probability-space SHAP, pass `model_output="probability"` to `TreeExplainer`.

## Reading SHAP plots — beyond the obvious

A summary plot shows three things at once:

- **Feature ranking** (top to bottom) by mean absolute SHAP.
- **Distribution** of SHAP values per feature (the violin shape).
- **Feature value** colour (red high, blue low).

What to look for:

- **A feature whose effect direction reverses** (red on both sides of zero) — non-linear or interaction-driven.
- **A feature with high mean |SHAP| but no clear value-direction relationship** — important but acting through interactions, not monotonically.
- **A feature that's always near zero SHAP** — the model doesn't use it; consider dropping.

## The first finance-specific trap: SHAP on training data

```python
# WRONG
shap.summary_plot(explainer.shap_values(X_train), X_train)
```

SHAP on training data shows what the model fit to. Some of that is real signal; some is overfit noise. The features that look most important in training may be the ones the model is overfitting on.

**Always compute SHAP on a held-out (test or OOS) set.** The features that matter for *generalisation* are the ones that drive predictions on unseen data. The training-set picture is often misleading.

## The second trap: SHAP for autocorrelated panels

For a single asset's time series, SHAP correctly attributes each prediction to its features. For a panel of many symbols where features are autocorrelated across both axes (within-symbol over time *and* across-symbol at the same time), the standard SHAP interpretation can mislead.

Specifically:

- A symbol with persistently high `vol_60` will have high `vol_60`-SHAP on every bar, but the *bar-to-bar* contribution is approximately zero (the feature isn't moving).
- Two symbols that always move together will have correlated SHAP for the same feature, even though only one is "causing" the prediction.

For attribution at the daily level, look at **changes in SHAP** rather than levels:

```python
shap_diff = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns).diff()
# shap_diff[t, f] = how much did feature f move the prediction from t-1 to t?
```

This is what you'd report when asked "what changed today?"

## The third trap: SHAP after preprocessing

If your model is `Pipeline([scale, model])`, SHAP on `model` operates on the *scaled* features. The plots look fine but the feature values are z-scores, not the original units. Use `shap.Explainer(pipeline)` to get explanations in original units.

For one-hot encoded categoricals, `TreeSHAP` gives one $\phi$ per one-hot column. If you want the original categorical's total contribution, sum across its one-hot columns.

## Permutation importance — the cheap alternative

For a sanity check on SHAP rankings, use **permutation importance**: shuffle one column at a time, measure the drop in model score.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
importance_ranking = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
```

Permutation importance is **global** (one number per feature, not per prediction) and **conditional on the model** (it asks "how much does the model rely on this column?"). It's a useful cross-check; if SHAP says feature X is the top driver and permutation says it's the bottom, something's off.

## A trick: removing a feature and refitting

The gold standard for feature importance: drop the feature, refit, measure the score drop. Computationally expensive, but unambiguous. If SHAP and permutation importance disagree, this is the tiebreaker.

## A worked example: stable feature ranking via SHAP

```python
import shap
import pandas as pd
import numpy as np

from engine.backtest import PurgedKFold

def stable_shap_ranking(X, y, horizon=5, n_folds=5):
    cv = PurgedKFold(n_splits=n_folds, purge=horizon, embargo=horizon)
    all_shap = []
    for tr, te in cv.split(len(X)):
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X.iloc[tr], y.iloc[tr])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.iloc[te])
        # |SHAP| per feature per sample → average per fold
        all_shap.append(pd.Series(np.abs(sv).mean(axis=0), index=X.columns))
    by_fold = pd.concat(all_shap, axis=1)
    return by_fold.mean(axis=1).sort_values(ascending=False), by_fold.std(axis=1)

mean_rank, fold_std = stable_shap_ranking(X, y, horizon=5)
print(mean_rank.head(15))
print("Stability (lower = better):")
print((fold_std / mean_rank).head(15))
```

Features that consistently rank in the top across folds are real. Features that rank high in one fold and low in another are noise.

## Interaction values

SHAP can also decompose interactions: how much of the SHAP value of feature $i$ comes from interaction with $j$? `shap.TreeExplainer.shap_interaction_values` returns a `(samples, features, features)` array.

```python
interactions = explainer.shap_interaction_values(X_test)
# interactions[s, i, i] = main effect of feature i for sample s
# interactions[s, i, j] = interaction effect between i and j for sample s

# Average absolute interaction across samples
inter_mat = np.abs(interactions).mean(axis=0)
import matplotlib.pyplot as plt
plt.imshow(inter_mat); plt.colorbar()
```

Bright off-diagonal cells mean strong interactions. Useful for spotting "feature X matters only when feature Y is high" relationships.

## Pitfalls

!!! warning "SHAP on highly correlated features"
    When two features carry essentially the same information, SHAP splits the credit between them in unpredictable ways. The ranking of either alone is meaningless; together they matter. Cluster correlated features and report group importance.

!!! warning "Looking only at mean |SHAP|"
    The mean absolute SHAP averages out direction. A feature that's pushing predictions equally up and down looks important by |SHAP| but isn't actually driving the model's behaviour. Look at the violin shape, not just the bar.

!!! warning "Global SHAP plots on regime-mixed data"
    A feature that matters in volatile regimes and not in calm ones shows up as "moderately important on average." Compute SHAP per regime to surface this.

!!! warning "TreeSHAP off-policy"
    SHAP uses the model's marginal distribution; if your inference data is meaningfully off-distribution (e.g., a crisis the model never trained on), SHAP attributions may be unreliable.

## Bottom line

For feature attribution:

- **TreeSHAP on a held-out set** is the right default for gradient-boosted models.
- **Cross-fold stability** is the test of whether a feature is real or noise.
- **Cross-check with permutation importance** for global rankings.
- **For "what changed today?" attribution**, use SHAP *differences*, not levels.
- **Watch for the autocorrelation trap** in panel data.

Continue to **[Survival analysis — time to stop-out](04-survival.md)**.
