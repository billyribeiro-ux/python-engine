# Calibration — Platt, isotonic, and why you need it

A classifier outputs a "probability" between 0 and 1. If that probability is well-calibrated, then among all the samples the model assigned probability 0.7, *exactly 70% should be class 1*. If it's not well-calibrated, you have a score — useful for ranking — but not a probability you can multiply against to size positions or compute expected values.

In trading, you usually want both. This chapter covers what calibration is, how to measure it, and how to fix a miscalibrated model.

## Why finance cares about calibration

Three reasons sizing depends on calibration:

1. **Kelly sizing** — $f^* = (p \cdot b - (1 - p)) / b$ for a binary bet. If your "probability" is actually a rank-only score, the formula gives wrong sizes.
2. **Expected-value gates** — "trade only when expected return > 5 bps" assumes the probability is real, not just a score.
3. **Meta-labeling** (chapter 2) — the meta model's probability is multiplied into position size. Miscalibration directly distorts size.

## Diagnosing miscalibration

The standard plot: **reliability diagram**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy="quantile")
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Predicted probability"); plt.ylabel("Empirical fraction")
plt.show()
```

If the curve is below the diagonal, the model over-states probabilities (over-confident). If above, it under-states (under-confident). Perfect calibration is on the diagonal.

A single-number summary: **Brier score**.

$$
\text{Brier} = \frac{1}{N} \sum_i (\hat p_i - y_i)^2
$$

Lower is better. It decomposes into reliability (calibration) + resolution (discrimination) − uncertainty. Compare two models by Brier score and you get both their accuracy and their calibration in one number.

## Why models are miscalibrated

**Tree-based models** (random forests, gradient boosting) tend to be poorly calibrated. They output scores that look like probabilities but aren't. Even a single decision tree returns "fraction of class 1 in this leaf" — for small leaves, that's noisy and biased toward 0/1.

**Linear classifiers** (logistic regression, SVM with Platt) are roughly calibrated out of the box — logistic regression optimises the log-loss, which is a proper scoring rule.

**Neural nets** with cross-entropy loss are usually under-confident on training data (because of regularisation, dropout, weight decay). On test data they're often over-confident — modern temperature-scaling literature exists for this.

## Platt scaling

Logistic regression fit on top of the model's raw scores. Two parameters ($A$, $B$):

$$
p_{\text{calibrated}}(x) = \frac{1}{1 + \exp(A \cdot s(x) + B)}
$$

where $s(x)$ is the model's raw score.

```python
from sklearn.calibration import CalibratedClassifierCV

base = GradientBoostingClassifier(...)
calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
```

The wrapper internally cross-validates: it splits training data, fits the base model on one part, fits a Platt scaler on the other, and combines them. The result is a model with a well-calibrated `predict_proba`.

## Isotonic regression

Non-parametric calibration: fit a monotonic step function that maps raw scores to calibrated probabilities. More flexible than Platt; needs more data (~1000+ samples for stable fits).

```python
calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
```

For most financial datasets (a few thousand samples), Platt and isotonic are roughly tied. Isotonic wins when the miscalibration is non-monotonic (e.g., a model that's over-confident at the extremes and under-confident in the middle).

## A worked example: calibrating a return-direction classifier

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Pretend we have features X and labels y already
# (1 if forward 5d return > 0, else 0)
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(2000, 8)))
y_cont = X.iloc[:, 0] * 0.05 + rng.normal(0, 1, size=2000)
y = (y_cont > 0).astype(int)

# Train / calibrate / test split
train, calib, test = X.iloc[:1200], X.iloc[1200:1600], X.iloc[1600:]
y_train, y_calib, y_test = y.iloc[:1200], y.iloc[1200:1600], y.iloc[1600:]

raw = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(train, y_train)
print(f"Raw Brier: {brier_score_loss(y_test, raw.predict_proba(test)[:, 1]):.4f}")

cal = CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=100, max_depth=3),
                              method="isotonic", cv=3).fit(pd.concat([train, calib]),
                                                            pd.concat([y_train, y_calib]))
print(f"Calibrated Brier: {brier_score_loss(y_test, cal.predict_proba(test)[:, 1]):.4f}")
```

Expected pattern: calibrated Brier is similar or slightly better than raw, but the reliability diagram is much closer to the diagonal. The actual sizing decisions you'd make from the calibrated model are far more honest.

## Calibration in a CV pipeline

For walk-forward, the calibrator must be re-fit each fold:

```python
def fit_predict_calibrated(X_train, y_train, X_test, label_horizon=5):
    base = GradientBoostingClassifier(...)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal.predict_proba(X_test)[:, 1]
```

The `cv=3` is the *inner* CV used for calibration — independent of your outer walk-forward.

## Pitfalls

!!! warning "Calibrating on training data"
    Always calibrate on data the *base model didn't see*. `CalibratedClassifierCV` handles this via its own CV; if you're hand-rolling, hold out a calibration set.

!!! warning "Isotonic on small samples"
    With < 500 samples, isotonic overfits the calibration set. Use Platt (sigmoid) instead.

!!! warning "Calibration after threshold tuning"
    If you also tune a decision threshold (e.g., "trade only if p > 0.55"), the threshold should be selected on a separate set from both training and calibration. Three holdouts: train, calibrate, threshold-tune. Tedious but correct.

!!! warning "Calibrating across regimes"
    A model calibrated on quiet markets may be miscalibrated in volatile ones. For regime-aware strategies, calibrate per-regime or recalibrate online with a rolling window.

## Bottom line

For any classifier whose outputs you'll multiply or threshold:

- **Check calibration with a reliability diagram and a Brier score**.
- **Use `CalibratedClassifierCV`** with `method="isotonic"` (preferred) or `method="sigmoid"` (when sample-starved).
- **Calibrate inside the CV fold**, not before.
- **Don't size on uncalibrated scores** — you'll over- or under-bet systematically.

Continue to **[Conformal prediction with MAPIE](05-conformal.md)**.
