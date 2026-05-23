# The bias-variance picture in low-SNR finance

In most machine learning problems — image classification, language modeling, recommendations — your model is bottlenecked by **representation power**. You add more parameters, more data, more compute, and the model gets better.

In financial prediction, your model is bottlenecked by **signal-to-noise ratio**. The "true" relationship between features and next-bar return has a tiny effect on top of an enormous amount of noise. Adding parameters mostly fits the noise; adding more data lowers variance but rarely changes the irreducible-error floor. The right mental model is **upside-down**.

## The numbers, roughly

For daily equity returns:

- Cross-sectional momentum's per-bar information coefficient (IC, the Spearman rank correlation between signal and forward return) is typically **0.02–0.05**.
- A genuinely great fundamental factor signal might hit IC = 0.10. Sustained.
- Image classifiers achieve top-1 accuracies of 80–95%. That maps to per-prediction IC > 0.9.

You are operating at roughly **one twentieth to one fortieth the signal-to-noise** of computer vision. That's not a marginal difference; that's a different planet.

## Consequences

### 1. Simple models win.

In high-SNR domains, deep nets dominate. In low-SNR finance, the typical leaderboard is:

- **Gradient-boosted trees** (XGBoost, LightGBM, CatBoost) — usually the best in absolute terms.
- **Linear models with thoughtful features** — close behind, sometimes wins.
- **Random forests** — solid baseline, simpler to tune.
- **Deep nets** — competitive for tasks with massive datasets (e.g. limit-order-book prediction) and dramatically worse on small ones.

The reason: low SNR means the *variance* of your estimator (sensitivity to training noise) dominates the *bias*. Simple models have low variance. Complex models have high variance. The bias-variance trade-off, in this regime, almost always favours simpler.

### 2. Regularisation is non-negotiable.

L1/L2 on linear models, max-depth + subsample on trees, weight decay + dropout on nets, early stopping on everything. Always. Default regularisation in libraries is tuned for high-SNR; for finance you usually want more.

### 3. More data helps for *variance*; it doesn't move the *bias floor*.

If your signal has IC = 0.04, doubling the training set lowers your estimator's variance by ~30%, but the long-run mean IC of your strategy is still 0.04. Don't expect more data to suddenly reveal a hidden Sharpe 3.

### 4. Out-of-sample is harder than out-of-fold.

A model that scores well in your CV (even purged) can still degrade live. Reasons:

- **Distribution shift** — the test period is more "different" from training than CV implies.
- **Selection bias** — you picked the model that scored best on CV.
- **Subtle leakage** — features built from things that wouldn't have been available in real time.

Plan for a 30%+ degradation between CV score and live. That's typical.

### 5. Calibration matters as much as ranking.

In a classifier where you act on the probability (e.g. position size proportional to the model's confidence), the **calibrated probability** is the actionable output. A model that perfectly ranks but produces uncalibrated probabilities will mis-size positions. Chapter 4 covers calibration.

## The mental model

Picture two extreme cases:

| | High SNR (vision) | Low SNR (finance) |
|---|---|---|
| Loss surface | clear global minimum, deep valley | gentle bowl with random noise on top |
| Adding parameters | finds finer structure, helps | fits noise, hurts |
| Adding data | helps until plateau, then helps slowly | helps slowly, never plateaus visibly |
| Cross-validation score | close to test score | systematically optimistic |
| Best practice | go bigger | go simpler, regularise harder |

Internalise the second column and most of your ML choices fall out naturally.

## What this means for model selection

A reasonable default stack for daily return prediction:

1. **Features**: 30–80 hand-crafted, regularised. Not 10,000.
2. **Model**: XGBoost or LightGBM with `max_depth=4`, `subsample=0.8`, `colsample=0.5`, `min_child_weight=10`, `n_estimators=200`, early stopping on a validation fold.
3. **Cross-validation**: PurgedKFold with purge = label horizon (Module 9 ch4).
4. **Hyperparameter search**: Optuna with TPE, ≤ 100 trials.
5. **Calibration**: isotonic regression on the validation fold (chapter 4).
6. **Conformal intervals**: MAPIE on the CV folds (chapter 5).
7. **Sizing**: position scaled by calibrated probability, with hard caps.

That's the working recipe. Module 11 (Modern ML) implements it end-to-end.

## A specific anti-pattern: deep nets for daily returns

I've watched smart people spend months trying to make a deep net beat XGBoost on daily return prediction. They do not succeed. The dataset sizes (a few thousand samples) and the signal levels (IC ~0.03) just don't justify the architecture's expressive power.

Where deep nets *do* win in finance:

- **Limit-order-book prediction** (millions of samples, microstructure structure).
- **Alternative data** (text, images, audio — high-SNR raw modalities).
- **Multi-task learning** across many symbols (a transformer trained to predict all S&P 500 returns can learn shared representations).
- **End-to-end execution policies** via RL (Module 13).

For "predict tomorrow's SPY return from yesterday's features", gradient boosting is the right tool. Use what works.

## The honest bench

When you start a new ML-for-trading project, the first thing to beat is **the dumb baseline**:

- A constant prediction (always predict the unconditional mean).
- Linear regression on the same features.
- An XGBoost with default hyperparameters.

If your fancy model isn't beating linear regression by a meaningful margin out-of-sample, simplify.

## Pitfalls

!!! warning "Treating finance like a benchmark"
    On ImageNet, doubling parameters → 1% accuracy gain. On daily returns, doubling parameters → 0% IC gain plus larger variance. The diminishing-returns curves are completely different.

!!! warning "Trusting CV score"
    Even with perfect CV hygiene, the live score will be lower. Plan for it; size for it; don't claim Sharpe 1.5 from a Sharpe-1.5 CV.

!!! warning "Hyperparameter selection on the holdout"
    Even more catastrophic than usual in low-SNR. Two parameter values that differ by `(epsilon)` on the holdout might be statistically identical — picking one over the other is selection bias.

!!! warning "Ensembling without diversity"
    Ten gradient-boosted trees with different seeds give you almost no diversity (they're all fitting the same gradients). Ensemble across *model families* — XGBoost + Random Forest + Linear — for variance reduction that's real.

## Bottom line

For ML on financial data:

- **Default to simple models** (linear, GBM). Move to deep only with a clear reason.
- **Regularise hard**; the default settings are usually too permissive.
- **Plan for 30%+ CV-to-live degradation**.
- **Calibrate** your model's outputs; don't size on raw scores.
- **Conformal-bound** your predictions; never act on a point estimate alone.

Continue to **[Feature engineering for markets](02-feature-engineering.md)**.
