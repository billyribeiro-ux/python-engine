# Module 11 — Modern ML for Trading

Module 10 covered the foundations; this module is the production layer. Gradient-boosted trees tuned with Optuna, ensembling and stacking, SHAP for feature attribution (and its specific traps in time series), survival analysis for "time to stop-out", and causal inference for honest alpha attribution.

Pages:

1. **[XGBoost, LightGBM, CatBoost — when each wins](01-gbm.md)** — the three modern GBM libraries and a working hyperparameter search with Optuna.
2. **[Stacking, ensembling, meta-labeling](02-stacking.md)** — combining models without overfitting; meta-labeling done right.
3. **[SHAP for feature attribution](03-shap.md)** — what SHAP tells you, what it doesn't, and the specific traps in time series.
4. **[Survival analysis — time to stop-out](04-survival.md)** — Kaplan-Meier and Cox regression on trade outcomes.
5. **[Causal inference for alpha attribution](05-causal.md)** — DoWhy and EconML; treatment-effect estimation; the difference between correlation and "would have worked if I had done it".
6. **[Gaussian processes for vol surfaces and curves](06-gp.md)** — GP regression for sparse, irregular data with calibrated uncertainty.

Start with **[XGBoost, LightGBM, CatBoost — when each wins](01-gbm.md)**.
