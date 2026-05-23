# Causal inference for alpha attribution

A prediction model answers "given the world looked like this, what's the likely outcome?" A causal model answers "if I had done X instead of Y, what would have happened?" The two are different. For trading, knowing the *causal* effect of a decision is what lets you attribute alpha properly and predict the effect of policy changes.

This chapter introduces the working tools — `DoWhy`, `EconML` — and the key concepts every quant should know.

## Why correlation is not what you want

A prediction model finds patterns in the joint distribution of features and outcomes. It will happily exploit a feature that *predicts* the outcome without *causing* it — for example, executions are correlated with future prints because both are caused by an information arrival; a "predict-from-executions" model picks up that correlation and looks great.

In production, your strategy might *change* the executions pattern (because you're now trading on it), breaking the correlation. The model loses its edge.

A causal model would say: "executions cause prints only via the underlying information; my own executions don't cause future prints (because I'm small); the feature is non-causal." The causal model is more conservative — and more honest about what survives deployment.

## The fundamental problem

Causation requires a **counterfactual**: what would have happened if treatment $T$ had been different? You only ever observe one treatment per unit. So causation is fundamentally about *estimating an unseen world*.

The classical solution: **randomised controlled trials** (RCTs) — assign treatments at random so the treatment and outcome are independent conditional on covariates. In trading, you can occasionally run RCTs (randomised execution algorithm assignment, e.g.) but mostly your data is observational.

## Confounders and the back-door

Suppose you want to know "does my momentum signal cause higher returns?" Your data shows momentum-positive days had higher next-day returns. But maybe both are caused by an overall bull-market regime ($R$). Then:

- $T$ (momentum signal) and $Y$ (return) share a common cause $R$.
- The observational correlation $P(Y | T)$ overstates the causal effect $P(Y | \text{do}(T))$.

To estimate the causal effect, you need to **adjust for $R$** — block the back-door path from $T$ to $Y$ via $R$.

```python
import dowhy
from dowhy import CausalModel

# Hypothetical: data has treatment T (signal fired), outcome Y (next-day return),
# and a regime indicator R that confounds them.
model = CausalModel(
    data=df,
    treatment="T",
    outcome="Y",
    common_causes=["R"],            # the confounders to adjust for
)
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print(estimate)
```

The estimate adjusts for $R$ — telling you what proportion of the observed effect is actually causal.

## Treatment effect estimators with EconML

`EconML` (from Microsoft Research) gives you machine-learning-based causal estimators:

- **DoubleML** — uses ML models for both the treatment-prediction and outcome-prediction nuisance functions, then estimates the causal effect from residuals.
- **DR-Learner** — doubly-robust estimator; consistent if either nuisance model is correctly specified.
- **Causal Forests** — random-forest-based estimator of *heterogeneous* treatment effects (the effect varies by features).

```python
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor

dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    random_state=0,
)
dml.fit(Y=df["Y"], T=df["T"], X=df[["feature_1", "feature_2"]], W=df[confounders])
print(f"ATE estimate: {dml.const_marginal_effect(X=df[['feature_1', 'feature_2']]).mean():.4f}")
print(f"CI: {dml.const_marginal_effect_interval(X=df[['feature_1', 'feature_2']])}")
```

The output: an average treatment effect (ATE) plus a confidence interval. For trading: "the signal causes a +3 bp daily edge, with 95% CI [+1bp, +5bp]." That's what you want.

## Heterogeneous treatment effects (HTE)

The treatment effect can vary by features — the signal works better in some regimes, on some symbols, at some times. **Causal forests** estimate this:

```python
from econml.dml import CausalForestDML

cf = CausalForestDML(n_estimators=500, max_depth=5, random_state=0)
cf.fit(Y=df["Y"], T=df["T"], X=df[features], W=df[confounders])
hte = cf.const_marginal_effect(X=df[features])
df["estimated_effect"] = hte
```

Now `estimated_effect` tells you, per observation, "the causal effect of treatment T for an observation with these features." If you only act on the top quartile of estimated effects, you concentrate your trading on the bars where the signal genuinely helps.

For trading: use a causal forest on past signal/outcome data to estimate the *expected causal alpha per bar*. Trade only the bars in the top tercile of estimated effect. This usually outperforms a naive "trade every signal" strategy because it implicitly learns the conditions under which the signal works.

## Instrumental variables

When you have an unobserved confounder you can't adjust for, an **instrument** $Z$ — a variable that affects $T$ but only affects $Y$ through $T$ — gives you a back door.

Classical finance example: distance from the nearest IPO (instrument) affects IPO participation rates (treatment), and only affects returns through participation. The 2SLS estimator:

```python
from econml.iv.dml import DMLIV

dmliv = DMLIV(...)
dmliv.fit(Y=df["Y"], T=df["T"], Z=df["Z"], X=df[features])
```

Instruments are rare in finance — they have to satisfy the exclusion restriction. Don't fake one. But when you have a real instrument (a randomised broker quote, a regulatory event, a fund-flow shock), IV gives you causal estimates from observational data.

## A worked example: did my new feature actually cause Sharpe improvement?

You added a new feature to your signal, refit, deployed. Sharpe went up by 0.2. Question: was that the feature, or was the test period just kinder?

```python
import pandas as pd
import numpy as np
from econml.dml import LinearDML

# 'T' = 1 if the strategy version with the new feature was active that day
# 'Y' = daily strategy return
# 'X' = market features that could confound (vol, regime, etc.)
df = build_panel(...)

dml = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor())
dml.fit(Y=df["Y"], T=df["T"], X=df[market_features])
ate = float(dml.const_marginal_effect(X=df[market_features]).mean())
ci = dml.const_marginal_effect_interval(X=df[market_features])
print(f"Causal ATE of new-feature version: {ate*252:.2%} annualised")
print(f"95% CI: [{ci[0].mean()*252:.2%}, {ci[1].mean()*252:.2%}]")
```

If the CI for the annualised ATE crosses zero, you don't have evidence the feature caused the improvement. The Sharpe gain might just be the regime.

## Pitfalls

!!! warning "Causal inference without a clear graph"
    `DoWhy` and `EconML` give you machinery; you supply the assumptions (what's a confounder, what's an instrument). Garbage in, garbage out. Sketch the DAG explicitly.

!!! warning "Causal inference on observational time series"
    All the methods assume i.i.d. observations. For time series, observations are autocorrelated. Use block-bootstrap CIs and respect time ordering in any train/test split.

!!! warning "ATE is not the same as 'this is what will happen'"
    Causal estimates are about the *average* effect under the same conditions. If you scale up trading on a feature, you change the conditions — the ATE doesn't account for your own market impact.

!!! warning "Strong assumptions are unavoidable"
    Causal inference requires assumptions you can't fully verify from data. Be explicit about them. Sensitivity analyses (`DoWhy` ships them) help you understand robustness.

## Bottom line

Causal inference is the right framework when you need to answer **"would this have worked if I had done it?"** rather than **"what's correlated with the outcome?"** For attribution, A/B-test analysis, and "did my new feature actually help" questions, the causal answer is the correct one.

For pure forecasting (the main job of most quant models), prediction is fine. Causal inference is the layer that turns prediction into reliable decision-making.

Continue to **[Gaussian processes for vol surfaces and curves](06-gp.md)**.
