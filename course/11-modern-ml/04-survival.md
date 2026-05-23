# Survival analysis — time to stop-out

Most trading research models "what's the probability of a profitable trade?". Survival analysis lets you model **"how long until something happens?"** — time-to-stop-out, time-to-target, time-to-regime-change, time-to-default. The framework handles **censoring** (trades that are still open, regimes that haven't broken yet) correctly, which is what makes it different from a plain regression.

This chapter is a working tour. The standard library is [`lifelines`](https://lifelines.readthedocs.io).

## The basic concepts

For each observation:

- **Time** $T$ — how long until the event.
- **Event indicator** $E$ — did the event happen ($E=1$), or was the observation censored ($E=0$, meaning we stopped watching before the event)?

A typical financial example: a trade entered, held until either profit-take (event = +1), stop-loss (event = -1), or vertical-barrier-time (event = 0, censored). Survival analysis cleanly handles all three.

## Kaplan-Meier — the non-parametric estimator

Estimates the **survival function** $S(t) = P(T > t)$ without assuming a distribution.

```python
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

durations = np.array([5, 10, 3, 12, 8, 15, 7])         # bars held
events = np.array([1, 1, 1, 0, 1, 0, 1])               # 1 = stopped out, 0 = still open

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events)
print(kmf.survival_function_.head())
# survival_function_[t] is the estimated probability the trade is still alive at time t
```

For trading, K-M tells you "given a trade is still open at bar 5, what's the probability it survives to bar 10?" Useful for time-aware stop-loss rules.

## Cox Proportional Hazards — features that change the hazard

Cox models the **hazard rate** $h(t | x) = h_0(t) \cdot \exp(\beta^\top x)$ — features multiplicatively scale a baseline hazard. Positive $\beta$ for feature $i$ means higher $x_i$ → faster failure.

```python
from lifelines import CoxPHFitter

df = pd.DataFrame({
    "duration": durations,
    "event": events,
    "vol": [0.01, 0.02, 0.015, 0.01, 0.025, 0.012, 0.02],
    "size": [1.0, 1.5, 0.8, 1.2, 1.0, 0.9, 1.3],
})

cph = CoxPHFitter().fit(df, duration_col="duration", event_col="event")
print(cph.summary)
```

The coefficient on `vol` tells you "for every unit increase in vol, the hazard of stop-out multiplies by exp(coef)." Highly intuitive output.

For trade-management: train Cox on past trades' (duration, stopped_or_not) outcomes with entry-time features. New trades get an expected time-to-stop forecast that adapts to their features. Use it to time-cap positions appropriately rather than using a uniform vertical barrier.

## A worked example: time-to-stop on a real strategy

```python
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from engine.features import triple_barrier_labels

# Suppose we have a primary signal and the triple-barrier output
events_idx = primary_signal[primary_signal != 0].index
side = primary_signal.loc[events_idx]

barriers = triple_barrier_labels(close, events=events_idx,
                                 pt_sl=(0.03, 0.02), horizon_bars=20, side=side)

# Build the survival dataset
df = pd.DataFrame({
    "duration": (barriers["t1"] - barriers.index).dt.days,
    "event": (barriers["label"] != 0).astype(int),       # 1 if hit a barrier, 0 if time-stopped
    "vol_20": vol_at_entry.reindex(events_idx),
    "trend": trend_at_entry.reindex(events_idx),
    "size_proxy": size_proxy_at_entry.reindex(events_idx),
}).dropna()

cph = CoxPHFitter(penalizer=0.01).fit(df, duration_col="duration", event_col="event")
print(cph.summary[["coef", "exp(coef)", "p"]])
```

If `vol_20` has a positive coefficient, high-vol entries hit a barrier faster — perhaps adjust your vertical barrier upward in low-vol regimes to capture more upside.

## Accelerated Failure Time models

An alternative to Cox: model **time itself** as a function of features, with parametric distributions (Weibull, log-normal, log-logistic, exponential):

```python
from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter().fit(df, duration_col="duration", event_col="event")
print(aft.summary)
```

AFT models say "feature X multiplicatively scales the timescale." Easier to interpret in some applications.

## Time-varying covariates

Real strategies have features that change *during* the trade — realised P&L, time decay, regime indicators. Cox handles this:

```python
from lifelines import CoxTimeVaryingFitter

# Long-format DataFrame: one row per (trade, time-interval) with feature snapshots
long_df = ...
ctv = CoxTimeVaryingFitter().fit(long_df, id_col="trade_id",
                                  event_col="event", start_col="start", stop_col="stop")
```

This lets the hazard adapt as the trade evolves. The model can say "trades held > 5 days with declining P&L are highly likely to stop out."

## Survival regression in trade management

Three concrete applications:

1. **Optimal time stops.** Estimate the survival function for an open trade. When it crosses some threshold (say, S(t) < 0.5 — 50% chance of being still alive), close manually.
2. **Position sizing decay.** Reduce position when expected remaining duration is short — the marginal P&L from holding longer is small.
3. **Strategy decision: trade or wait.** If the Cox model predicts very fast hazard (vol regime is extreme), maybe skip this entry.

## Calibration of survival models

Just like classifiers, survival models need calibration. The `lifelines` library provides `calibration_curve_for_survival` analogues. The simplest check: plot the model's predicted survival at 10 days against the empirical survival at 10 days on a held-out set.

## Pitfalls

!!! warning "Censoring is required"
    If your "trades" all hit a barrier (no censored observations), survival analysis adds little over a simple regression on duration. The framework's value comes from the censored cases.

!!! warning "The proportional hazards assumption"
    Cox assumes feature effects are constant over time. Test with `cph.check_assumptions(df)`. If violated, use AFT or time-stratify the Cox model.

!!! warning "Very few events"
    With fewer than ~50 events, Cox regression coefficients are noisy. Stick to non-parametric K-M for small samples.

!!! warning "Confusing survival and probability"
    "30% survival at day 10" means 30% are *still alive* at day 10 — i.e., 70% have been stopped out. Don't flip the interpretation.

## Bottom line

Survival analysis is **the right framework whenever you care about "time to X"** and you have right-censoring (open trades, ongoing regimes). For trade management it gives you principled time-stops, expected duration forecasts, and a way to reason about the trade-off between holding longer and stopping out.

For pure direction prediction, classification (Module 10) is still right. Survival sits alongside, not in place of.

Continue to **[Causal inference for alpha attribution](05-causal.md)**.
