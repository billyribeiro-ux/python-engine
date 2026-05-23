# Combinatorial purged CV and the Deflated Sharpe Ratio

Walk-forward gives you one out-of-sample path through history. Combinatorial purged CV (CPCV) gives you many — by leaving out *combinations* of folds rather than single folds. The reason this matters: from many paths you can compute the **probability of backtest overfitting** (PBO), which finally lets you ask "is this strategy's edge real or did I select for noise?"

This chapter pairs CPCV with the **Deflated Sharpe Ratio** (DSR) from Module 6. Together they're the strongest defence against selection bias in quant research.

## CPCV in one picture

With $K$ folds total and $N$ test folds left out per path:

- Number of train/test partitions: $\binom{K}{N}$.
- For $K=10$, $N=2$: 45 paths. Each path's out-of-sample is **two non-contiguous fold-sized slices**.
- The set of OOS predictions, across paths, covers each fold approximately $\binom{K-1}{N-1}$ times.

This gives you many independent estimates of out-of-sample performance, instead of one walk-forward path that you might have overfit to.

## A working CPCV implementation

```python
from itertools import combinations
import numpy as np

def cpcv_paths(n: int, k: int = 10, n_test: int = 2, purge: int = 0):
    """Yield (train_idx, test_idx) for every combinatorial purged path."""
    fold_size = n // k
    bounds = [(i * fold_size, (i + 1) * fold_size if i < k - 1 else n) for i in range(k)]
    all_idx = np.arange(n)
    for combo in combinations(range(k), n_test):
        test = np.concatenate([all_idx[bounds[i][0]:bounds[i][1]] for i in combo])
        keep = np.ones(n, dtype=bool)
        for i in combo:
            a, b = bounds[i]
            keep[max(0, a - purge):min(n, b + purge)] = False
        train = all_idx[keep]
        yield train, test
```

For each path, you train on the surviving training data and produce predictions for the test slices. After all paths, each timestamp has multiple OOS predictions; average them for the final OOS series.

## Probability of Backtest Overfitting (PBO)

Bailey, Borwein, López de Prado, Zhu (2014). For each path's resulting Sharpe (from `n_paths` strategies you tested), check whether the strategy that was *best in-sample on that path* was also *best out-of-sample*. If it usually isn't, you have PBO close to 0.5 — your "selection" is no better than chance.

```python
def pbo(in_sample_perf: np.ndarray, oos_perf: np.ndarray) -> float:
    """
    in_sample_perf: shape (n_strategies, n_paths) — IS Sharpe per strategy per path
    oos_perf:       shape (n_strategies, n_paths) — OOS Sharpe per strategy per path
    Returns the PBO in [0, 1]: probability that the IS-best is below-median OOS.
    """
    n_strats, n_paths = in_sample_perf.shape
    inverted = 0
    for p in range(n_paths):
        best_is = np.argmax(in_sample_perf[:, p])
        rank_oos = (oos_perf[:, p] >= oos_perf[best_is, p]).sum()
        # rank is in [1, n_strats]; "below median" = rank > n_strats / 2
        if rank_oos > n_strats / 2:
            inverted += 1
    return inverted / n_paths
```

PBO close to 0 → IS-best is also OOS-best most of the time → selection is meaningful.
PBO close to 0.5 → IS-best is random OOS → your selection is overfit.

For a sweep over many parameters or many strategies, **PBO > 0.4 is a red flag**. Below 0.2 is encouraging.

## The Deflated Sharpe Ratio (refresher)

From Module 6: the DSR adjusts an observed Sharpe for the number of trials and the variance of the trial Sharpes. It returns the **probability** that the observed maximum Sharpe is real.

```python
from engine.backtest import deflated_sharpe

trial_sharpes = [bt.stats["sharpe"] for bt in all_backtests]
champion = max(trial_sharpes)
dsr = deflated_sharpe(observed=champion, sharpes=trial_sharpes, T=len(returns))
print(f"DSR: {dsr:.2f}")
```

DSR > 0.95: very likely real edge.
DSR 0.7–0.95: probably real, verify with walk-forward.
DSR < 0.7: probably selection.

## A worked example: a parameter sweep with CPCV + DSR

```python
import numpy as np
import pandas as pd
from itertools import product
from engine.backtest import vectorized_backtest, deflated_sharpe

windows = [50, 100, 150, 200, 250]
exit_lookbacks = [5, 10, 20]
costs = 2e-4

# Build every (entry, exit) combination
trials = list(product(windows, exit_lookbacks))
sharpes = []
for entry, exit_lb in trials:
    sig = some_signal(prices, entry, exit_lb)
    res = vectorized_backtest(returns, sig, cost_per_unit_turnover=costs)
    sharpes.append(res.stats["sharpe"])

best_idx = int(np.argmax(sharpes))
best = sharpes[best_idx]
dsr = deflated_sharpe(observed=best, sharpes=sharpes, T=len(returns))
print(f"Best trial: {trials[best_idx]}, IS Sharpe={best:.2f}, DSR={dsr:.2f}")
```

If `DSR > 0.9`, the best trial is real. If `DSR < 0.5`, you've found noise. The single line of DSR computation is one of the highest-value calls in your whole research workflow.

## Combining the two — research hygiene

A complete research workflow:

1. **Walk-forward** the candidate strategy first. Look at the OOS Sharpe path. Stable? Decaying?
2. **Sweep hyperparameters** on the train-validation portion only (never the final test set).
3. **CPCV** the sweep to compute PBO. PBO < 0.3 → continue. > 0.4 → simplify the model and start over.
4. **DSR** the champion against the sweep set. > 0.9 → continue.
5. **Promote to the holdout** for a final read.
6. **Paper-trade** for 30+ days.
7. **Live**.

Most "great" backtests die between steps 3 and 4.

## A note on capacity

PBO and DSR are about whether the *signal* is real. Capacity is about whether you can *trade* it. Even a confirmed-real signal may not be tradeable at meaningful AUM. Test by:

1. Bootstrap-resample the trade list to estimate the distribution of P&L under random execution timing.
2. Apply realistic slippage that scales with size (chapter 6).
3. Find the AUM at which marginal Sharpe drops below your target (e.g. 0.5).

That AUM is your capacity. For most retail-readable signals it's $10–100M; for institutional-quality signals it can be $1B+.

## Pitfalls

!!! warning "PBO on small trial counts is noisy"
    With fewer than ~20 trials, the PBO estimate has high variance. Either sweep more parameters or interpret the number cautiously.

!!! warning "DSR on a single backtest"
    DSR requires the set of trials. If you ran only one strategy and got a Sharpe of 1.5, you can compute DSR with `sharpes=[1.5]` but the result will be near 1.0 because there's no spread. That's by construction — DSR's whole job is to penalise selection from a *set*.

!!! warning "Reusing CV folds for selection and evaluation"
    If you used your purged CV to pick hyperparameters, that same CV's "OOS score" is biased high. Always reserve a final holdout that the selection never touched.

!!! warning "CPCV is computationally heavy"
    $\binom{10}{2} = 45$ paths × model fit per path = 45 model fits. For deep models, this is slow. Use 6 folds × 2 test (15 paths) as a tractable compromise.

## Bottom line

- For sweeps and selection: **CPCV + PBO** check.
- For champion claims: **Deflated Sharpe Ratio**.
- For deployment readiness: **walk-forward** on a holdout that selection never touched.

The combination of these three takes you from "backtest result" to "honest read on expected live performance." That's the whole game.

Continue to **[Slippage and the cost of trading](06-slippage.md)**.
