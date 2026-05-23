# Drawdown discipline

A 20% drawdown requires a 25% gain to recover. A 50% drawdown requires 100%. A 75% drawdown requires 300%. These aren't intellectually challenging facts but in practice they're underweighted: people accept big drawdowns when their model didn't predict them, then discover the recovery math is brutal.

This chapter is the working framework for drawdown management — measurement, prevention, recovery.

## The math

For an equity curve $E_t$ with running peak $P_t = \max_{s \leq t} E_s$:

$$
\text{Drawdown}_t = \frac{E_t}{P_t} - 1 \in (-1, 0]
$$

The **maximum drawdown** (over a period) is the most extreme value of $\text{Drawdown}_t$ — always negative or zero.

```python
import numpy as np
import pandas as pd


def max_drawdown(pnl: pd.Series) -> dict:
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    end = dd.idxmin()
    start = peak.loc[:end].idxmax()
    # Recovery time: first index after `end` where equity returns to `peak.loc[end]`
    target_equity = peak.loc[end]
    after = equity.loc[end:]
    recovery_mask = after >= target_equity
    recovery = recovery_mask.idxmax() if recovery_mask.any() else None
    return {
        "max_dd": float(dd.min()),
        "start": start, "end": end,
        "recovery": recovery,
        "duration_to_trough_days": (end - start).days,
        "duration_to_recovery_days": (recovery - start).days if recovery else None,
    }
```

## The required-gain table

| Drawdown | Required gain to recover |
|---|---|
| -5% | +5.3% |
| -10% | +11.1% |
| -20% | +25.0% |
| -30% | +42.9% |
| -40% | +66.7% |
| -50% | +100.0% |
| -75% | +300.0% |
| -90% | +900.0% |

The pattern is non-linear: small drawdowns are cheap to recover from, big ones are catastrophic. The behavioural implication: **prevent the small drawdown becoming big**. Don't double down to "make it back."

## Drawdown-based risk overlays

The Module 16 strategies all include a drawdown overlay. The pattern: scale position size down as drawdown approaches your tolerance.

```python
def drawdown_scaler(pnl: pd.Series, max_dd: float = 0.20, exponent: float = 2.0) -> pd.Series:
    """Smooth scaler in [0, 1]. = 1 at no drawdown; → 0 as DD → max_dd."""
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    # Factor decays smoothly from 1 to 0 as DD approaches max_dd
    factor = (1 - (-dd / max_dd).clip(0, 1)) ** exponent
    return factor.shift(1).fillna(1.0)
```

The squared exponent makes the de-risking proportional to drawdown's square — gentle in normal regimes, aggressive as the cliff approaches. Higher exponent = more aggressive.

## Investor psychology

Drawdowns don't just hurt returns — they hurt your investor relationships and your own decision quality. After a 20% drawdown:

- **Investors redeem.** They may have committed not to, but most do anyway.
- **You stop trusting your signal.** Even if it's still valid, the temptation is to tinker.
- **Risk-taking ability drops.** Your psychological capital is depleted; bold trades become impossible.

The right defence is **avoiding the drawdown in the first place**. The drawdown overlay above is the structural way to do that.

## A taxonomy of drawdowns

- **Realised** — current peak-to-trough.
- **Unrealised** — peak-to-current, where the equity may still be recovering.
- **Time-bounded** — max DD within a specific period (e.g., calendar year).
- **Conditional on regime** — DD only in high-vol regimes.

Report all of these in the strategy tearsheet. The "max DD over the backtest" headline number is the least informative.

```python
def comprehensive_drawdown_report(pnl: pd.Series) -> dict:
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd_series = equity / peak - 1
    # Top 5 drawdowns
    in_dd = dd_series < 0
    dd_episodes = []
    start = None
    for ts, in_d in in_dd.items():
        if in_d and start is None:
            start = ts
        elif not in_d and start is not None:
            sub = dd_series.loc[start:ts]
            dd_episodes.append({"start": start, "end": sub.idxmin(), "recovery": ts,
                                 "depth": float(sub.min()),
                                 "duration_days": (ts - start).days})
            start = None
    dd_episodes.sort(key=lambda d: d["depth"])
    return {
        "max_dd": float(dd_series.min()),
        "top_5_episodes": dd_episodes[:5],
        "dd_percent_of_time": float(in_dd.mean()),
        "avg_dd_when_in": float(dd_series[dd_series < 0].mean()),
    }
```

## Drawdown sensitivity by strategy type

Some strategies have systematic drawdowns built in:

- **Momentum** — momentum crashes (Module 8 chapter 2). Concentrate around regime shifts.
- **Carry** — disorderly de-leveraging (Module 16 chapter 4). Concentrate around vol spikes.
- **Mean reversion** — when a "reversion" doesn't revert because the spread broke. Concentrate around credit / earnings events.
- **Vol selling** — sudden vol expansions (2018 XIV blow-up). Concentrate around exogenous shocks.

Each strategy's drawdown distribution looks different. The risk overlay should be tuned to that strategy's specific failure mode.

## Pitfalls

!!! warning "Drawdown is path-dependent"
    Two strategies with the same end-Sharpe and same max-DD can have very different drawdown *paths*. One can spend 80% of its time in drawdown; the other only 20%. Investors prefer the latter.

!!! warning "Backtest drawdowns are optimistic"
    Walk-forward backtests average across many regimes. Live trading happens in *one* regime — possibly worse than any in the backtest. Plan for live drawdowns to be 1.5-2x the worst-in-backtest.

!!! warning "Drawdown-conditioned strategy modification"
    Changing strategy parameters during a drawdown is almost always wrong. The drawdown happened because of regime, not because the parameters are bad. Wait, then re-evaluate calmly.

!!! warning "The "recovery" math is brutal"
    A strategy that draws down 40% needs +67% to recover. That requires either better-than-expected performance or a much longer time horizon. Don't assume mean-reversion of equity.

## Bottom line

For drawdown discipline:

- **Measure** — top-5 episodes, time-in-drawdown, recovery durations.
- **Limit** — explicit hard caps via the drawdown overlay scaler.
- **Diversify** — combine strategies with uncorrelated drawdown profiles.
- **Communicate** — be transparent with investors about expected DD distribution before they invest.

A strategy with a 0.8 Sharpe and a -20% max DD is worth more than a 1.0 Sharpe with -40% max DD, almost always.

Continue to **[VaR and CVaR — honestly](02-var-cvar.md)**.
