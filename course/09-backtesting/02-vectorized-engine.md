# A vectorised backtester in 30 lines

For most strategies you'll backtest in this course — daily-or-coarser, single asset, no broker-specific constraints — the right engine is a one-page vectorised one. Anything more is over-engineering. This chapter walks through `engine.backtest.vectorized_backtest`.

## The whole engine

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True, slots=True)
class BacktestResult:
    pnl: pd.Series
    equity: pd.Series
    turnover: pd.Series
    positions: pd.Series

    @property
    def stats(self) -> dict:
        r = self.pnl.dropna()
        ann = 252
        ann_ret = r.mean() * ann
        ann_vol = r.std(ddof=1) * np.sqrt(ann)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        equity = self.equity
        peak = equity.cummax()
        max_dd = (equity / peak - 1.0).min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")
        return {
            "sharpe": float(sharpe),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "turnover_ann": float(self.turnover.sum() * ann / len(r)),
        }


def vectorized_backtest(returns, positions, cost_per_unit_turnover=0.0, shift_signal=True):
    pos_aligned = positions.reindex(returns.index).fillna(0.0)
    pos_traded = pos_aligned.shift(1).fillna(0.0) if shift_signal else pos_aligned
    turnover = pos_traded.diff().abs().fillna(pos_traded.abs())
    gross = pos_traded * returns.fillna(0.0)
    cost = cost_per_unit_turnover * turnover
    pnl = gross - cost
    equity = (1.0 + pnl).cumprod()
    return BacktestResult(pnl=pnl, equity=equity, turnover=turnover, positions=pos_traded)
```

That's it. Forty lines (with the helper class), correctness baked in:

- **`shift_signal=True` by default** — today's position acts on tomorrow's return. Setting it `False` is for diagnostic comparisons only.
- **Turnover is `|Δposition|`** — both opening and closing a position cost.
- **Cost is `turnover × per_unit_cost`** — model linear costs as `2e-4` for 2 bps per turn, or higher for less liquid names.

## Using it

```python
from engine.data import YFinanceFeed, ParquetCache
from engine.backtest import vectorized_backtest
import numpy as np

feed = ParquetCache(YFinanceFeed(), root="data/bars")
spy = feed.bars("SPY", "2014-01-01", "2024-12-31")["close"]
returns = spy.pct_change().dropna()

# Simple long-when-above-200d-MA signal
ma200 = spy.rolling(200).mean()
signal = (spy > ma200).astype(int)         # 0 or 1

res = vectorized_backtest(
    returns=returns,
    positions=signal,
    cost_per_unit_turnover=2e-4,           # 2 bps round-trip equivalent
)
print(res.stats)
# {'sharpe': 0.58, 'ann_return': 0.103, ..., 'turnover_ann': 0.42}
```

Three things to notice:

1. We passed the **raw signal** as `positions`. The engine shifted it.
2. The result includes turnover so you can see how active the strategy is.
3. The stats are computed lazily as a property — call `.stats` only when needed.

## Sweeping parameters

The vectorised engine is fast enough that exhaustive sweeps are cheap:

```python
import pandas as pd

rows = []
for window in range(50, 251, 10):
    ma = spy.rolling(window).mean()
    sig = (spy > ma).astype(int)
    res = vectorized_backtest(returns, sig, cost_per_unit_turnover=2e-4)
    s = res.stats
    rows.append({"window": window, **s})
sweep = pd.DataFrame(rows).set_index("window")
print(sweep[["sharpe", "max_drawdown", "turnover_ann"]])
```

20 backtests of 10 years of SPY in well under a second. **Beware**: you've just done multiple testing. The best window's Sharpe is upward-biased. Chapter 5 explains the Deflated Sharpe correction.

## What the engine is not

- **Not for multi-asset portfolios.** You can run it per-asset and aggregate, but for a true portfolio backtest with rebalancing logic, write a small wrapper.
- **Not for options.** Options have non-linear payoffs in the underlying; the linear "position × return" doesn't apply. Module 14 has an options backtester.
- **Not for event-driven logic.** Limit orders, stop-losses, takeovers, partial fills — these need the event-driven engine (next chapter).

For 80% of the strategies in the rest of the course, vectorised is what you'll use.

## A multi-asset extension

Five lines on top of the engine:

```python
def vectorized_portfolio_backtest(returns_df, positions_df, cost=2e-4):
    """Per-asset backtest, then equal-weight aggregate."""
    results = {
        sym: vectorized_backtest(returns_df[sym], positions_df[sym], cost)
        for sym in returns_df.columns
    }
    pnl = pd.DataFrame({sym: r.pnl for sym, r in results.items()})
    portfolio_pnl = pnl.mean(axis=1)
    return pnl, portfolio_pnl, (1 + portfolio_pnl).cumprod()
```

For a real multi-asset engine with covariance-aware rebalancing, use [`bt`](https://pmorissette.github.io/bt/) or [`vectorbt`](https://vectorbt.dev/) — both production-grade. The hand-rolled version above is plenty for course examples.

## Common extension: drawdown stops

```python
def with_drawdown_stop(positions, returns, max_dd=0.20, recovery_factor=2.0):
    """Scale positions down when drawdown approaches max_dd."""
    # First, a vanilla backtest to compute live equity
    res = vectorized_backtest(returns, positions)
    peak = res.equity.cummax()
    dd = res.equity / peak - 1
    # Multiplier in [0, 1] that decays as we approach max_dd
    factor = (1 - (-dd / max_dd).clip(0, 1)) ** recovery_factor
    return positions * factor.shift(1).fillna(1.0)
```

Plug back into `vectorized_backtest` for the final P&L. The pattern generalises — any path-dependent overlay (regime overlay, vol-targeting, drawdown stops) can be layered as a positions transformation.

## Reporting

For pretty equity curves and tearsheets, the standard library is [`quantstats`](https://github.com/ranaroussi/quantstats):

```python
import quantstats as qs
qs.reports.html(res.pnl, output="report.html")
```

You get a full HTML tearsheet — Sharpe by year, drawdown heatmap, monthly returns, distribution plots. For research, run it after every backtest.

## Pitfalls

!!! warning "Forgetting to align positions to returns"
    If `positions` has a different index from `returns`, the engine reindexes and fills missing with zero. That can mask alignment bugs (e.g. monthly positions filled with zero for every non-rebalance day). Always check that `len(pos_traded.nonzero()) > 0` makes sense.

!!! warning "Costs that should depend on position size"
    Linear-cost is an approximation. For small-cap, low-volume names, costs scale faster (chapter 6). Don't backtest a small-cap strategy with the same 2 bps you'd use for SPY.

!!! warning "Compounding vs. constant-dollar P&L"
    `equity = (1+pnl).cumprod()` is compounding. If you want constant-dollar (e.g. "always trade $1M notional"), the math is `pnl.cumsum()`. The two have very different drawdown characteristics over long horizons.

## Bottom line

Forty lines. Honest by construction. Fast enough for sweeps. The right starting point for almost every strategy in this course.

Continue to **[An event-driven engine](03-event-driven.md)**.
