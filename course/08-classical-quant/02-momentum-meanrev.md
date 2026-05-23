# Momentum and mean reversion

The two oldest signals in markets, and the two that refuse to die. Momentum says winners keep winning over the medium term (3–12 months). Mean reversion says short-term moves get partly undone (1–5 days). Both have been documented for over 50 years, both have economic stories behind them, and both still earn risk-adjusted returns net of reasonable costs at retail scale.

This chapter implements both and discusses what makes them work and where they break.

## Cross-sectional momentum

Sort assets by their trailing return. Long the top, short the bottom. Re-balance monthly.

The classical Jegadeesh-Titman (1993) signal: rank by past 12-month return *skipping the most recent month* (to avoid microstructure-driven short-term reversal).

```python
import numpy as np
import pandas as pd

def xsec_momentum_signal(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    returns: T x N daily simple returns.
    Returns a T x N DataFrame of centered ranks in [-0.5, +0.5].
    """
    cum = (1 + returns).rolling(lookback - skip).apply(np.prod, raw=True).shift(skip) - 1
    ranks = cum.rank(axis=1, pct=True) - 0.5            # centred on zero
    return ranks
```

To turn the score into positions:

```python
def positions_from_score(score: pd.DataFrame, vol_target: float = 0.10) -> pd.DataFrame:
    """Convert a centred-rank score to dollar-neutral positions targeting vol."""
    # Daily realized vol per name
    vol = returns.rolling(60).std() * np.sqrt(252)
    # Inverse-vol weight within each side
    raw = score / vol.replace(0, np.nan)
    raw = raw.sub(raw.mean(axis=1), axis=0)             # dollar-neutral per day
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(gross, axis=0)                       # gross-exposure-normalised
```

That's a working cross-sectional momentum strategy in ~10 lines. The economic story: investor under-reaction to news, slow diffusion of information across investor segments. The risk: **momentum crashes** — sudden, sharp reversals after market bottoms (most famously in March 2009).

## Time-series momentum

Less subtle: be long if the asset is above its 12-month moving average, short if below. Applied across many assets and asset classes (equities, bonds, FX, commodities) and combined, it has paid astonishingly well — see Moskowitz, Ooi, Pedersen (2012).

```python
def ts_momentum_signal(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """+1 if price > 12-month MA, -1 otherwise. Scaled to inverse-vol."""
    above_ma = prices > prices.rolling(lookback).mean()
    signal = above_ma.astype(int) * 2 - 1
    vol = prices.pct_change().rolling(60).std() * np.sqrt(252)
    return signal / vol.replace(0, np.nan)
```

For a diversified time-series momentum portfolio across asset classes, this single-rule strategy has historically delivered ~0.8 Sharpe net of costs. Not a money printer; not a curiosity either.

## Short-term mean reversion

The opposite signal at a much shorter horizon. Lo and MacKinlay (1990) showed equities exhibit short-term reversal at weekly frequency. The cleanest expression: rank by past 5-day return, **short** the top, **long** the bottom.

```python
def xsec_reversal_signal(returns: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Short the past winners, long the past losers."""
    cum = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    return -(cum.rank(axis=1, pct=True) - 0.5)          # NOTE the negation
```

This works because of liquidity-provision dynamics — names that moved a lot recently are typically expensive to trade *into* (rich) and cheap to trade *against* (you're providing liquidity). The catch: it requires aggressive trading and crumbles under realistic transaction costs at retail scale. **Implementation costs are the entire game** for this strategy.

## Combining momentum and reversal

The signals point in opposite directions but at different horizons. The combination — long medium-term winners *minus* short-term winners — is the **residual momentum** trade:

```python
medium = xsec_momentum_signal(returns, lookback=252, skip=21)
short = -xsec_reversal_signal(returns, lookback=5)   # un-negate to get pure rank
combined = medium - 0.3 * short                       # subtract a fraction of short-term momentum
```

The intuition: an asset that has rallied over 12 months AND not just popped is on a stable trend. An asset that has rallied 12 months but just popped is likely to give some back.

## Pairs trading — a teaser

A close cousin of mean reversion: the spread between two cointegrated assets is *itself* a mean-reverting series. Trade it the same way you'd trade a single-asset mean reversion, but the spread is constructed from two raw assets with a time-varying hedge ratio.

We dedicate chapter 4 of this module to a full pairs implementation using a Kalman filter for the hedge ratio.

## A worked example: a simple TSM portfolio

```python
import numpy as np
import pandas as pd
from engine.data import YFinanceFeed

feed = YFinanceFeed()
universe = ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "VNQ", "EFA"]

prices = pd.concat(
    {sym: feed.bars(sym, "2014-01-01", "2024-12-31")["close"] for sym in universe},
    axis=1,
).dropna()

returns = prices.pct_change().dropna()
signal = ts_momentum_signal(prices, lookback=252).shift(1).reindex(returns.index)  # signal lagged

# Equal-vol positions; vol-target 10% annual
positions = signal.div(signal.abs().sum(axis=1), axis=0).fillna(0)
pnl = (positions * returns).sum(axis=1)
ann_sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
print(f"TSM Sharpe (gross): {ann_sharpe:.2f}")
```

With ~10 years of post-Lehman data, expect Sharpe ~0.6-0.8 net of reasonable costs. Add bond carry, FX carry, commodity carry, and you have a diversified systematic macro program. AQR runs versions of this; so do Man AHL, Winton, and twenty other quant CTAs.

## Why these signals persist

For momentum:
- **Under-reaction to news** — investors take time to update.
- **Confirmation bias** — winners get bought because they were winners.
- **Index inclusion** — flows from passive funds reinforce existing leaders.

For reversal:
- **Liquidity provision premium** — patient traders earn a spread for absorbing flow.
- **Microstructure noise** — bouncing between bid and ask creates apparent moves that revert.

Both have economic underpinnings, both have been documented out-of-sample for decades. That said:

!!! warning "Costs are not optional"
    Short-term reversal in particular **does not work at retail commissions and spreads on small accounts**. For monthly rebalancing of liquid ETFs, momentum often works at retail. For tick-level reversal in individual stocks, you need institutional execution.

## Pitfalls

!!! warning "Forecasting momentum vs riding momentum"
    Predicting *when* momentum will work is much harder than just running the strategy through. Don't try to time it — the regime-switching variants generally underperform the boring vanilla.

!!! warning "Momentum crashes"
    After major market bottoms (March 2009, March 2020), prior losers (high-beta stocks that fell most) lead the rebound. Momentum strategies, which were short these names, lose 20%+ in a few weeks. Risk management is *the* edge here — vol-targeting and drawdown stops.

!!! warning "Look-ahead in lookback construction"
    `rolling(252).mean()` includes today. For a signal, you want yesterday's MA. `.shift(1)` before using.

## Bottom line

Two strategies you can build today:

- **TSM across ETFs**, monthly rebalanced, vol-targeted at 10%. ~0.7 Sharpe gross historically.
- **Cross-sectional momentum on liquid stocks**, monthly rebalanced, dollar-neutral, vol-capped. ~0.6 Sharpe net.

Don't reinvent these. Get them working, then layer your alpha *on top* — Module 16 shows you how with an HMM regime overlay and a meta-labeling classifier.

Continue to **[Ornstein-Uhlenbeck and stationary spreads](03-ornstein-uhlenbeck.md)**.
