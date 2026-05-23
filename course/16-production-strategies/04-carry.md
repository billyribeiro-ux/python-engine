# Vol-targeted carry across asset classes

Carry is the simplest macro factor and one of the most persistent. The idea: hold the high-yielding asset, fund with the low-yielding asset, harvest the spread. It works in FX (high-rate currencies vs low), in rates (long bond futures with positive carry), in commodities (curve roll-down), and even in equities (high-dividend-yield baskets).

Carry's risk profile is asymmetric: it slowly compounds in good times and gives back a few years' worth in days during crises. Vol-targeting is what makes it tradeable.

## What "carry" means in each asset class

**FX carry**: long high-rate currency, short low-rate currency. Earn the interest-rate differential. Classical examples: long AUD/JPY (Aussie vs Yen) during the post-2000 yield differential.

**Rates carry**: long bond futures (e.g., long 10y Treasury futures). When the yield curve slopes up, you earn carry as the bond rolls down the curve.

**Commodity carry**: long futures contracts in backwardation (curve sloping down). As time passes, the spot rises toward the futures price; you earn the roll.

**Equity carry**: long high-yield/value stocks. Income from dividends + the value premium.

## A unified framing

For any underlying with a known carry rate $c$ and volatility $\sigma$:

$$
\text{Position} \propto \frac{c}{\sigma^2}
$$

That's the Sharpe-optimal sizing (Module 8 chapter 6 — Kelly-style). The numerator is the expected return; the denominator is the risk you take to earn it.

In code:

```python
import pandas as pd
import numpy as np


def carry_position(carry_rate: pd.Series, returns: pd.Series,
                    target_vol: float = 0.10, vol_lookback: int = 60) -> pd.Series:
    """Vol-targeted carry position."""
    realised_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    # Sharpe-implied raw position (scaled by carry expectation)
    raw_pos = (carry_rate / realised_vol ** 2).clip(lower=-2, upper=2)
    # Vol target by scaling
    expected_vol = realised_vol * raw_pos.abs()
    scale = (target_vol / expected_vol.replace(0, np.nan)).clip(upper=3.0)
    return (raw_pos * scale).shift(1).fillna(0.0)
```

## A worked FX carry example

A simple G10 FX carry: long high-rate currencies, short low-rate currencies, weighted by the carry rate.

```python
# Pretend we have monthly carry rates and daily returns
# carry_rates: T x N DataFrame of (annualised carry %) per currency pair
# returns: T x N DataFrame of daily returns

def fx_carry_portfolio(returns: pd.DataFrame, carry_rates: pd.DataFrame,
                        target_vol: float = 0.08) -> pd.DataFrame:
    """Cross-sectional FX carry: rank by carry, long top quintile, short bottom."""
    # Daily carry → use the most recent monthly rate, ffilled
    carry_rates_daily = carry_rates.reindex(returns.index, method="ffill")
    # Rank cross-sectionally
    ranks = carry_rates_daily.rank(axis=1, pct=True) - 0.5
    # Vol-scale per pair
    vol = returns.rolling(60).std() * np.sqrt(252)
    raw = ranks / vol
    # Normalise gross exposure to 1
    gross = raw.abs().sum(axis=1)
    weights = raw.div(gross.replace(0, np.nan), axis=0)
    # Target overall portfolio vol
    pnl_naive = (weights.shift(1) * returns).sum(axis=1)
    realised = pnl_naive.rolling(60).std() * np.sqrt(252)
    scale = (target_vol / realised.replace(0, np.nan)).clip(upper=3.0).shift(1).fillna(1.0)
    return weights.multiply(scale, axis=0)
```

For G10 FX, this strategy has historically delivered Sharpe ~0.5-0.7 — modest but with negative correlation to most equity strategies, making it valuable in a portfolio.

## The crash risk

Carry's enemy is **disorderly de-leveraging**. When global financial conditions tighten, leveraged carry positions get unwound simultaneously, the high-yielders collapse, the low-yielders rally, and carry P&L plummets.

The 2008 AUD/JPY carry trade: years of slow gains, then a 30% drop in October-November 2008.

Mitigations:

- **Vol-target aggressively** — when realised vol spikes, your position shrinks automatically.
- **VIX overlay** — zero the carry position when VIX > some threshold (e.g., 30).
- **Credit-spread overlay** — same idea using HY credit spreads as the risk-off indicator.
- **Cap individual currency exposure** — no more than 20% in any one pair.

```python
def crash_overlay(positions: pd.Series, vix_series: pd.Series,
                   risk_off_threshold: float = 30.0) -> pd.Series:
    """Zero positions when VIX exceeds threshold."""
    vix_aligned = vix_series.reindex(positions.index, method="ffill")
    mask = (vix_aligned > risk_off_threshold)
    return positions.where(~mask.shift(1).fillna(False), 0)
```

## A worked rates carry example

For US Treasury futures:

```python
from engine.data import YFinanceFeed, ParquetCache

feed = ParquetCache(YFinanceFeed(), root="data/bars")
tlt = feed.bars("TLT", "2014-01-01", "2024-12-31")["close"]  # 20+ Year Treasury ETF
ief = feed.bars("IEF", "2014-01-01", "2024-12-31")["close"]  # 7-10 Year Treasury ETF

# Carry proxy: TLT yield - IEF yield (in a real backtest, get actual yields)
# Position: long TLT when yield curve slopes up (high carry)
# This is a simplification; real rates carry uses the curve and roll-down explicitly.
```

For production rates carry, use the proper futures curve data and compute the roll explicitly.

## Why carry persists

Three economic stories:

1. **Risk premium for liquidity provision** — carry traders provide liquidity to currency markets; they earn for that service.
2. **Insurance premium for crash risk** — carry is bid down (yields don't equalise) because investors fear the crash. The premium is the insurance compensation for those willing to bear it.
3. **Behavioural anchoring** — investors anchor on home-country interest rates; foreign yields aren't fully arbitraged.

All three are economically reasonable. None are about predicting that high-yield currency moves; the carry is the *cash flow*, not the price.

## Combining carry with momentum

Carry has long stretches of mediocre returns. Momentum adds responsiveness to price moves. A simple combination:

```python
def carry_plus_mom(carry_rates, prices, returns, mom_lookback=252):
    carry_signal = carry_rates.rank(axis=1, pct=True) - 0.5
    mom_signal = (prices / prices.shift(mom_lookback) - 1).rank(axis=1, pct=True) - 0.5
    combined = 0.6 * carry_signal + 0.4 * mom_signal
    return combined
```

Backtested combinations of carry + momentum + value at the asset-class level (AQR's "Style Premia" papers) deliver Sharpe ~1.0 with much lower drawdowns than carry alone.

## Pitfalls

!!! warning "Carry rates from EOD data are stale"
    For FX, the overnight swap rate fluctuates intraday. The carry you actually earn is overnight, not "the last month's rate." Use proper interbank rate data for accurate backtests.

!!! warning "Currency hedged returns"
    If you're a USD investor trading non-USD assets, your P&L includes the FX move on the cash legs. Carry strategies that look attractive in local currency may not be after FX hedging cost.

!!! warning "Funding cost"
    Real carry requires borrowing in the low-yield currency. The cost of that borrowing (broker margin, FX swap roll) eats some of the carry. Model explicitly.

!!! warning "Tax inefficiency**
    For US investors, FX gains are ordinary income. A 0.5 Sharpe FX strategy delivers ~0.35 Sharpe after-tax. Compare apples to apples.

## Bottom line

Vol-targeted carry strategies:

- **Long high-yield, short low-yield**, vol-targeted at 8-10%.
- **Crash overlay** via VIX, credit spreads, or similar risk-off proxy.
- **Cap individual position exposure** at 20-30% of total gross.
- **Combine with momentum** for higher Sharpe and lower drawdowns.

Expected Sharpe: 0.5-0.8 in production; combined with mom/value, 0.8-1.2 portfolios.

Continue to **[Post-earnings drift with bias controls](05-pead.md)**.
