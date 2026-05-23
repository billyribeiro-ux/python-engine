# Slippage and the cost of trading

A backtest that doesn't include slippage is a backtest of a strategy you can't trade. Even modest slippage assumptions turn marginally-profitable backtests into clear losers. This chapter walks the three models you'll use and how to pick numbers that aren't fantasy.

## Where costs come from

Three real components of trading cost:

1. **Commissions** — explicit. Whatever your broker charges per share / per trade.
2. **Bid-ask spread** — you cross half of it on every trade as a market order.
3. **Market impact** — your order moves the price against you. Scales with order size relative to liquidity.

A fourth, often-forgotten:

4. **Borrow cost** for shorts. For hard-to-borrow names, can be 5-20% annualised.

## The linear cost model

The simplest: charge a constant `c` bps per unit of turnover (turn = position change of 1 unit).

```python
cost_per_turn_bps = 2.0          # 0.5 bp commission + 1 bp half-spread + 0.5 bp impact

cost = cost_per_turn_bps * 1e-4 * turnover
```

When to use: liquid large-caps, daily or coarser frequency, modest order sizes. SPY is well-modelled at 1-2 bps round-trip. AAPL at 2-4 bps. Smaller-cap stocks easily 10+ bps.

## Square-root impact model

A better approximation for orders that meaningfully move the market. The standard empirical observation: impact scales as the **square root of (order size / average daily volume)**.

$$
\text{impact}(q) = \sigma \cdot \eta \cdot \sqrt{q / V}
$$

where $\sigma$ is daily volatility, $q$ is order size in shares, $V$ is average daily volume, and $\eta \approx 0.5$-$1.0$ is an empirical constant (the "Kyle's lambda" sigma).

```python
import numpy as np

def square_root_impact(qty: float, adv: float, daily_vol: float, eta: float = 0.6) -> float:
    """Impact as a fraction of price."""
    if qty <= 0 or adv <= 0:
        return 0.0
    return daily_vol * eta * np.sqrt(qty / adv)
```

Example: trading 5% of ADV in SPY (daily vol ~1%):

```python
square_root_impact(qty=5_000_000, adv=100_000_000, daily_vol=0.01)
# 0.01 * 0.6 * sqrt(0.05) = ~0.13% = 13 bps per side
```

Two-sided round-trip: ~26 bps. Far above the linear 1-2 bps. This is why size matters.

## Almgren-Chriss optimal execution

Almgren-Chriss (1999) is the framework for **scheduling** large orders to minimise the sum of expected impact and timing risk:

- Spread the order over time → less impact per child, but exposure to price drift (timing risk).
- Execute fast → high impact, low timing risk.

The optimal schedule is exponential — execute heavier at the start, taper toward the end. The model has closed-form solutions for risk-neutral and mean-variance preferences.

```python
import numpy as np

def almgren_chriss_schedule(X: float, T: int, sigma: float, eta: float, lambda_risk: float = 0.0):
    """Optimal trajectory for executing X shares over T discrete steps."""
    # Time-impact coefficient
    if lambda_risk == 0:
        return np.linspace(X, 0, T + 1)               # linear schedule: risk-neutral
    kappa = np.sqrt(lambda_risk * sigma ** 2 / eta)
    t = np.arange(T + 1)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
```

For a backtest, you typically don't simulate the full schedule per trade — you assume some realistic VWAP-style execution and charge that as the cost. Module 19 (Execution) has the full AC implementation.

## What slippage to use in a backtest?

For different strategy types:

| Strategy | Reasonable cost (round-trip) |
|---|---|
| Daily ETF strategies (SPY, QQQ, IWM) | 1-3 bps |
| Daily large-cap single stocks | 4-8 bps |
| Daily small-cap single stocks | 15-30 bps |
| Intraday large-cap | 2-4 bps (with VWAP execution) |
| Intraday small-cap | 30-50 bps (rapidly worse with size) |
| Options on liquid underlyings (SPY, AAPL) | 50-200 bps round-trip per leg |
| Options on illiquid singles | 500+ bps |

**Always start with the high end of these ranges**. If your strategy isn't profitable at conservative costs, no amount of execution skill will fix it.

## A worked example: cost-sensitivity sweep

```python
import pandas as pd
from engine.backtest import vectorized_backtest

results = {}
for cost_bps in [0, 1, 2, 5, 10, 20]:
    res = vectorized_backtest(returns, signal, cost_per_unit_turnover=cost_bps * 1e-4)
    results[cost_bps] = res.stats
print(pd.DataFrame(results).T[["sharpe", "ann_return", "turnover_ann"]])
```

The output usually looks like:

```
  cost_bps  sharpe  ann_return  turnover_ann
        0    1.5      0.18         4.2
        1    1.1      0.13         4.2
        2    0.7      0.09         4.2
        5    0.0      0.00         4.2
       10   -0.7     -0.07         4.2
       20   -1.4     -0.14         4.2
```

The cost at which Sharpe goes to zero is the **break-even cost**. If your real cost is at or above the break-even, you're trading noise.

## Reducing turnover

If costs are eating you, the answer is rarely "trade smarter execution"; it's **trade less**. Three patterns:

1. **Higher entry / exit thresholds** — for spread/threshold strategies (mean reversion, breakout), wider thresholds → fewer signals → less turnover.
2. **Less frequent rebalancing** — monthly instead of weekly, weekly instead of daily.
3. **Buffered position changes** — only act on signal changes above a threshold:

```python
def buffered(target_pos: pd.Series, buffer: float = 0.05) -> pd.Series:
    """Only update position when |target - current| > buffer."""
    cur = 0.0
    out = []
    for t in target_pos:
        if abs(t - cur) > buffer:
            cur = t
        out.append(cur)
    return pd.Series(out, index=target_pos.index)
```

Buffering can cut turnover by 50-70% with minimal loss of edge, because most position changes are tiny noise on top of the real signal.

## Borrow costs for shorts

For long-short equity strategies, borrow is a real cost. Easy-to-borrow large caps: 25-50 bps per year. Hard-to-borrow specials: 5-20% per year.

```python
def borrow_cost(position: pd.Series, borrow_rate_ann: float) -> pd.Series:
    """Daily charge on negative positions (shorts)."""
    short = position.clip(upper=0).abs()
    return -short * borrow_rate_ann / 252
```

For a strategy that's persistently short, this is a daily drip that can dominate the gross alpha. Always check.

## Pitfalls

!!! warning "Assuming a constant cost when liquidity varies"
    Costs on a "wide universe of mid-caps" strategy are dramatically different from costs on SPY. Use per-asset cost estimates if your strategy trades a wide universe.

!!! warning "Backing out cost from your broker statement"
    Your broker's commission is one component. Don't forget the spread and impact you implicitly paid.

!!! warning "Pricing on the close, executing at next-bar open"
    You priced the signal on `close[t]` but you can only execute on `open[t+1]`. The gap is implicit slippage — sometimes 10s of bps for volatile names overnight. The vectorised engine's shift handles this if your signal is generated *from* `close[t-1]` data only.

!!! warning "Tax effects"
    For US equities, short-term gains taxed at ordinary income, long-term at preferential rate. A strategy that turns over 200% per year is incurring short-term gains on every trade. For taxable accounts this matters; for IRAs and institutional accounts it doesn't.

## Bottom line

For honest backtests:

- Start with **3-5 bps round-trip** for liquid equity ETFs.
- Use **square-root impact** for any strategy trading > 1% of ADV.
- Model **borrow** explicitly for short positions.
- Run a **cost-sensitivity sweep** as part of every backtest — know your break-even.
- If the strategy doesn't survive realistic costs, **reduce turnover** before you "optimise execution."

## End of Module 9

You now have the discipline (Module 9) plus the engines (`engine.backtest`) to run honest backtests. The next module is **ML Foundations for trading** — feature engineering, labeling, and conformal prediction — the pieces that turn "I have a backtest" into "I have a model whose forecasts are honestly calibrated."

Continue to **[Module 10 — ML Foundations](../10-ml-foundations/index.md)**.
