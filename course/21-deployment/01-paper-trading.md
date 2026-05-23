# Paper trading first

Before a single real-money trade, you paper-trade. For at least 30 days. Ideally 60-90. Even when you're confident the strategy is correct. This chapter is why, and what to do during it.

## What paper trading catches that backtesting doesn't

A successful backtest validates: the signal logic, the cost model, the sizing logic. It does not validate:

- **Broker-side latency** — how long does an order actually take to fill?
- **Real intraday data quality** — are vendor quotes consistent with broker fills?
- **Pre-market / after-hours behaviour** — historical bars are often EOD-only.
- **Corporate actions handling** — does your code understand splits, dividends, mergers?
- **Holidays** — partial sessions, early closes.
- **Your own code bugs** — symbol mapping, decimal precision, timezone errors.
- **Operational edge cases** — what happens if your broker's API is down for 30 minutes?

Paper trading exposes all of these in a low-stakes environment.

## What "paper trading" actually means

Three flavours, in increasing realism:

1. **Synthetic paper trading**: your strategy emits orders to a fake broker that fills them at next-bar open. Cheap; tests the strategy logic end-to-end.
2. **Broker-paper account**: your strategy sends orders to a *real broker's paper-trading endpoint* (Alpaca, IBKR, Tradier all offer this). Tests the broker integration, latency, order types — without real money at risk.
3. **Live with tiny capital**: deploy to live trading with $1,000-10,000. Real fills, real costs, real consequences — but capped.

A typical schedule: 14 days synthetic → 30 days broker-paper → 14 days tiny-live → full deployment.

## The broker-paper trading loop

For Alpaca:

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(API_KEY, SECRET_KEY, paper=True)


def submit_order(symbol: str, qty: int, side: str):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order_data=order)


# Every strategy decision:
order = submit_order("SPY", 100, "BUY")
print(f"Submitted: {order.id}")
```

Alpaca's paper-trading endpoint mimics live fills with simulated latency and slippage. Tradier and IBKR offer similar.

## Capturing paper-trading P&L correctly

You need to track:

1. **Submitted orders** — what your algorithm intended.
2. **Filled orders** — what actually executed.
3. **Theoretical P&L** — what the backtest formula says.
4. **Realised P&L** — what the broker reports.

The gap between (3) and (4) is your **execution slippage**. If the gap is small (< few bps), your backtest cost model is honest. If it's large (10s of bps), you have a problem.

```python
def reconcile_pnl(theoretical_pnl: dict, broker_pnl: dict) -> dict:
    """Compare backtest-equivalent P&L vs what the broker reports."""
    diff = {}
    for date in theoretical_pnl:
        if date in broker_pnl:
            diff[date] = theoretical_pnl[date] - broker_pnl[date]
    return diff


def slippage_diagnosis(reconciliation: dict) -> dict:
    """Investigate persistent slippage."""
    daily_slippage = pd.Series(reconciliation).sort_index()
    return {
        "mean_slippage_bps": float(daily_slippage.mean() * 10000),
        "std_slippage_bps": float(daily_slippage.std() * 10000),
        "max_slippage_day": daily_slippage.idxmin().isoformat() if len(daily_slippage) else None,
    }
```

## What to look for during paper trading

After each session, check:

- **Order placement times**: are you sending orders at the times the strategy demands?
- **Fill prices vs target**: how close are paper fills to expected?
- **Number of orders per day**: matches the backtest's turnover?
- **Position consistency**: end-of-day positions match the strategy's expected state?
- **Any errors / warnings**: surfaced and logged?

A clean paper-trading week looks boring: no surprises, P&L close to backtest expectations.

## When to move on

Move from paper to live when:

1. **30+ days of clean paper trading** with no unexpected errors.
2. **Realised paper P&L is within ±20% of backtested P&L** (more is suspicious).
3. **Slippage looks like the backtest's cost model** (or you've updated the model).
4. **You've simulated at least one "off day"** — a holiday, an unusual market move — and the strategy handled it.

If any of these fail, paper longer.

## The trap of paper trading

The biggest risk: **paper-trading complacency**. Paper losses don't hurt; you may become less rigorous. Some tricks:

- **Treat paper results as if they were real.** Compute P&L daily; report it like you would investor returns; agonise over drawdowns.
- **Don't tinker.** If the strategy underperforms on paper, that's information. Don't change the strategy mid-paper unless there's a bug.
- **Document everything.** Bugs you find are knowledge for future strategies.

## Pitfalls

!!! warning "Paper broker behaviour diverges from live broker"
    Paper-trading endpoints sometimes have different latency, different fill realism, different rate limits. Plan for live to be slightly worse than paper.

!!! warning "Different vendor data feed in paper vs live"
    Some brokers serve different real-time data in paper vs live accounts. If your strategy relies on a specific feed quality, verify it's the same.

!!! warning "Position-state sync at startup"
    Your strategy thinks it has a position; the broker disagrees. On every startup, **reconcile** your position state from the broker before placing orders.

!!! warning "Skipping paper entirely"
    "I've backtested for 5 years; I'm sure." No. You haven't tested the broker integration. Always paper.

## Bottom line

For deployment:

- **Paper trade 30-90 days** before going live.
- **Treat paper results as real** — same rigor, same reporting.
- **Reconcile theoretical and broker P&L daily**.
- **Don't tinker** mid-paper unless there's a bug.

The cost: a month of delay. The benefit: catching bugs that would have cost real money.

Continue to **[Brokerage integration](02-brokerages.md)**.
