# Put-call parity and synthetic positions

Put-call parity is a no-arbitrage identity: for European options on the same underlying, strike, and expiry,

$$
C - P = S e^{-qT} - K e^{-rT}
$$

If this is violated, you can construct a risk-free profit. Parity is what makes the options market consistent — and what lets you build *synthetic* equity positions from options alone.

This short chapter covers the parity relationship and the synthetic positions traders use it for.

## The identity

For a European call $C$ and put $P$ with the same $(S, K, T, r, q)$:

$$
C(S, K, T) - P(S, K, T) = S e^{-qT} - K e^{-rT}
$$

The right side: a long position in $e^{-qT}$ shares of underlying (which costs $S e^{-qT}$) plus a short position in $K e^{-rT}$ of cash. The left side: a synthetic long forward (long call + short put at the same strike).

## Synthetic positions you can build

### Synthetic long underlying

Long call + short put (same strike & expiry) = synthetic long underlying. Cost: $C - P = S e^{-qT} - K e^{-rT}$. Net delta = +1.

When to use: you want long exposure without buying the underlying (capital constraint, tax considerations, ability to lever).

### Synthetic short underlying

Short call + long put = synthetic short. Net delta = -1.

When to use: shorting the underlying is hard or expensive (hard-to-borrow). Synthetic short via options sidesteps the borrow constraint.

### Conversion

Long stock + short call + long put (same strike, same expiry) = a riskless position with profit equal to dividends - interest - parity residual. Used by market-makers to arbitrage parity violations.

### Reversal

Short stock + long call + short put = inverse of conversion. Also riskless when parity holds; arbitrage when it doesn't.

## A worked example: checking for parity violations

```python
import pandas as pd
import numpy as np
from engine.data import YFinanceFeed

feed = YFinanceFeed()
chain = feed.option_chain("SPY")

calls = chain.calls
puts = chain.puts
# Match calls and puts on (strike, expiry)
matched = calls.merge(puts, on=["strike", "expiry"], suffixes=("_call", "_put"))
matched["call_mid"] = (matched["bid_call"] + matched["ask_call"]) / 2
matched["put_mid"] = (matched["bid_put"] + matched["ask_put"]) / 2

S = chain.spot
r, q = 0.045, 0.013
matched["T"] = (matched["expiry"] - pd.Timestamp.utcnow()).dt.days / 365
matched["parity_lhs"] = matched["call_mid"] - matched["put_mid"]
matched["parity_rhs"] = S * np.exp(-q * matched["T"]) - matched["strike"] * np.exp(-r * matched["T"])
matched["residual_bps"] = (matched["parity_lhs"] - matched["parity_rhs"]) / S * 10_000
print(matched[["strike", "T", "parity_lhs", "parity_rhs", "residual_bps"]].head(20))
```

For liquid options, residuals should be within ±2-5 bps. Larger residuals indicate:

- Stale mid (one leg's mid is from an old print).
- Hard-to-borrow on the underlying (forces parity to depart by the borrow rate).
- A real arbitrage opportunity (rare and quickly closed by HFT).

## Using parity for IV recovery

Quoted IVs for deep ITM options are usually unreliable (vega → 0, last-trade-based). Use parity to recover them from the deep OTM side:

```
For a strike K with C, P on the chain:
  C_implied_from_P = P + S * e^{-qT} - K * e^{-rT}
  IV from C_implied_from_P
```

For an ATM-skewed equity option, the OTM call and the OTM put have well-defined IVs; the ITM versions inherit via parity. The "smile" is then symmetric in log-moneyness — useful when building no-arbitrage surfaces (Module 15).

## Borrow cost from parity

The parity residual, on a stock that's hard to borrow, encodes the borrow cost:

$$
C - P = S e^{-(q + b)T} - K e^{-rT}
$$

where $b$ is the implied borrow rate. Solving:

$$
b = -\frac{1}{T} \ln\left(\frac{C - P + K e^{-rT}}{S}\right) - q
$$

For a name with a 10-bp parity residual on a 90-day option, the implied annualised borrow rate is around 4 bps × 4 = 16 bps. For hard-to-borrow specials, this can be 100s of bps.

```python
def implied_borrow(C, P, S, K, T, r, q):
    """Recover the implied borrow rate from parity residual."""
    target_disc = (C - P + K * np.exp(-r * T)) / S
    if target_disc <= 0:
        return float("nan")
    return -np.log(target_disc) / T - q
```

This is one of the cleanest ways to estimate borrow cost from options market data — useful for sizing short positions when the broker's borrow rate is opaque.

## Pitfalls

!!! warning "American options break parity"
    The identity $C - P = S - K e^{-rT}$ only holds for European options. American puts have early-exercise premium, so American parity is an inequality, not an equality.

!!! warning "Dividend dates within the option's life"
    Discrete dividends affect parity. The right form uses the present value of expected dividends, not just continuous yield.

!!! warning "Bid-ask vs mid"
    Parity holds for *mid* prices, not for the bid-ask. A trader who tries to construct a parity arbitrage will pay the spread on every leg and the trade is usually unprofitable.

!!! warning "Settlement timing"
    Quarterly index options (SPX) settle at the open. Monthly index options settle at the close. Their parity reference price differs.

## Bottom line

Put-call parity is:

- **An exact no-arbitrage identity** for European options.
- **Approximate for American options** (call exact under no dividends; put has early-exercise premium).
- **The right way to construct synthetic positions** (long forward = long call + short put).
- **A diagnostic** for stale quotes, borrow costs, dividend assumptions.

## End of Module 14

You have the working options-pricing foundation. The next module — **Vol Surface and Greeks** — takes the smile seriously: SVI parameterisation, SABR, Heston, local vol, dealer gamma exposure (GEX), and deep hedging. The full options machinery is ready by the end of Module 15.

Continue to **[Module 15 — Vol Surface and Greeks](../15-vol-surface-greeks/index.md)**.
