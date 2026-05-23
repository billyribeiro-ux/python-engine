# Calendar and diagonal spreads on IV term-structure

The implied vol term structure — the curve of ATM IVs against time-to-expiry — is typically upward-sloping in calm regimes (further-dated options have higher IV) and inverted in stress (front-dated options spike). A **calendar spread** is short the front-month, long the back-month — earning the spread when the term structure flattens or normalises. A **diagonal spread** is similar but with different strikes for the two legs.

These are options-only production strategies — they don't require a directional view, they don't require predicting volatility level, they just require the term structure to behave as it has historically.

## The structures

**Long calendar spread**:
- Sell front-month ATM call (or put), expiring in 30 days.
- Buy back-month ATM call (or put), expiring in 60 days.
- Same strike on both.
- Pay the difference in premium (back-month is more expensive).

**Diagonal spread**:
- Same as calendar but the back-month strike is different from the front-month.
- Lets you tilt the position toward bullish, bearish, or vol-skewed views.

## P&L profile

A long calendar spread benefits from:

1. **Time decay** — the front leg decays faster than the back (theta on calendars works for you).
2. **Term-structure normalisation** — if the curve was inverted at entry, it normalising back to upward-sloping is profitable.
3. **Mean-reversion in vol** — if both legs' IVs drift toward their average, the calendar P&L is approximately neutral but you've collected the theta.

Hurts:

1. **Big underlying moves** — the strike was ATM at entry; a large move makes both legs ITM or OTM, and the calendar loses its theta edge.
2. **Vol spikes** — if the back-month IV spikes more than the front, the back leg's vega beats the front's, and the calendar profits *or* loses depending on which IV moved more.

## When the term structure presents a setup

The trade lights up when:

- **Front-month IV is unusually high relative to back-month** (term structure inverted). The classic case: a known event (earnings, FOMC) in the front-month's life.
- **Both IVs are at multi-year lows** — calendar is cheap to enter, has positive expected return if vol regresses to mean.

The metric:

$$
\text{Term-Structure Slope} = \text{IV}_\text{back-month} - \text{IV}_\text{front-month}
$$

Negative slope = inverted = potential long-calendar setup. The most extreme inversions historically have been before FOMC meetings, before earnings, and during corrections.

## A simple scanner

```python
import pandas as pd
import numpy as np
from engine.options import implied_vol


def term_structure_scanner(chain_data: dict[tuple[str, pd.Timestamp], pd.DataFrame],
                            min_inversion: float = 0.05) -> pd.DataFrame:
    """
    chain_data: maps (underlying, expiry) → chain DataFrame with strike, mid, etc.
    Returns rows for symbols with inverted front-month IV.
    """
    rows = []
    for (symbol, _), groupings in pd.DataFrame(chain_data).groupby("underlying"):
        # Find front-month and second-month expiries
        # Compute ATM IV from the mid for each
        # Compare
        ...
    return pd.DataFrame(rows)
```

A real scanner needs live option chain data (Module 5) and the IV solver from Module 14. Module 18 puts this into the full scanner framework.

## A concrete strategy

For each candidate underlying:

1. Compute ATM IV for the front and back months from current bid-ask mid.
2. If $\text{IV}_\text{back} - \text{IV}_\text{front} > -3\%$ (not too inverted), skip.
3. If $\text{IV}_\text{back} - \text{IV}_\text{front} < -8\%$ (deeply inverted, probably justified), skip — the inversion likely has a fundamental reason (earnings, etc.) that won't normalise predictably.
4. **Open long-calendar at ATM** with delta around 50.
5. Exit when:
   - Front expires (let it go to settlement).
   - Term structure normalises (back - front > 0).
   - Loss exceeds 50% of initial debit.

```python
def calendar_spread_pnl_at_expiry(spot, K, IV_back_at_expiry, T_back_remaining,
                                    r, q, initial_debit):
    """P&L at front-month expiry: front leg expires worthless / intrinsic;
    back leg is valued at the new ATM."""
    from engine.options import price
    # Front leg (short): pays intrinsic if ITM
    front_payout = max(spot - K, 0)  # for a call
    # Back leg (long): worth the BS price at T_back_remaining
    back_value = float(price(spot, K, T_back_remaining, r, q, IV_back_at_expiry, kind="call"))
    # P&L = -front_payout + back_value - initial_debit
    return -front_payout + back_value - initial_debit
```

## Risk overlays

- **Position size**: each spread risks max 1-2% of capital (the initial debit is your max loss in the worst case).
- **No more than 5 spreads on the same underlying** — vega risk concentration.
- **Avoid known events** (earnings, FOMC) if the position's expiries straddle them — the IV dynamics around events are unpredictable.

## Why this works

Two persistent risk premia:

1. **Variance risk premium** — implied vol is on average higher than realised. The short-vol leg (front-month) of the calendar captures this premium.
2. **Term-structure premium** — investors pay a premium for the longer-dated option's protection. The back-month leg holds value as time passes.

A calendar combines both: short the over-priced front (cap on harvest), long the back (the protection).

## Modern variants

**IV percentile/rank filter**: only enter when IV rank is in the lower 30% — calendars are cheaper, expected mean reversion is upward.

**Vega-neutral diagonal**: choose the back strike so the spread's net vega is zero. Removes one risk dimension; the strategy becomes pure term-structure carry.

**Multi-leg term spreads**: short two front-months, long one mid-month, long one back-month — a "butterfly" across expiries. Even tighter risk profile, smaller premium.

## A worked example (synthetic data)

```python
from engine.options import price

S, K, r, q = 100.0, 100.0, 0.045, 0.013
front_T, back_T = 30/365, 60/365
front_IV, back_IV = 0.25, 0.18      # inverted term structure

# Initial premiums (sell front, buy back)
front_call = float(price(S, K, front_T, r, q, front_IV, kind="call"))
back_call = float(price(S, K, back_T, r, q, back_IV, kind="call"))
initial_debit = back_call - front_call
print(f"Initial debit: ${initial_debit:.2f}")

# At front-month expiry, suppose term structure normalised
# (front went away; back IV stayed at 0.20, say)
spot_at_expiry = S * 1.005          # tiny move
front_payout = max(spot_at_expiry - K, 0)
back_value = float(price(spot_at_expiry, K, back_T - front_T, r, q, 0.20, kind="call"))
pnl = -front_payout + back_value - initial_debit
print(f"P&L: ${pnl:.2f}")
```

Calendars typically realise their P&L right around front-month expiry. The trade isn't over until the front contract is closed/settled.

## Pitfalls

!!! warning "Liquidity"
    Multi-leg options have wide spreads. A 30-bp net edge dies if the bid-ask is 50 bps round-trip. Use only the most liquid underlyings (SPY, QQQ, IWM, AAPL, MSFT, etc.) for calendars.

!!! warning "Pin risk"
    At front-month expiry, if the underlying closes very near the strike, the front leg's settlement is uncertain (early exercise, dividend manipulation). Close the spread the day *before* expiry to avoid pin risk.

!!! warning "Term structure changes meaning"
    A bull market with VIX falling has calendars working through both legs decreasing IV (calendar's vega is positive on back). A bear market with VIX rising has both legs increasing, but front more than back — calendar loses on the differential.

!!! warning "Earnings hedge effects"
    Front-month IV spikes before earnings; back-month doesn't move much. A calendar entered just before earnings on a "deep front inversion" loses if the earnings move is moderate (vol crush hurts the back leg too).

## Bottom line

Calendar spreads are:

- **Theta and term-structure capture** trades, not directional.
- **Low-cost to enter, capped downside**, modest upside per trade.
- **Crowd less than directional strategies** — useful for diversification.
- **Require liquid underlyings** and careful term-structure timing.

Expected gross alpha per trade: 30-50% of initial debit. Net of costs: 15-25%. Realistic Sharpe across a portfolio of calendar spreads: 0.5-0.8.

## End of Module 16

You now have six battle-tested production strategies, each built on the course's components: HMM regime + momentum, Kalman pairs, GBM + conformal, vol-targeted carry, PEAD, and calendar spreads. The next module — **Frontier Strategies** — leaves the conservative playbook and gets into Hawkes processes, topological data analysis, transfer-entropy networks, signature methods, graph neural nets, and other techniques rarely seen in retail or even most prop shops.

Continue to **[Module 17 — Frontier Strategies](../17-frontier-strategies/index.md)**.
