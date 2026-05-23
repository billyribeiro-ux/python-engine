# Dealer GEX, vanna, and charm flows

The single most underappreciated signal in equity index trading: **dealer hedging flow**. Options market-makers run delta-neutral books, which means they're constantly buying or selling the underlying to offset their options exposure. The aggregate of those hedges — across all open options on SPY, QQQ, etc. — is large enough to move the market.

This chapter is the mechanics, the metric (Gamma Exposure, GEX), and the canonical "frontier" signals that come out of it.

## The dealer's position

The convention (somewhat oversimplified):

- **Calls**: dealers are typically **short**. Retail and yield-enhancers buy calls; dealers sell them.
- **Puts**: dealers are typically **long**. Institutions buy puts as portfolio hedges; dealers buy them as offsets to call sales (and from skew traders).

So the dealer's options book is roughly **short gamma on the upside, long gamma on the downside** — both in the conventional sign (since short call gamma is negative, long put gamma is positive). The net position depends on the strike distribution.

## Gamma Exposure (GEX)

For each option in the chain, the dealer's gamma exposure (per unit underlying price move) is:

$$
\text{GEX}_i = \text{gamma}_i \cdot \text{OI}_i \cdot \text{contract size} \cdot S^2 \cdot 0.01
$$

The $S^2$ and $0.01$ convert "per share gamma" into "dollar P&L per 1% spot move" (the common GEX convention).

Net dealer GEX (with the long-put / short-call convention):

$$
\text{Net GEX} = \sum_\text{puts} \text{GEX}_i^\text{put} - \sum_\text{calls} \text{GEX}_i^\text{call}
$$

In code, using `engine.options.greeks`:

```python
import numpy as np
import pandas as pd
from engine.options import greeks


def gex(chain: pd.DataFrame, spot: float, T_years_col: str = "T_years",
        r: float = 0.045, q: float = 0.013, contract_size: int = 100) -> pd.Series:
    """Per-row dealer GEX in dollars per 1% spot move.
    Dealer is long puts, short calls (convention)."""
    rows = []
    for _, row in chain.iterrows():
        g = greeks(spot, row["strike"], row[T_years_col], r, q, row["iv"], kind=row["right"])
        # Sign: long puts contribute positive gamma; short calls contribute negative gamma to dealer's book.
        sign = +1 if row["right"] == "put" else -1
        rows.append(sign * g.gamma * row["open_interest"] * contract_size * spot ** 2 * 0.01)
    return pd.Series(rows, index=chain.index)


def net_gex(chain: pd.DataFrame, spot: float, **kw) -> float:
    return float(gex(chain, spot, **kw).sum())
```

## Why net GEX matters

When net GEX is **positive**: dealers profit from low realised vol. They hedge by buying dips and selling rallies (mean-reverting flow). Reaction: market moves are dampened.

When net GEX is **negative**: dealers lose from low realised vol. They hedge by chasing — selling into dips, buying into rallies (trend-following flow). Reaction: market moves are amplified, often into stops.

This is one of the cleanest predictive signals about *realised volatility regimes* available from publicly observable data. SpotGamma, SqueezeMetrics, and similar services built businesses on it.

## Zero gamma

A key concept: the **zero-gamma price** — the underlying level at which net GEX crosses zero. Above this level, dealers are net-positive gamma; below it, net-negative.

Use it as a regime threshold: if spot is well above zero-gamma, expect dampened moves. If spot dips below, expect amplification.

```python
def zero_gamma_level(chain: pd.DataFrame, T_years_col: str = "T_years",
                    spot_range: tuple = None) -> float:
    """Search for the spot level where net GEX changes sign."""
    if spot_range is None:
        atm = chain["strike"].median()
        spot_range = (atm * 0.9, atm * 1.1)
    from scipy.optimize import brentq
    def f(s): return net_gex(chain, s, T_years_col=T_years_col)
    try:
        return float(brentq(f, spot_range[0], spot_range[1]))
    except ValueError:
        return float("nan")    # no sign change in the range
```

## Vanna flow

Vanna = $\partial \Delta / \partial \sigma$. As implied vol rises, the dealer's delta changes; they hedge by trading the underlying. **Net vanna exposure** measures this:

$$
\text{Net Vanna} = \sum_\text{puts} \text{vanna}_i \cdot \text{OI}_i \cdot \text{contract size} - \sum_\text{calls} \text{vanna}_i \cdot \text{OI}_i \cdot \text{contract size}
$$

Big net vanna means a vol spike will trigger large hedging flows. Vanna's biggest contributions come from out-of-the-money options near expiry — the "wings."

## Charm flow

Charm = $\partial \Delta / \partial t$. Dealers' delta drifts as time passes, even without spot moves. They hedge that drift continuously. Aggregate charm is largest in the days **right before expiry** and concentrated near ATM strikes.

The famous "**OpEx Friday**" effect: monthly equity option expiry frequently coincides with strange-looking moves into 4pm because dealers are unwinding big charm-driven positions.

## A canonical "Frontier" scanner: GEX inflection

The pattern: estimate **net GEX as spot ranges across nearby levels**. The slope $\partial \text{GEX} / \partial S$ tells you how fast dealer positioning is changing. Large negative slope near current spot is a **fragile** condition — a small move flips the regime.

```python
def gex_inflection_score(chain: pd.DataFrame, spot: float, T_years_col: str,
                          spot_range_pct: float = 0.05) -> dict:
    """Compute net GEX at current and nearby spot levels."""
    spots = np.linspace(spot * (1 - spot_range_pct), spot * (1 + spot_range_pct), 11)
    gexs = [net_gex(chain, s, T_years_col=T_years_col) for s in spots]
    # Slope at current spot via finite difference
    idx = len(spots) // 2
    slope = (gexs[idx + 1] - gexs[idx - 1]) / (spots[idx + 1] - spots[idx - 1])
    zero_gamma = zero_gamma_level(chain, T_years_col=T_years_col,
                                  spot_range=(spots[0], spots[-1]))
    return {"current_gex": gexs[idx], "slope": slope, "zero_gamma": zero_gamma,
            "distance_to_zero_pct": (zero_gamma - spot) / spot * 100 if np.isfinite(zero_gamma) else float("nan")}
```

A high-quality alert: net GEX is positive but the zero-gamma level is within 1% of spot, AND vanna exposure is high. That's a setup where a small vol spike could push spot below zero-gamma and trigger an amplification cascade.

## What you actually need to compute this for real

- **Live options chain** with strikes, expiries, IVs, and **open interest** (not just volume).
- **Underlying spot** real-time.
- **Greeks per contract** — re-compute from IV every quote update.
- **Aggregation per underlying** (and per expiry, if you want the term-structure breakdown).

For SPY, the full chain has 5,000-10,000 active contracts. Recompute GEX every minute on a single CPU is easy; intraday at higher frequency needs a careful pipeline.

The course's `engine.options.greeks` is the building block. Module 18 (Scanners) puts these signals together into a complete dealer-positioning scanner.

## Pitfalls

!!! warning "The dealer-direction convention isn't always right"
    For some products (single-stock options, expiring weeklies), retail positioning is different. The long-put / short-call assumption is a heuristic, not a guarantee. Cross-check with put/call open-interest ratios.

!!! warning "Stale OI"
    Open interest updates with a one-day lag. Your "real-time" GEX is actually based on yesterday's positions. For most regime-monitoring, that's fine.

!!! warning "Hedge-vs-trade conflation"
    Not all flow is hedging. Active vol traders take their own positions; their flow shows up in OI too. The "dealer" is an idealisation.

!!! warning "Out-of-the-money OI dominates everywhere"
    Naive GEX summed over the whole chain is dominated by far-OTM puts (huge OI from portfolio hedgers). Restrict to relevant strikes (e.g., within ±10% of spot) for meaningful signals.

## Bottom line

Dealer GEX and its cousins (vanna, charm) are some of the **highest-information signals** available about likely intraday volatility regime. They explain why some days feel pinned to a level and others feel runaway.

For a trading application:

- Compute net GEX, vanna, and charm at start of day.
- Track them intraday as spot moves.
- The **zero-gamma level** is your primary regime threshold.
- Combine with vol-of-vol signals (Module 18) for a complete dealer-positioning scanner.

Continue to **[Deep hedging](06-deep-hedging.md)**.
