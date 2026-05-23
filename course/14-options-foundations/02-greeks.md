# All the Greeks: first, second, and the cross ones you've never used

The Greeks are the partial derivatives of the option price with respect to the inputs. Every options trader knows delta, gamma, vega, theta, rho. Fewer know vanna, charm, vomma, speed, color, zomma. The latter group is where dealer-flow analysis (Module 18) lives. This chapter covers them all, with intuition and code from `engine.options.greeks`.

## First-order Greeks

### Delta — sensitivity to underlying price

$$
\Delta = \frac{\partial C}{\partial S}
$$

For a call: $\Delta = e^{-qT} N(d_1) \in (0, 1)$.
For a put: $\Delta = -e^{-qT} N(-d_1) \in (-1, 0)$.

Intuition: how many shares of underlying you'd hold to hedge a 1-share-equivalent option. ATM delta is ~0.5 (call). Deep ITM call delta → 1. Deep OTM call delta → 0.

### Vega — sensitivity to volatility

$$
\nu = \frac{\partial C}{\partial \sigma} = S e^{-qT} \phi(d_1) \sqrt{T}
$$

Same for calls and puts. Quoted "per 1% change in vol" by dividing by 100. Vega is **largest at the money** and **proportional to $\sqrt{T}$** — long-dated ATM options are the most vol-sensitive.

### Theta — sensitivity to time

$$
\Theta = \frac{\partial C}{\partial t}
$$

Negative for long options (they lose value as time passes). Quoted per day by dividing by 365. Theta is largest for short-dated ATM options.

### Rho — sensitivity to risk-free rate

$$
\rho_\text{call} = K T e^{-rT} N(d_2)
$$

Usually small for equity options. Matters for long-dated options and for rates products.

## Second-order Greeks

### Gamma — second derivative wrt underlying

$$
\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt T}
$$

Same for calls and puts. Largest at the money. **The gamma of the option is what makes delta-hedging unprofitable on the diagonal** — you have to constantly re-hedge as price moves.

### Vanna — cross sensitivity to underlying *and* vol

$$
\text{vanna} = \frac{\partial^2 C}{\partial S \partial \sigma} = -e^{-qT} \phi(d_1) \frac{d_2}{\sigma}
$$

How does delta change when vol moves? Positive vanna means "delta increases when vol rises" — important for risk-reversal and skew trades.

### Charm — delta decay

$$
\text{charm} = \frac{\partial \Delta}{\partial t}
$$

How does delta change as time passes? Largest for ATM options near expiry. Charm is the reason **delta-neutral positions need rebalancing as time goes by** even without price moves.

### Vomma (volga) — vega's vol sensitivity

$$
\text{vomma} = \frac{\partial \nu}{\partial \sigma} = \nu \cdot \frac{d_1 d_2}{\sigma}
$$

How does vega change when vol changes? Zero at the money. **Long vomma means you make money on big vol moves in either direction** — characteristic of OTM strangles.

## Third-order — for the truly curious

- **Speed** — $\partial \Gamma / \partial S$. How gamma changes with the underlying. Used in delta hedging at high-frequency.
- **Color** — $\partial \Gamma / \partial t$. How gamma decays with time.
- **Zomma** — $\partial \Gamma / \partial \sigma$. How gamma reacts to vol changes.

Rarely matter outside HFT and dealer-flow modelling. We compute the first eight in `engine.options.greeks`; for the rest, look at QuantLib.

## The course's Greeks function

```python
from engine.options import greeks

g = greeks(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, kind="call")
print(g.delta, g.gamma, g.vega, g.theta, g.vanna)
# 0.6368   0.0188   37.5240   -6.4140   -0.2814
# (vanna is negative for this ATM call because d2 > 0 with the chosen rates)
```

The dataclass exposes attribute access. For vectorised computation across a chain, reach into `engine.options.black_scholes._d1_d2` and assemble what you need.

## A worked example: net dealer gamma exposure

A teaser for Module 18's GEX scanner. Dealers in equity options are typically **short calls** (sold to retail) and **long puts** (bought as hedges by institutions). The net gamma their book carries determines whether they buy or sell on rallies and dips.

```python
import numpy as np
import pandas as pd
from engine.options import greeks

def chain_gamma(chain: pd.DataFrame, spot: float, T_years: float,
                r: float = 0.045, q: float = 0.013, contract_size: int = 100) -> pd.Series:
    """Per-row gamma contribution from this option position."""
    gammas = []
    for _, row in chain.iterrows():
        g = greeks(spot, row["strike"], T_years, r, q, row["iv"], kind=row["right"])
        gammas.append(g.gamma * row["open_interest"] * contract_size)
    return pd.Series(gammas, index=chain.index)

# Dealer position convention: short calls (negative gamma), long puts (positive gamma)
# net gamma = (-call_gamma) + (+put_gamma) = put_gamma - call_gamma
```

The full GEX scanner (Module 18) combines this with the underlying's spot to estimate the dollar value of dealer hedging flow per unit price move. Big-picture: high positive GEX → dealers dampen moves (sell into rallies, buy into dips). Negative GEX → dealers chase moves (buy into rallies, sell into dips, often into stops).

## Verifying Greeks numerically

If you're unsure about a Greek's sign or magnitude, finite-difference is the sanity check:

```python
from engine.options import price

S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
h = 0.01
delta_fd = float((price(S + h, K, T, r, q, sigma) - price(S - h, K, T, r, q, sigma)) / (2 * h))
print(f"FD delta: {delta_fd:.4f}")          # should match analytic delta

gamma_fd = float((price(S + h, K, T, r, q, sigma) - 2 * price(S, K, T, r, q, sigma) +
                  price(S - h, K, T, r, q, sigma)) / h ** 2)
print(f"FD gamma: {gamma_fd:.4f}")          # should match analytic gamma
```

Always cross-check a new analytical Greek against finite difference. Small discrepancies (1e-5) are numerical; large ones are bugs.

## Pitfalls

!!! warning "Greek units"
    Delta is per share of underlying. Vega is per 1.0 change in sigma (= 100 vol points). Quote conventions vary. The `engine.options.greeks` outputs are in the "raw" form; divide by 100 for per-vol-point vega, by 365 for per-day theta, etc.

!!! warning "Position vs option Greeks"
    The Greek of one option contract is computed for one share. Multiply by contract multiplier (100 for US equity, varies for other markets) and by position size to get position Greek.

!!! warning "Greek aggregation across underlyings"
    Adding delta across SPY and AAPL positions only makes sense if you convert to a common denominator (e.g., notional dollar delta). Naive sum is meaningless.

!!! warning "Greeks at expiration"
    Delta jumps to 0 or 1 instantly at expiry; gamma spikes to infinity at ATM. Numerically, near expiry, all Greeks are unstable. Use binomial or LSMC for very near-expiry pricing.

## Bottom line

For working with options:

- **Memorise the signs and intuitions** of delta, gamma, vega, theta.
- **Know vanna and charm exist** — they explain a lot of dealer flow.
- **Compute Greeks with `engine.options.greeks`** for analytical correctness.
- **Verify with finite-difference** when in doubt.
- **Aggregate to position dollar terms** before reasoning about portfolio risk.

Continue to **[Implied volatility — robust solvers](03-implied-vol.md)**.
