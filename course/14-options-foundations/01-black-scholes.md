# Black-Scholes properly

The Black-Scholes-Merton formula (1973) is the most famous equation in finance for a reason: it gives a closed-form price for European options assuming the underlying is a geometric Brownian motion. Almost every later development — local vol, stochastic vol, SVI — is a way to *correct* Black-Scholes for one or another of its assumptions. Knowing it cold is non-negotiable.

## The model

Underlying follows GBM:

$$
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t
$$

with constant risk-free rate $r$, constant continuous dividend yield $q$, and constant volatility $\sigma$.

European call price at time 0, strike $K$, expiry $T$:

$$
C(S, K, T, r, q, \sigma) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

where:

$$
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2} \sigma^2) T}{\sigma \sqrt T}, \quad d_2 = d_1 - \sigma \sqrt T
$$

and $N(\cdot)$ is the standard normal CDF.

Put: $P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$.

That's the whole formula. The mathematical justification (Itô, risk-neutral measure, replicating portfolio) is in Hull, Shreve, or any options textbook — for our purposes, the formula above plus a working implementation is enough.

## The course's implementation

`engine.options.price` implements this, vectorised:

```python
from engine.options import price

# Single-strike call price
c = float(price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, kind="call"))
print(f"ATM 1y call: {c:.4f}")            # 10.4506

# Vectorised over strikes (great for surface fitting)
import numpy as np
strikes = np.array([85, 90, 95, 100, 105, 110, 115])
calls = price(100.0, strikes, 0.5, 0.05, 0.0, 0.25, kind="call")
print(calls)
```

The function returns a NumPy array; pass scalars or arrays for any of $(S, K, T, r, q, \sigma)$.

## What Black-Scholes gets right

- **No-arbitrage bounds** on option prices.
- **Put-call parity** — the relationship $C - P = S e^{-qT} - K e^{-rT}$ holds exactly.
- **The qualitative shape** of vega, gamma, theta.
- **Pricing of European at-the-money short-dated options** — close enough to market prices in calm regimes.

## What Black-Scholes gets wrong

- **Constant volatility** — markets have a vol surface (smile + term structure).
- **Constant rate** — bonds prove this isn't true, but for short-dated equity options the error is small.
- **Continuous trading and no transaction costs** — assumed for the hedging argument; ignored in pricing.
- **Log-normal returns** — actual returns have fat tails and skew.
- **No jumps** — overnight gaps and earnings moves are discontinuous.

The smile (next section) is what the market quotes to *correct* Black-Scholes for these failures. The Black-Scholes price plus the smile-implied vol is how most equity options are quoted in practice.

## A worked example: pricing an SPY option

```python
from engine.options import price
import numpy as np

# SPY at $475, 30-day call at strike $480
S = 475.0
K = 480.0
T = 30 / 365            # in years
r = 0.045               # short-rate proxy
q = 0.013               # SPY dividend yield
sigma = 0.18            # IV from market

c = float(price(S, K, T, r, q, sigma, kind="call"))
p = float(price(S, K, T, r, q, sigma, kind="put"))
print(f"Call: ${c:.2f}, Put: ${p:.2f}")

# Verify put-call parity
parity_residual = (c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))
print(f"Parity residual: {parity_residual:.2e}")    # ~ 1e-10
```

The put-call parity check is the simplest possible sanity check for any Black-Scholes implementation.

## Vectorised pricing for a chain

```python
import pandas as pd
from engine.options import price

chain = pd.DataFrame({
    "strike": [460, 465, 470, 475, 480, 485, 490, 495, 500],
    "expiry_days": [30] * 9,
    "iv": [0.20, 0.19, 0.185, 0.18, 0.18, 0.185, 0.19, 0.20, 0.21],   # the smile
})
chain["call_theoretical"] = price(
    S=S, K=chain["strike"].values, T=chain["expiry_days"].values / 365,
    r=r, q=q, sigma=chain["iv"].values, kind="call",
)
print(chain)
```

You'd compare `call_theoretical` to the observed market price to spot dislocations — that's the basis of any options arbitrage strategy.

## Numerical considerations

- **`T → 0`** (at expiry) — the formula's `sigma * sqrt(T)` term goes to zero; `d1, d2 → ∞`. Our code clamps `T >= 1e-12` to avoid `NaN`s.
- **`sigma → 0`** — same issue. Code clamps.
- **Very deep ITM or OTM** — the option's value is dominated by intrinsic ± a tiny extrinsic. Black-Scholes is fine but watch implied-vol estimation (next chapter).
- **`r - q` very large** — for short rates, this approximation breaks down for long-dated options. Use the proper bond-curve.

## Pitfalls

!!! warning "T in days vs years"
    Black-Scholes' $T$ is in years. A 30-day option is $T = 30/365$, not $T = 30$. Off-by-365 is a classic.

!!! warning "Dividends as discrete cash vs continuous yield"
    Index ETFs (SPY, VOO) approximate continuous yield. Single stocks pay discrete dividends. For single-stock options near a dividend date, use the discrete-dividend model (subtract present value of the dividend from S before pricing).

!!! warning "Volatility as a percentage"
    `sigma=0.20` means 20%. `sigma=20` means 2000% — not what you meant. Always check.

!!! warning "American vs European"
    Black-Scholes is for European options (exercise at expiry only). For American options (early exercise), use the binomial tree (chapter 4).

## Bottom line

Black-Scholes is the calibration target everything else corrects. Memorise the formula. Verify any pricing implementation with put-call parity. The course's `engine.options.price` is your production-grade reference.

Continue to **[All the Greeks: first, second, and the cross ones you've never used](02-greeks.md)**.
