# Avellaneda-Stoikov market making in detail

Module 13 chapter 4 introduced Avellaneda-Stoikov (AS) as an RL benchmark. This chapter is the full implementation — the closed-form quotes, the production refinements, and the practical workflow for live deployment.

## The model recap

A market maker holds inventory $q$, posts bid at $S - \delta^b$ and ask at $S + \delta^a$. The mid evolves as Brownian motion $dS = \sigma dW$. Orders arrive with intensity $\lambda(\delta) = A e^{-\kappa \delta}$ — fewer fills at wider quotes.

Inventory penalty: the MM wants to be flat near close, so utility is $-\gamma \cdot q^2$ at terminal time $T$.

## The closed-form quotes

For the indifference price:

$$
r(s, q, t) = s - q \gamma \sigma^2 (T - t)
$$

This is the "reservation price" — the mid the MM would prefer their book centred on, given inventory $q$. Long inventory → reservation below mid (MM wants to sell). Short → above mid.

Optimal half-spread (symmetric):

$$
\delta^{a} + \delta^{b} = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \log\left(1 + \frac{\gamma}{\kappa}\right)
$$

Asymmetric ask vs bid offsets:

$$
\delta^a = (r - s) + \frac{1}{2}(\delta^a + \delta^b), \quad \delta^b = (s - r) + \frac{1}{2}(\delta^a + \delta^b)
$$

In code (clean implementation):

```python
import numpy as np


def avellaneda_stoikov_quotes(spot: float, inventory: int, t_remaining: float,
                                sigma: float, gamma: float, kappa: float, A: float) -> dict:
    """Returns the optimal bid and ask prices."""
    res = spot - inventory * gamma * sigma ** 2 * t_remaining
    half_spread = 0.5 * gamma * sigma ** 2 * t_remaining + (1 / gamma) * np.log(1 + gamma / kappa)
    bid = res - half_spread
    ask = res + half_spread
    return {"bid": float(bid), "ask": float(ask), "reservation": float(res),
            "half_spread": float(half_spread)}
```

## Calibrating the parameters

To use AS, you need:

- $\sigma$ — local volatility of the mid (in price units, not log). Estimate from recent returns.
- $\gamma$ — risk aversion. The most important free parameter. Typically tuned to match the desired inventory size.
- $\kappa$ — exponential decay of arrival intensity with quote distance. Estimate by fitting an exponential to observed fill rates at different quote distances.
- $A$ — base arrival intensity at zero distance from mid. Estimate from observed fill counts.

```python
def calibrate_arrival(historical_fills: list[dict], distance_bins: list[float]) -> tuple[float, float]:
    """Fit A * exp(-kappa * distance) to fill rates by distance from mid.
    historical_fills: list of {distance, time_window, n_fills}."""
    import numpy as np
    from scipy.optimize import curve_fit
    distances = np.array([f["distance"] for f in historical_fills])
    rates = np.array([f["n_fills"] / f["time_window"] for f in historical_fills])
    def expmodel(d, A, kappa): return A * np.exp(-kappa * d)
    popt, _ = curve_fit(expmodel, distances, rates, p0=[1.0, 1.0], bounds=(0, [100, 100]))
    return popt[0], popt[1]
```

For a liquid US stock, typical SoP values: $\kappa \approx 1.5$-$3.0$ per dollar from mid; $A \approx 5$-$20$ fills per second at the mid.

## Adding asymmetry to the arrival rate

The basic AS assumes symmetric arrival rates. In reality, retail flow is often one-sided (e.g., more retail buys around earnings). The model extends to asymmetric $A^b, A^a$:

$$
\delta^a + \delta^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \log\left(1 + \frac{\gamma}{\kappa}\right)
$$

(same)

$$
\text{Adjusted skew: } r = s + \frac{1}{\gamma} \log\left(\frac{A^b \kappa}{A^a \kappa}\right) - q \gamma \sigma^2 (T - t)
$$

When $A^b > A^a$ (more buyers), the MM moves the book slightly up (sell into the demand). This is what real shops do beyond pure AS.

## A live quoting loop

```python
class ASMarketMaker:
    def __init__(self, target_inventory: int = 0, gamma: float = 0.01,
                  sigma_initial: float = 0.5, kappa: float = 2.0, A: float = 10.0):
        self.gamma = gamma; self.sigma = sigma_initial
        self.kappa = kappa; self.A = A
        self.target_inventory = target_inventory
        self.inventory = 0
        self.last_quotes = {"bid": None, "ask": None}

    def on_mid_change(self, new_mid: float, t_remaining: float):
        """Recompute quotes when the mid changes."""
        quotes = avellaneda_stoikov_quotes(
            new_mid, self.inventory - self.target_inventory,
            t_remaining, self.sigma, self.gamma, self.kappa, self.A,
        )
        # Replace existing quotes if they've shifted enough
        if (self.last_quotes["bid"] is None or
                abs(quotes["bid"] - self.last_quotes["bid"]) > self.sigma * 0.1):
            place_bid(quotes["bid"])
            place_ask(quotes["ask"])
            self.last_quotes = quotes
        return quotes

    def on_fill(self, side: str, size: int, price: float):
        if side == "B":
            self.inventory += size
        else:
            self.inventory -= size
        # Update sigma estimate (exponential smoothing)
        # ...

    def on_vol_estimate(self, new_sigma: float):
        self.sigma = new_sigma
```

The loop is event-driven: mid changes → re-quote; fills update inventory → re-quote; vol estimate updates → re-quote.

## Production refinements beyond pure AS

Real market-makers layer on:

1. **Adverse selection penalty** — widen quotes when recent trades have been one-sided (informed flow). Use VPIN (chapter 2).
2. **Inventory hard cap** — refuse to quote one side when at max inventory.
3. **Multi-tier quotes** — post at multiple levels with decreasing size, capturing more fills without all going at the top.
4. **Toxic-flow detection** — pull quotes when a counterparty's trades are systematically predictive of next-mid moves.
5. **Cross-asset hedging** — for an ETF MM, hedge inventory by trading the constituent stocks.

All of these reduce realised P&L variance and limit toxic-flow losses.

## A worked example: simulating AS on a simple price path

```python
import numpy as np
from collections import deque

rng = np.random.default_rng(0)
T = 1000             # 1000 seconds
sigma_mid = 0.5      # price units per sqrt(second)
S = np.cumsum(rng.normal(0, sigma_mid, T)) + 100.0
mm = ASMarketMaker(gamma=0.01, sigma_initial=sigma_mid, kappa=2.0, A=5.0)

inventory_history = []
pnl = 0.0; cash = 0.0
for t in range(T):
    q = mm.on_mid_change(S[t], t_remaining=(T - t))
    # Simulate fills with Poisson-ish arrival
    fill_buy = rng.random() < mm.A * np.exp(-mm.kappa * (q["bid"] - q["bid"] + 0.01)) * (1/T)  # simplified
    fill_sell = rng.random() < mm.A * np.exp(-mm.kappa * (q["ask"] - q["ask"] + 0.01)) * (1/T)
    if fill_buy:
        mm.on_fill("B", 1, q["bid"]); cash -= q["bid"]
    if fill_sell:
        mm.on_fill("S", 1, q["ask"]); cash += q["ask"]
    inventory_history.append(mm.inventory)

mark_to_market = cash + mm.inventory * S[-1]
print(f"Final P&L: {mark_to_market:.2f}")
print(f"Max abs inventory: {max(abs(i) for i in inventory_history)}")
```

In a clean simulation matching AS's assumptions, you should see the inventory bounded (the gamma penalty keeps it near zero) and P&L positive on average (the captured spread).

## Pitfalls

!!! warning "Gamma too small → unbounded inventory"
    A near-zero gamma means the MM doesn't care about inventory. It will accumulate large directional positions on top of its market-making book. Tune gamma to bound inventory at acceptable levels.

!!! warning "Sigma estimate stale"
    If sigma is updated only daily, you under-react to intraday vol changes. Use realised kernels (chapter 3) on the past 10-30 minutes.

!!! warning "Quote thrashing"
    Re-quoting on every tiny mid change creates huge cancel-replace volume and broker fees. Use the 0.1σ "minimum move" threshold in the example, or post at the limit-tier and let the book come to you.

!!! warning "Theory vs reality"
    Pure AS assumes Poisson arrivals and Brownian mid. Both are violated. Real MMs deviate from theoretical AS — and the RL approach (Module 13) is one way to learn the corrections empirically.

## Bottom line

Avellaneda-Stoikov gives you the **structural backbone** of market making: a closed-form, inventory-aware quote-pair. For deployment:

- Calibrate $(\sigma, \kappa, A)$ from live data.
- Tune $\gamma$ to your inventory-risk tolerance.
- Layer on adverse-selection penalties, inventory caps, and toxic-flow detection.
- Use RL (Module 13) to learn the deviations from pure AS that work for your specific market.

Continue to **[Almgren-Chriss optimal execution in detail](05-almgren-chriss.md)**.
