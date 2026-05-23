# Local volatility via Dupire

A local volatility model assumes the underlying follows:

$$
dS_t = (r - q) S_t \, dt + \sigma_\text{loc}(t, S_t) S_t \, dW_t
$$

where $\sigma_\text{loc}$ is a deterministic function of time and spot. The remarkable result (Dupire 1994): given any arbitrage-free implied vol surface, there is a *unique* local-vol function that, when used to price European options, recovers the surface exactly.

This makes local-vol the "perfect calibration" for European prices. Its weakness: the model's *dynamics* (how the surface evolves with spot) are wrong for many real markets, which limits its use for exotics and hedging.

## The Dupire formula

For European call prices $C(K, T)$ on a continuous surface:

$$
\sigma_\text{loc}^2(T, K) = \frac{\frac{\partial C}{\partial T} + (r - q) K \frac{\partial C}{\partial K} + q C}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}
$$

The denominator is proportional to the implied (risk-neutral) density of $S_T$ — that's why butterfly arbitrage (negative second derivative) breaks Dupire.

In code:

```python
import numpy as np

def dupire_local_vol(K_grid, T_grid, C_grid, r, q):
    """K_grid: (n_K,) array of strikes. T_grid: (n_T,) array of expiries.
       C_grid: (n_T, n_K) array of call prices.
       Returns local vol on a (n_T, n_K) grid."""
    # Numerical partial derivatives via finite differences
    dC_dT = np.gradient(C_grid, T_grid, axis=0)
    dC_dK = np.gradient(C_grid, K_grid, axis=1)
    d2C_dK2 = np.gradient(dC_dK, K_grid, axis=1)
    K = K_grid[None, :]
    num = dC_dT + (r - q) * K * dC_dK + q * C_grid
    denom = 0.5 * K ** 2 * d2C_dK2
    sigma_sq = np.where(denom > 1e-10, num / denom, np.nan)
    return np.sqrt(np.maximum(sigma_sq, 0.0))
```

To use it, start from a fitted SVI/SSVI smile (chapter 1), convert IVs to call prices, apply Dupire numerically, and you have a local-vol surface.

## A more stable approach: variance in $T, K$

Numerical derivatives on a noisy market surface produce noisy local vols (especially the second derivative in K). The standard trick: work with **total implied variance** $w(k, T) = \sigma^2(K, T) \cdot T$ in log-moneyness $k = \log(K/F)$, and use the equivalent identity:

$$
\sigma_\text{loc}^2(t, K) = \frac{\partial w / \partial T}{1 - \frac{k}{w} \frac{\partial w}{\partial k} + \frac{1}{2} \frac{\partial^2 w}{\partial k^2} + \frac{1}{4} \left(\frac{\partial w}{\partial k}\right)^2 \left(\frac{1}{w} - \frac{1}{4} - \frac{k^2}{w^2}\right)}
$$

Looks complicated; computes smoothly because SVI's $w(k)$ is analytically nice. Most production local-vol implementations use this form.

## When local vol is the right tool

- **Pricing exotics that are mostly path-dependent on the underlying** (barrier options, knockouts). Local vol gets these about right.
- **Storing the market surface in a compact form** that prices Europeans exactly.
- **As a baseline for stochastic-local-vol (SLV) hybrid models** that combine local vol's exact European fit with stochastic vol's better exotic dynamics.

## When local vol breaks

The model's forward-looking smile dynamics are off. Specifically:

- **The forward smile flattens too fast.** If today's 1-year smile is steep, the local-vol model predicts that in 6 months the 6-month smile will be nearly flat. Real markets don't behave that way.
- **Volatility of volatility is too low.** Local vol has zero vol-of-vol by construction. For trades that care about it (cliquets, vol swaps), local vol underprices.

For these reasons, the industry standard for serious exotics pricing is **stochastic-local-vol (SLV)**: a model with a stochastic-vol component (Heston-like) and a local-vol multiplier that perfectly fits the European surface.

## A worked example: local vol from a synthetic surface

```python
import numpy as np
from engine.options import price

# Synthetic surface: SVI total variance with mild skew
def smile_iv(K, T, S0=100.0, atm=0.20, skew=-0.5):
    k = np.log(K / S0)
    return atm + skew * k + 0.5 * skew ** 2 * k ** 2

K_grid = np.linspace(70, 130, 30)
T_grid = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
S0, r, q = 100.0, 0.04, 0.0

C_grid = np.zeros((len(T_grid), len(K_grid)))
for i, T in enumerate(T_grid):
    for j, K in enumerate(K_grid):
        sigma = smile_iv(K, T, S0)
        C_grid[i, j] = float(price(S0, K, T, r, q, sigma, kind="call"))

lv = dupire_local_vol(K_grid, T_grid, C_grid, r, q)
# lv has shape (len(T_grid), len(K_grid))
print(lv[2])     # 0.25-year local vols across strikes
```

For a stable fit on real data, smooth the surface first (apply SVI fits per slice, sample $C$ on a dense grid, then differentiate).

## Pitfalls

!!! warning "Numerical noise in second derivatives"
    Even with a smooth surface, finite-difference $\partial^2 C / \partial K^2$ is noisy. Use SVI's analytical derivatives where possible.

!!! warning "Local vol explodes at the wings"
    Where the market gives noisy quotes (far OTM), the implied density goes to zero, the denominator vanishes, and local vol blows up. Always cap local vol at a reasonable maximum.

!!! warning "Time mesh"
    Local vol is needed on a fine T-mesh for PDE-based exotic pricing. Interpolate the available expiries; don't extrapolate beyond the market's longest expiry.

!!! warning "Calibration to thin chains"
    A surface with few strikes per expiry produces a poorly-conditioned Dupire problem. Parametric SVI/SSVI fit is what you want as the input — never raw market quotes.

## Bottom line

Local vol via Dupire is the right way to get a surface that exactly prices European options. It is **not** the right model for forward-smile dynamics or for exotics that depend on vol-of-vol. For production:

- Fit SVI/SSVI to the market surface first.
- Compute local vol analytically from the SVI fit's analytical derivatives.
- For exotics needing dynamics, switch to stochastic-local-vol (SLV).

Continue to **[Dealer GEX, vanna, and charm flows](05-dealer-flows.md)**.
