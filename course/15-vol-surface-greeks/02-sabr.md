# SABR — the rates / FX skew model

SABR (Hagan, Kumar, Lesniewski, Woodward, 2002) is the standard parametric vol model for interest-rate options and FX. Unlike SVI (which fits a static smile), SABR is a *dynamic* model — a stochastic differential equation that produces a smile. That makes it well-suited for risk computation: shock the parameters and you get a consistent surface response.

## The SDE

$$
dF_t = \alpha_t F_t^\beta \, dW^1_t
$$
$$
d\alpha_t = \nu \alpha_t \, dW^2_t
$$
$$
\langle dW^1, dW^2 \rangle = \rho \, dt
$$

Four parameters: $\alpha$ (initial vol level), $\beta$ (CEV exponent — 0 for normal, 1 for log-normal, often around 0.5 for rates), $\rho$ (correlation), $\nu$ (vol-of-vol).

The forward price $F_t$ has stochastic volatility $\alpha_t$ with its own SDE; the two Brownians are correlated.

## The Hagan approximation

Hagan et al. derived an asymptotic formula for the Black implied vol as a function of strike $K$, forward $F$, expiry $T$, and the SABR parameters. The full formula is messy but tractable:

```python
import numpy as np

def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan SABR implied (log-normal) volatility. Robust for K != F and K == F."""
    if abs(F - K) < 1e-8:
        # ATM expansion
        FK_beta = F ** (1 - beta)
        a = ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
        b = 0.25 * rho * beta * nu * alpha / FK_beta
        c = ((2 - 3 * rho ** 2) / 24) * nu ** 2
        return (alpha / FK_beta) * (1 + (a + b + c) * T)
    log_FK = np.log(F / K)
    FK_pow = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_pow * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    num = alpha * (z / x_z)
    denom = FK_pow * (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2
                          + ((1 - beta) ** 4 / 1920) * log_FK ** 4)
    correction = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / FK_pow ** 2
                       + 0.25 * rho * beta * nu * alpha / FK_pow
                       + ((2 - 3 * rho ** 2) / 24) * nu ** 2) * T)
    return num / denom * correction
```

Plug into Black's formula (`engine.options.price` with the forward) to get prices.

## Calibration

Given an observed smile of $(K_i, \sigma_i)$ for a given $F, T$, calibrate $\alpha, \beta, \rho, \nu$ by minimising sum-squared error between Hagan IV and observed IV.

```python
from scipy.optimize import minimize


def fit_sabr_slice(F, T, strikes, iv_obs, beta=0.5):
    """Fit alpha, rho, nu with beta held fixed."""
    def loss(p):
        alpha, rho, nu = p
        if not (-0.999 < rho < 0.999):
            return 1e10
        fits = np.array([sabr_iv(F, k, T, alpha, beta, rho, nu) for k in strikes])
        return float(np.sum((fits - iv_obs) ** 2))
    x0 = [0.2, -0.3, 0.5]
    bounds = [(1e-4, 5.0), (-0.95, 0.95), (1e-4, 5.0)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return dict(zip(["alpha", "rho", "nu"], res.x.tolist()))
```

Convention: fix $\beta$ to a domain-typical value (0 for rates, 0.5 for rates pre-zero-bound, 1 for equity/FX log-normal) and fit the other three.

## Why SABR is so popular in rates / FX

1. **Dynamics**. Shock $\alpha$ → vol-of-vol moves; shock $\rho$ → skew rotates. The model has consistent risk responses across the surface.
2. **Smile shape**. SABR naturally produces smiles with positive vol-of-vol and skews with non-zero $\rho$. Matches FX and rates surfaces well.
3. **Sticky-strike vs sticky-delta**. The vol-vol-correlated dynamics give you a natural model for both behaviours — and you can tune $\rho$ to switch between them.

## A worked example: calibrating SABR to a synthetic smile

```python
import numpy as np

F, T = 100.0, 1.0
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
true_params = (0.25, 0.5, -0.3, 0.4)
iv_obs = np.array([sabr_iv(F, k, T, *true_params) for k in strikes])
# Add noise
rng = np.random.default_rng(0)
iv_obs += rng.normal(0, 0.001, len(strikes))

params = fit_sabr_slice(F, T, strikes, iv_obs, beta=0.5)
print(f"Fitted: {params}")
print(f"True:   alpha=0.25, rho=-0.3, nu=0.4")
```

In typical fits, alpha and nu recover within a few percent; rho is the slipperiest parameter (smile fit can be insensitive to small rho changes).

## SABR vs SVI

Both fit smiles. The differences:

| | SVI | SABR |
|---|---|---|
| Type | Static parametric fit | SDE → asymptotic IV |
| Parameters per slice | 5 (a, b, ρ, m, σ) | 4 (α, β, ρ, ν), often 3 with β fixed |
| Dynamics | None | Built-in |
| Wing behaviour | Asymptotically linear | Power-law growth via CEV |
| Industry use | Equity / index smiles | Rates / FX / commodities |

For an equity-only strategy, stick with SVI/SSVI. For multi-asset shops trading rates or FX, SABR is the lingua franca.

## Pitfalls

!!! warning "Hagan formula breaks at low strikes / high vol"
    The asymptotic expansion is in (α√T), (νT)... For very-out-of-the-money options or very long expiries, it can produce negative or non-monotone IVs. Modern alternatives (Antonov-Konikov-Spector "no-arb" SABR) are stable but slower.

!!! warning "Calibration's sensitivity to initial guess"
    SABR's loss surface has local minima. Multi-start optimisation (a few random initial guesses) is worth the small extra cost.

!!! warning "Beta isn't really free"
    Beta and alpha trade off heavily — many (β, α) combinations give the same fit. Always fix β to a sensible value for your market.

!!! warning "Sticky-strike vs sticky-moneyness with SABR"
    Real markets exhibit both. SABR's response to F-moves depends on β; you should calibrate to which regime you're in.

## Bottom line

For rates / FX / commodity smiles: SABR with $\beta$ fixed. Use the Hagan formula for IV; cross-check with no-arbitrage tests at the wings.

For equity smiles: prefer SVI / SSVI (previous chapter).

Continue to **[Heston — stochastic volatility with closed-form](03-heston.md)**.
