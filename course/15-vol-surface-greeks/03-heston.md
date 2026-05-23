# Heston — stochastic volatility with closed-form

The Heston model (1993) describes the underlying with stochastic variance:

$$
dS_t = \mu S_t \, dt + \sqrt{v_t} S_t \, dW^1_t
$$
$$
dv_t = \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW^2_t
$$
$$
\langle dW^1, dW^2 \rangle = \rho \, dt
$$

with five parameters: $\kappa$ (mean reversion), $\theta$ (long-run variance), $\sigma_v$ (vol-of-vol), $\rho$ (correlation), and $v_0$ (initial variance).

The big deal: Heston has a **semi-closed-form characteristic function**, which means you can price European options by Fourier inversion (Carr-Madan or COS method) much faster than Monte Carlo. That makes calibration to a whole vol surface tractable.

## The characteristic function

For $\phi_T(u) = \mathbb{E}[\exp(i u \log S_T)]$, Heston gives an explicit formula involving complex hyperbolic functions. The COS pricing method (Fang-Oosterlee 2008) then computes the call price as a truncated cosine-series expansion of $\phi_T$.

```python
import numpy as np

def heston_cf(u, T, S0, r, q, kappa, theta, sigma, rho, v0):
    """Heston characteristic function. u can be complex."""
    iu = 1j * u
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * iu - b) ** 2 + (sigma ** 2) * (iu + u ** 2))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)
    expDT = np.exp(-d * T)
    C = (r - q) * iu * T + (a / sigma ** 2) * (
        (b - rho * sigma * iu - d) * T - 2 * np.log((1 - g * expDT) / (1 - g))
    )
    D = ((b - rho * sigma * iu - d) / sigma ** 2) * (1 - expDT) / (1 - g * expDT)
    return np.exp(C + D * v0 + iu * np.log(S0))


def heston_price_cos(S0, K, T, r, q, kappa, theta, sigma, rho, v0, N=128, L=8.0):
    """Heston European call price via the COS method."""
    # Truncation range based on cumulants
    c1 = (r - q) * T
    c2 = T  # rough; full formula is in Fang-Oosterlee
    a, b = c1 - L * np.sqrt(c2), c1 + L * np.sqrt(c2)
    k = np.arange(N)
    u_k = k * np.pi / (b - a)

    cf = heston_cf(u_k, T, 1.0, r, q, kappa, theta, sigma, rho, v0)
    # Apply correction so the CF is for log(S/S0), not log(S0) + log(S/S0)
    cf_adj = cf * np.exp(-1j * u_k * np.log(S0))

    x = np.log(S0 / K)
    # V_k coefficients for a call payoff
    def U_k(k_arr, a, b):
        # Cosine series coefficients for max(e^y - 1, 0) on [a, b]
        chi = (1 / (1 + (k_arr * np.pi / (b - a)) ** 2)) * (
            np.cos(k_arr * np.pi * (b - a) / (b - a)) * np.exp(b)
            - np.cos(k_arr * np.pi * (0 - a) / (b - a))
            + (k_arr * np.pi / (b - a)) * np.sin(k_arr * np.pi * (b - a) / (b - a)) * np.exp(b)
            - (k_arr * np.pi / (b - a)) * np.sin(k_arr * np.pi * (0 - a) / (b - a))
        )
        psi = np.where(k_arr == 0,
                       b - 0,
                       (np.sin(k_arr * np.pi * (b - a) / (b - a)) - np.sin(k_arr * np.pi * (0 - a) / (b - a)))
                       * (b - a) / (k_arr * np.pi))
        return (2 / (b - a)) * (chi - psi)

    V_k = K * U_k(k, a, b)
    weights = np.ones(N); weights[0] = 0.5

    integrand = weights * cf_adj * np.exp(1j * u_k * (x - a))
    return float(np.exp(-r * T) * np.real(integrand @ V_k))
```

The implementation above is a teaching-level reference. Production Heston pricers (QuantLib, in-house C++) are 100-1000× faster via vectorisation and proper FFT.

## Calibration

Given a full market surface $\sigma_{ij}^\text{mkt}(K_i, T_j)$, fit Heston parameters by minimising squared error:

```python
from scipy.optimize import differential_evolution
from engine.options import implied_vol

def calibrate_heston(market_iv: dict, S0: float, r: float, q: float):
    """market_iv: dict mapping (K, T) → observed IV."""
    def loss(params):
        kappa, theta, sigma, rho, v0 = params
        err = 0.0
        for (K, T), iv_mkt in market_iv.items():
            price_heston = heston_price_cos(S0, K, T, r, q, kappa, theta, sigma, rho, v0)
            iv_model = implied_vol(price_heston, S0, K, T, r, q, kind="call")
            if not np.isfinite(iv_model):
                err += 1.0
                continue
            err += (iv_model - iv_mkt) ** 2
        return err

    bounds = [(0.1, 10), (0.001, 1.0), (0.01, 5.0), (-0.99, 0.99), (0.001, 1.0)]
    res = differential_evolution(loss, bounds, seed=0, maxiter=100, polish=True)
    return dict(zip(["kappa", "theta", "sigma", "rho", "v0"], res.x.tolist()))
```

Differential evolution is global; the loss surface is rugged enough that local minimisers (L-BFGS-B) get stuck.

## When Heston wins

- **You need a *dynamic* model**, not just a static surface fit. Heston gives you Vega, vanna, vomma consistent with the model.
- **You're pricing exotics** (barrier, lookback, cliquet) under a stochastic-vol assumption.
- **You're constructing a hedging strategy** that accounts for vol-of-vol.

For pure smile fitting on a snapshot, SVI/SABR is usually faster and equally accurate. Heston is the right choice when you need to *price* the model's behaviour going forward.

## Neural-network surrogate

Heston calibration is slow (each loss evaluation involves Fourier integration per strike). A common trick: train a neural network to map (Heston params, K, T) → IV in offline batches, then use the NN as the pricing function in the calibration loop. Speedup: ~100-1000×.

```python
# Schematic
def train_heston_surrogate(n_samples=100_000):
    rng = np.random.default_rng(0)
    X = rng.uniform([0.1, 0.01, 0.01, -0.99, 0.01, 0.7, 0.02],
                    [10, 1.0, 5.0, 0.99, 1.0, 1.5, 5.0],
                    size=(n_samples, 7))
    # Compute IVs for each sample (slow; do once)
    y = np.array([compute_heston_iv(*row) for row in X])
    # Train a small MLP
    model = some_nn().fit(X, y)
    return model
```

The first batch is slow; every subsequent calibration is sub-second.

## A worked example: synthetic calibration round-trip

```python
# Pretend we have observed prices generated from known Heston params
true = dict(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
S0, r, q = 100.0, 0.04, 0.0
strikes = [80, 90, 100, 110, 120]
expiries = [0.25, 0.5, 1.0, 2.0]
market_iv = {}
for T in expiries:
    for K in strikes:
        p = heston_price_cos(S0, K, T, r, q, **true)
        iv = implied_vol(p, S0, K, T, r, q, kind="call")
        if np.isfinite(iv):
            market_iv[(K, T)] = iv

recovered = calibrate_heston(market_iv, S0, r, q)
print("True:     ", true)
print("Recovered:", recovered)
```

Recovery is typically within 5-10% of true on the four shape parameters (kappa, theta, sigma, rho) for synthetic data; v0 is the easiest to recover. Real market data has noise that limits recovery further.

## Feller condition

For variance to stay positive, $2 \kappa \theta > \sigma^2$ (the Feller condition). If your calibration violates this, the underlying SDE can hit zero variance, which is a model failure mode. Most production calibrators enforce Feller as a constraint.

## Pitfalls

!!! warning "COS truncation"
    The COS method depends on a truncation range [a, b]. Too narrow → bias. Too wide → wasted computation. L = 8 standard deviations is usually safe.

!!! warning "Calibration to noisy quotes"
    Vendor IVs include stale quotes. Down-weight illiquid strikes; pre-filter for bid-ask spread.

!!! warning "Heston greeks are not Black-Scholes greeks"
    Vega under Heston is different from vega under Black-Scholes — Heston's vega includes the model's correlation and mean-reversion responses. Computing the right greek for hedging requires the right pricer.

!!! warning "The model can't fit short-dated smiles well"
    SPY's 7-day smile is extreme; Heston's smile generated by GBM-of-vol can't produce sharp enough wings. For very short-dated options, jumps (Bates model) or local-stochastic-vol are needed.

## Bottom line

Heston is the right tool when:

- You need a **dynamic** stochastic-vol model (not just smile fitting).
- You're **pricing exotics** with stochastic-vol risk.
- You're **building risk-sensitivity tools** consistent with stochastic vol.

Calibrate with differential evolution or use a neural surrogate for speed. Always check the Feller condition. For pure smile fitting, SVI/SABR is simpler and faster.

Continue to **[Local volatility via Dupire](04-local-vol.md)**.
