# Realised kernels and pre-averaging — vol from noisy ticks

If you have tick-level price data, you can compute **realised volatility** much more accurately than from daily bars. But the naive estimator (sum of squared tick returns) is **biased** because of market microstructure noise — bid-ask bounce, asynchronous trades, etc. The fix: realised kernels (Barndorff-Nielsen et al.) or pre-averaging (Jacod et al.). These are standard tools in modern high-frequency econometrics.

## The naive estimator and its bias

For tick returns $r_i$ in a time interval $[0, T]$:

$$
\widehat{RV}_\text{naive} = \sum_i r_i^2
$$

In the absence of noise, this converges to the integrated variance as the sampling frequency increases. With microstructure noise (each observed price = true price + iid noise), the estimator's bias **grows without bound** as you sample more frequently. The "signature plot" of estimated RV against sampling frequency shows the bias.

## The realised kernel

Barndorff-Nielsen, Hansen, Lunde, Shephard (2008, 2011). Instead of just summing squared returns, take a weighted average of *autocovariances* up to some lag $H$:

$$
\widehat{RV}_\text{kernel} = \gamma_0 + \sum_{h=1}^H k\!\left(\frac{h-1}{H}\right) (\gamma_h + \gamma_{-h})
$$

where $\gamma_h = \sum_i r_i r_{i+h}$ and $k$ is a kernel weight function (Parzen, Tukey-Hanning, modified Bartlett...).

The autocovariance terms **cancel out the microstructure noise's autocorrelation**, leaving an unbiased estimate of integrated variance.

```python
import numpy as np


def parzen_kernel(x: np.ndarray) -> np.ndarray:
    """Parzen kernel — standard choice for realised kernels."""
    x = np.abs(x)
    return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3,
            np.where(x <= 1.0, 2 * (1 - x) ** 3, 0))


def realised_kernel(returns: np.ndarray, H: int = 20, kernel: callable = parzen_kernel) -> float:
    """Realised kernel estimator of integrated variance."""
    n = len(returns)
    gamma_0 = np.sum(returns ** 2)
    rk = gamma_0
    for h in range(1, H + 1):
        weight = kernel((h - 1) / H)
        gamma_h = np.sum(returns[:-h] * returns[h:])
        rk += 2 * weight * gamma_h
    return float(rk)


# Naive vs kernel
ticks_per_day = 20_000     # ~e.g. SPY at 1-second resolution
returns = np.random.normal(0, 0.0001, ticks_per_day)
print(f"Naive RV: {np.sum(returns ** 2):.6f}")
print(f"Kernel RV: {realised_kernel(returns, H=20):.6f}")
```

For a clean (no-noise) simulation, both give the same answer. For real data with noise, kernel RV is much closer to true integrated variance.

## Choice of H

The bandwidth $H$ controls the bias-variance trade-off:

- Small $H$ → less variance, more bias (noise not fully cancelled).
- Large $H$ → less bias, more variance (averaging over many noisy lags).

BNHLS suggest $H \approx c \cdot n^{1/2}$ for $n$ observations; $c$ is typically 1-5 depending on the asset.

## Pre-averaging

Jacod, Li, Mykland, Podolskij, Vetter (2009). Alternative noise-correction approach: at each window of $k_n$ ticks, compute an average return; then sum the squared averaged returns.

$$
\widehat{RV}_\text{pre-avg} = \frac{1}{\psi_2 k_n} \sum_i (\bar r_i)^2 - \frac{\psi_1}{2 \psi_2 k_n^2} \sum_i r_i^2
$$

where $\bar r_i$ is the average return over the $i$-th window, $\psi_1, \psi_2$ are constants depending on the averaging weights, and the second term is a noise-correction.

```python
def pre_averaged_returns(returns: np.ndarray, k_n: int) -> np.ndarray:
    """Pre-average returns over windows of k_n ticks."""
    n = len(returns)
    out = []
    for i in range(0, n - k_n + 1, k_n // 2):       # half-overlap
        window = returns[i:i + k_n]
        # Use a triangular weight
        weights = 1 - np.abs(np.linspace(-1, 1, k_n))
        out.append(np.sum(window * weights) / np.sum(weights))
    return np.array(out)


def pre_avg_rv(returns: np.ndarray, k_n: int = 50) -> float:
    """Pre-averaged realised variance with noise correction."""
    pre_avg = pre_averaged_returns(returns, k_n)
    naive_term = np.sum(returns ** 2)
    pre_avg_term = np.sum(pre_avg ** 2)
    psi_1, psi_2 = 1.0, 1/12         # for triangular weights (approximation)
    return pre_avg_term / (psi_2 * k_n) - psi_1 / (2 * psi_2 * k_n ** 2) * naive_term
```

Both methods are theoretically equivalent at the limit. In practice, choose by what's easier to implement and tune for your asset class.

## A worked example: comparing estimators on simulated noisy ticks

```python
import numpy as np


def simulate_noisy_ticks(true_iv: float, n_ticks: int, noise_std: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    # True log-price path (Brownian)
    dt = 1 / n_ticks
    true_log_price = np.cumsum(rng.normal(0, np.sqrt(true_iv * dt), n_ticks))
    # Add iid noise to each tick
    observed_log_price = true_log_price + rng.normal(0, noise_std, n_ticks)
    returns = np.diff(observed_log_price)
    return returns


true_iv = 0.04                       # 20% annual vol → 0.04 daily variance
returns = simulate_noisy_ticks(true_iv, n_ticks=20_000, noise_std=0.0001)

naive = np.sum(returns ** 2)
kernel = realised_kernel(returns, H=int(np.sqrt(len(returns)) * 3))
pre_avg = pre_avg_rv(returns, k_n=int(np.sqrt(len(returns))))

print(f"True IV: {true_iv:.6f}")
print(f"Naive: {naive:.6f}   ({naive / true_iv - 1:+.0%} bias)")
print(f"Kernel: {kernel:.6f}   ({kernel / true_iv - 1:+.0%} bias)")
print(f"Pre-avg: {pre_avg:.6f}   ({pre_avg / true_iv - 1:+.0%} bias)")
```

Typical: naive has +50-200% bias; kernel and pre-avg are within ±10%.

## Multivariate realised covariance

The kernel approach generalises to covariance: compute the same kernel-weighted sum of cross-asset returns. The result is an unbiased estimator of the integrated covariance matrix.

For portfolio risk on intraday data, this is much sharper than daily-bar sample covariance.

## Where this matters in trading

- **High-frequency vol estimation** for sizing — replace daily-bar EWMA with intraday realised kernel.
- **Vol-of-vol modelling** — daily series of kernel RV is much less noisy than daily-bar squared returns.
- **Jump detection** — separate continuous IV from jump variation by combining kernel RV with bipower variation.
- **Realised correlation** for intraday portfolio risk.

## Pitfalls

!!! warning "Bandwidth choice matters"
    Both $H$ (kernel) and $k_n$ (pre-avg) are critical. Bad choices give huge bias or variance. Validate empirically on data where you know the answer.

!!! warning "Trading hours and overnight gaps"
    Realised vol from intraday ticks captures intraday vol only. Overnight gaps (between close and next-day open) are missed. For total vol, add the overnight return squared (or model separately).

!!! warning "Microstructure regime changes**
    Noise patterns change — e.g., during market stress, the bid-ask bounce gets bigger. Re-calibrate the bandwidth periodically.

!!! warning "Asynchronous multi-asset data"
    Two assets don't trade at the same instants. Naive sum of cross-products is biased ("Epps effect"). Use the Hayashi-Yoshida estimator for proper multivariate realised covariance.

## Bottom line

Realised kernels and pre-averaging are the right tools for **noise-robust vol estimation from tick data**. The naive estimator is wrong by an order of magnitude on real data. For any HFT or intraday-vol strategy, these methods are mandatory.

Continue to **[Avellaneda-Stoikov market making in detail](04-avellaneda-stoikov.md)**.
