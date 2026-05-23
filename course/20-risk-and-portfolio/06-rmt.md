# Random matrix theory for correlation cleaning

The sample correlation matrix from finite-sample financial data has **eigenvalues that don't all correspond to real signal**. Most of them are noise — driven by the random fluctuations in the sample. Random Matrix Theory (RMT) gives you a clean way to separate the noise eigenvalues from the signal ones, and to "clean" the correlation matrix by replacing the noise with their average.

For portfolio construction on many assets (N > 50), RMT cleaning often produces dramatically more stable allocations.

## The Marchenko-Pastur distribution

For a correlation matrix estimated from $T$ observations of $N$ time series, with $q = N / T$, the eigenvalues of a *random* correlation matrix follow the Marchenko-Pastur distribution:

$$
f(\lambda) = \frac{1}{2\pi \sigma^2} \frac{\sqrt{(\lambda_\max - \lambda)(\lambda - \lambda_\min)}}{q \lambda} \quad \text{for } \lambda_\min \leq \lambda \leq \lambda_\max
$$

with edges:

$$
\lambda_\max = \sigma^2 (1 + \sqrt{q})^2, \quad \lambda_\min = \sigma^2 (1 - \sqrt{q})^2
$$

where $\sigma^2 = 1 - $ (the variance attributable to genuine signal).

The implication: **eigenvalues within $[\lambda_\min, \lambda_\max]$ are statistically consistent with noise**. Eigenvalues *outside* this range carry real signal.

## The cleaning procedure

1. Compute the sample correlation matrix $C$.
2. Eigendecompose: $C = V \Lambda V^\top$.
3. Find the eigenvalues that lie outside the Marchenko-Pastur edges → "signal" eigenvalues.
4. Replace the "noise" eigenvalues with their average.
5. Reconstruct: $C_\text{cleaned} = V \Lambda_\text{cleaned} V^\top$.

```python
import numpy as np
import pandas as pd


def mp_edges(q: float, sigma2: float = 1.0):
    return (sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2)


def rmt_clean_correlation(returns: pd.DataFrame, market_factor: bool = True) -> np.ndarray:
    """Clean a sample correlation matrix using RMT."""
    T, N = returns.shape
    q = N / T
    corr = returns.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    # Estimate noise level from the bulk
    lambda_min, lambda_max = mp_edges(q)
    noise_mask = (eigvals >= lambda_min) & (eigvals <= lambda_max)
    noise_avg = float(eigvals[noise_mask].mean()) if noise_mask.any() else 1.0
    # The largest eigenvalue often corresponds to the market factor — keep it
    cleaned_eigvals = np.where(noise_mask, noise_avg, eigvals)
    cleaned = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)
    return cleaned
```

For $N = 100$ assets with $T = 252$ daily observations ($q = 100/252 \approx 0.4$), the MP edges are about $[0.13, 2.59]$. Eigenvalues from the actual correlation matrix that fall in this range are replaced with their mean.

## What signals survive

For US equity panels, typically:

- **The largest eigenvalue** is far above $\lambda_\max$ — this is the **market factor** (everything is positively correlated to the market).
- **A few more eigenvalues** (3-10) above $\lambda_\max$ — sector / style factors.
- **The bulk** of eigenvalues in $[\lambda_\min, \lambda_\max]$ — noise.
- **A few below $\lambda_\min$** — anti-correlated factors (rare, often suspicious).

After cleaning, the correlation matrix retains the market + sector / style structure and discards the noise.

## A worked example

```python
import numpy as np
import pandas as pd

# Simulate: 5 latent factors + noise
rng = np.random.default_rng(0)
n_assets = 50
n_days = 252
true_loadings = rng.normal(0, 1, (n_assets, 5))
factor_returns = rng.normal(0, 1, (n_days, 5))
idiosyncratic = rng.normal(0, 1, (n_days, n_assets))
returns_df = pd.DataFrame(factor_returns @ true_loadings.T + idiosyncratic)

sample_corr = returns_df.corr().values
cleaned_corr = rmt_clean_correlation(returns_df)

# Compare frobenius norms
print(f"Sample corr Frobenius: {np.linalg.norm(sample_corr, 'fro'):.2f}")
print(f"Cleaned corr Frobenius: {np.linalg.norm(cleaned_corr, 'fro'):.2f}")

# Eigenvalue spectrum
ev_sample = np.linalg.eigvalsh(sample_corr)
ev_cleaned = np.linalg.eigvalsh(cleaned_corr)
print(f"Sample top 6 eigenvalues: {ev_sample[-6:]}")
print(f"Cleaned top 6 eigenvalues: {ev_cleaned[-6:]}")
```

Top eigenvalues match (those are the real factors). The bulk eigenvalues in `cleaned_corr` should all equal their average.

## Using the cleaned matrix

Plug cleaned correlation (converted to covariance via $\sigma_i \sigma_j C_{ij}$) into any portfolio construction algorithm — mean-variance, HRP, NCO. The cleaned version produces:

- More stable weights across time (less sensitivity to noise).
- Lower out-of-sample realised variance for the same target.
- Less estimation error in inverse-covariance methods.

```python
def cleaned_covariance(returns_df: pd.DataFrame) -> np.ndarray:
    cleaned_corr = rmt_clean_correlation(returns_df)
    std = returns_df.std().values
    return cleaned_corr * np.outer(std, std)
```

## Comparison with Ledoit-Wolf shrinkage

Both RMT cleaning and Ledoit-Wolf are responses to the noisy-covariance problem. The differences:

- **Ledoit-Wolf** shrinks all eigenvalues toward the average. Simple, closed-form.
- **RMT** keeps the signal eigenvalues exactly and replaces only the noise. More targeted.

Empirically, they perform similarly for most use cases. RMT is more interpretable (you can see which eigenvalues were "denoised"). LW is faster and easier to implement.

For production: Ledoit-Wolf as the default; RMT when you want explicit factor-vs-noise separation.

## Rotational invariance

The RMT cleaning is **rotationally invariant** — it doesn't depend on the ordering of assets. The Ledoit-Wolf shrinkage is too. HRP and clustering-based methods, by contrast, are **not** rotationally invariant — they depend on the cluster structure.

In production, this means RMT / LW are deterministic given the data. HRP / NCO have some non-determinism via the clustering choice.

## Pitfalls

!!! warning "Noise variance estimation"
    The formula above assumes $\sigma^2 = 1 - $ (signal fraction). In practice, fit it by looking at the empirical distribution of bulk eigenvalues. Iterative methods (Laloux et al. 1999) give better estimates.

!!! warning "Not all signal is above the MP top edge"
    Some real factors have eigenvalues *inside* the MP bulk (smaller than $\lambda_\max$). The cleaning will incorrectly replace these. For most equity panels this is rare; for sector-narrow portfolios, beware.

!!! warning "Stationary correlation assumption**
    RMT cleaning is computed once on a window of data. If correlations change between training and deployment, the cleaning is for the past, not the present.

!!! warning "Time-varying volatility ignored**
    Standardising returns by volatility before computing the correlation matrix improves estimates. For very heteroskedastic data, do this explicitly.

## Bottom line

Random matrix theory cleaning is the right tool when:

- You have **many assets** ($N > 50$).
- Your **estimation window is short** ($T \approx N$ or smaller).
- You're feeding the covariance into a **matrix-inversion-based** optimiser (mean-variance, multivariate Kelly).

For HRP-based portfolios, the cleaning matters less (HRP doesn't invert). For mean-variance, it makes a big difference.

## End of Module 20

You now have the institutional risk and portfolio toolkit: drawdown discipline, honest VaR/CVaR, fractional Kelly, multi-layer vol targeting, HRP/NCO, and RMT cleaning. The next and final module — **Deployment** — covers what happens when the strategy goes live.

Continue to **[Module 21 — Deployment](../21-deployment/index.md)**.
