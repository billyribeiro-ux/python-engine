# Copulas and tail dependence

The correlation coefficient is one of the most over-used and least-understood statistics in finance. It tells you about *linear* dependence between two variables when both are Gaussian. The moment either condition breaks — and they almost always do — it lies.

Copulas are the right framework for joint distributions when the linear-Gaussian assumption fails. This chapter is the working-engineer introduction.

## The setup

Sklar's theorem (1959): any joint distribution $F(x, y)$ can be written as

$$
F(x, y) = C(F_X(x), F_Y(y))
$$

where $F_X, F_Y$ are the marginal CDFs and $C$ is a **copula** — a joint CDF on the unit square with uniform marginals.

The interpretation: *separate* the marginal distributions (the shape of each variable alone) from the *dependence structure* (how the variables move together, expressed in their ranks).

This separation is enormously useful because:

1. You can model the marginals however you want (a t-distribution for fat tails, a kernel density, anything).
2. You can plug in any copula to model the dependence (Gaussian, Student-t, Clayton, Gumbel, ...).
3. The two choices are independent.

## A quick numerical demo

```python
import numpy as np
from scipy.stats import norm, t

rng = np.random.default_rng(0)

# Two t-distributed marginals with the SAME Pearson correlation,
# but different DEPENDENCE structures.

# Gaussian copula
u = norm.cdf(rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=10_000))
x_gauss = t.ppf(u, df=4)         # transform back to t marginals

# Student-t copula with low df → strong tail dependence
z = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=10_000)
chi2 = rng.chisquare(df=3, size=10_000)
z = z * np.sqrt(3 / chi2)[:, None]
u_t = t.cdf(z, df=3)
x_t = t.ppf(u_t, df=4)

print(f"Pearson correlation (gauss): {np.corrcoef(x_gauss.T)[0,1]:.2f}")
print(f"Pearson correlation (t):     {np.corrcoef(x_t.T)[0,1]:.2f}")
```

Both samples have nearly identical Pearson correlation. But if you look at *extreme* events — both variables in the bottom 5% — the second sample has dramatically more co-occurrences. That's **tail dependence**, and the Gaussian copula has none of it. In a crisis, everything collapses together; the Gaussian copula systematically under-prices this.

## Tail dependence, numerically

```python
def lower_tail_dep(x: np.ndarray, q: float = 0.05) -> float:
    """P(Y < y_q | X < x_q) where x_q, y_q are the q-quantiles."""
    x_q = np.quantile(x[:, 0], q)
    y_q = np.quantile(x[:, 1], q)
    cond = (x[:, 0] < x_q) & (x[:, 1] < y_q)
    return cond.sum() / (x[:, 0] < x_q).sum()

print(f"Lower tail dep (gauss): {lower_tail_dep(x_gauss):.2f}")
print(f"Lower tail dep (t):     {lower_tail_dep(x_t):.2f}")
```

Typical output: Gaussian ~0.10–0.15, t-copula ~0.30–0.40 with df=3. Same correlation, very different tails.

## The copulas you'll use

| Copula | Tail dependence | When to use |
|---|---|---|
| **Gaussian** | None | benchmark; **don't** use it for risk |
| **Student-t** | Symmetric tails | the workhorse for asset returns |
| **Clayton** | Lower tail only | downside contagion, defaults |
| **Gumbel** | Upper tail only | joint extremes on the upside |
| **Frank** | None, but asymmetric | when you need asymmetry without tails |

For multivariate equity returns: **Student-t copula** with the marginals modelled separately (often as t-distributions themselves, possibly with skew). For credit / default modelling, Clayton.

## Fitting a Student-t copula

```python
from copulas.multivariate import GaussianMultivariate, VineCopula
# Caveat: the `copulas` package's APIs evolve; check the docs.

# Or fit manually:
# 1. Compute pseudo-observations: rank-based uniforms
# 2. Map to a multivariate normal via inverse CDF of standard normal
# 3. Estimate the covariance + degrees of freedom by MLE

import numpy as np
from scipy.stats import rankdata, norm

def pseudo_obs(X: np.ndarray) -> np.ndarray:
    """Convert each column to ranks/(n+1), giving (0,1) uniform pseudo-obs."""
    n = X.shape[0]
    return rankdata(X, axis=0) / (n + 1)

def fit_gaussian_copula(X: np.ndarray):
    """Returns the copula correlation matrix."""
    U = pseudo_obs(X)
    Z = norm.ppf(U)
    return np.corrcoef(Z, rowvar=False)
```

For the Student-t copula, swap `norm.ppf` for the t inverse CDF and fit `df` by MLE. The `copulas`, `pyvinecopulib`, and `copulae` packages do this cleanly.

## A worked example: pairs strategy risk

You hold a long-short pair (long A, short B) with a beta-1 hedge. You want a 95% one-day VaR. Two approaches:

```python
# Naive: assume normal returns, use historical mean/std of pair P&L
sigma = pair_returns.std()
var_naive = -1.645 * sigma * portfolio_size

# Copula-based: model marginals as t, dependence as t-copula
fit_marginal_a = t.fit(returns_a)
fit_marginal_b = t.fit(returns_b)
copula = fit_t_copula(np.column_stack([returns_a, returns_b]))

# Sample 100k joint draws, transform to original marginals, compute pair P&L
samples_u = sample_from_t_copula(copula, n=100_000)
samples_a = t.ppf(samples_u[:, 0], *fit_marginal_a)
samples_b = t.ppf(samples_u[:, 1], *fit_marginal_b)
pair_pnl = samples_a - samples_b
var_copula = -np.percentile(pair_pnl, 5) * portfolio_size
```

For a pair like SPY-QQQ during normal regimes the two estimates agree. During crises (when tail dependence spikes), the copula-based VaR is *substantially* larger — and substantially more honest. The naive VaR is the one that gets blown through repeatedly in 2008/2020-style episodes.

## Vine copulas — for many dimensions

A single multivariate copula scales poorly. **Vine copulas** decompose a high-dimensional dependence structure into a cascade of pairwise copulas. For modelling 10–100 assets jointly with rich tail structure, vines are the standard.

The `pyvinecopulib` package is the production tool. The math is in Joe (2014); for working purposes:

```python
import pyvinecopulib as pv

data = your_returns_matrix         # shape (T, N), each column a return series
u = pseudo_obs(data)                # uniform marginals
controls = pv.FitControlsVinecop(family_set=[pv.gaussian, pv.student, pv.clayton])
vine = pv.Vinecop(data=u, controls=controls)

# Simulate joint draws
sim_u = vine.simulate(n=10_000, seeds=[1])
sim_real = np.column_stack([
    your_marginal[i].ppf(sim_u[:, i]) for i in range(N)
])
```

The vine picks the best pairwise copula family for each edge. For portfolio stress-testing across many assets, this is the closest you'll get to a believable joint generator.

## Pitfalls

!!! warning "Don't fit copulas to non-stationary data"
    Copulas assume the joint distribution doesn't change. Returns are roughly stationary on short windows; over years they drift. Fit on rolling windows, not the whole history.

!!! warning "Goodness-of-fit is hard"
    Standard $\chi^2$ tests don't apply directly. The literature uses bootstrap-based GoF tests (Genest et al). For practical work, **test out-of-sample** — does the copula give honest tails on a holdout?

!!! warning "Tail dependence ≠ Pearson correlation"
    A Student-t copula and a Gaussian copula can have the same correlation matrix. The tail story is *only* in the copula choice. Don't mistake matching correlations for matching joints.

## Bottom line

For trading risk:

- Default copula: **Student-t**, with degrees of freedom 4–8 for daily equity returns.
- Use Pearson correlation for *reporting*, but **never** for tail estimates.
- For >5 dimensions, use **vine copulas** (`pyvinecopulib`).
- Always validate out-of-sample — does the model put the right amount of mass in the joint tails?

Continue to **[Extreme value theory](05-evt.md)**.
