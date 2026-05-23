# Factor models — building blocks for everything

A factor model decomposes the return of an asset into the *systematic* part (driven by a small set of common factors) and the *idiosyncratic* part (specific to the asset). This decomposition is not just academic — it is the lens through which every serious quant looks at returns. Once you internalise it, half of "alpha generation" becomes "finding factors with positive premia that you can harvest cheaply."

## The picture

For an asset $i$ at time $t$:

$$
r_{i,t} = \alpha_i + \beta_{i,1} f_{1,t} + \beta_{i,2} f_{2,t} + \dots + \beta_{i,K} f_{K,t} + \varepsilon_{i,t}
$$

- $f_{k,t}$: the $k$-th factor return at time $t$.
- $\beta_{i,k}$: the loading of asset $i$ on factor $k$.
- $\alpha_i$: the asset's mean return after the factors are accounted for. **This is the "alpha".**
- $\varepsilon_{i,t}$: the idiosyncratic shock.

The point: *what looks like alpha in a single-asset analysis is usually a loading on a known factor*. If your "edge" is actually a momentum tilt, the right benchmark is a momentum factor, not zero.

## The Fama-French 3-factor model

The original three factors:

- **Market (MKT)** — the cap-weighted market return minus the risk-free rate.
- **Small Minus Big (SMB)** — small-cap return minus large-cap return.
- **High Minus Low (HML)** — value (high book-to-market) return minus growth.

In code, given a returns matrix `R` (T × N) and a factor matrix `F` (T × K):

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def factor_regress(R: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Time-series regression of each asset return on factors. Returns alphas + betas."""
    X = sm.add_constant(F)               # adds the intercept (= alpha)
    rows = []
    for name in R.columns:
        y = R[name].dropna()
        x = X.loc[y.index]
        res = sm.OLS(y, x).fit()
        rows.append({"asset": name, **dict(zip(["alpha"] + list(F.columns), res.params)),
                     "t_alpha": res.tvalues["const"], "r2": res.rsquared})
    return pd.DataFrame(rows).set_index("asset")
```

For Fama-French factor data: French's data library (downloadable as CSV), or via `pandas_datareader.famafrench`.

## Fama-MacBeth — cross-sectional factor returns

The other direction: at each time $t$, regress *cross-sectional* returns on $t-1$ factor loadings. The resulting "factor returns" are estimates of the per-period premium for each factor.

```python
def fama_macbeth(R: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    R: T x N returns at time t.
    B: T x N x K loadings as of time t-1 (known by t).
    Returns T x K time series of factor returns (Newey-West-able).
    """
    T = R.shape[0]
    Ks = []
    for t in range(T):
        x = sm.add_constant(B.iloc[t])
        y = R.iloc[t].dropna()
        x = x.loc[y.index]
        Ks.append(sm.OLS(y, x).fit().params)
    return pd.DataFrame(Ks, index=R.index)
```

The means of those columns are the **factor premia**; their t-stats (with Newey-West correction) test whether they're real.

## The modern factor zoo

Beyond the original three, the literature has piled on dozens of factors:

| Factor | Definition (rough) |
|---|---|
| MOM (momentum) | past 12-month return minus past 1-month |
| RMW (profitability) | high gross profit / total assets minus low |
| CMA (investment) | low asset growth minus high |
| QMJ (quality) | combined profitability + safety + growth |
| BAB (betting-against-beta) | low-beta minus high-beta, levered |
| LIQ (liquidity) | illiquid minus liquid |
| LowVol | low realised vol minus high |

The honest fact is that most factor returns *outside of the original five* are very fragile out-of-sample (the "factor zoo" problem; see Harvey, Liu, Zhu 2016). Treat with healthy skepticism. The robust ones for US equities, by most analyses: **MKT, MOM, QMJ, LowVol**.

## Computing your own factor

To build a "value" factor for US equities from scratch:

1. Universe — current S&P 500 (or your investable universe, point-in-time, see Module 5).
2. Score — book-to-market ratio as of $t-3$ months (lag for reporting).
3. Construct long and short legs — top 30% minus bottom 30% of the score.
4. Re-balance monthly.
5. Compute the daily return of the long-short portfolio.

```python
def long_short_factor(scores: pd.DataFrame, returns: pd.DataFrame, q: float = 0.3) -> pd.Series:
    """
    scores: T x N point-in-time scores (e.g. book-to-market). NaN where the asset is
      outside the universe at time t.
    returns: T x N realized daily returns.
    """
    long_threshold  = scores.quantile(1 - q, axis=1)
    short_threshold = scores.quantile(q,     axis=1)
    long_mask  = scores.ge(long_threshold,  axis=0)
    short_mask = scores.le(short_threshold, axis=0)
    n_long  = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)
    long_ret  = (returns * long_mask).sum(axis=1) / n_long.replace(0, np.nan)
    short_ret = (returns * short_mask).sum(axis=1) / n_short.replace(0, np.nan)
    return (long_ret - short_ret).rename("factor")
```

Now `factor_regress(my_strategy_returns, F=pd.concat([market_excess, my_factor], axis=1))` tells you whether your "strategy" is just a loading on the factor or whether there's something more.

## The risk-decomposition view

For a portfolio with weights $w$:

$$
\sigma_p^2 = w^\top (B \Sigma_F B^\top + D) w
$$

where $B$ is the asset-by-factor loading matrix, $\Sigma_F$ the factor covariance, and $D$ the diagonal idiosyncratic variance.

Splitting risk into **factor contributions** lets you ask: is your strategy's volatility coming from a value tilt? a size tilt? actual stock-specific bets? For risk reporting and capacity analysis this decomposition is gold.

## Pitfalls

!!! warning "Selection bias in factor construction"
    Factor returns reported in papers are usually computed on *today's* data — survivorship-included. Real-world factor returns are 1-2% lower per year than the paper claims.

!!! warning "Factor mining"
    The literature has reported >300 "factors". The probability that any single one survives strict out-of-sample testing is roughly equal to the probability you'd reject the null *by chance* at p=0.001 in 300 trials. Treat new factors as guilty until proven innocent.

!!! warning "Confusing alpha with risk-premia"
    A high alpha against a small factor model isn't necessarily real edge. It might be a loading on a factor you didn't include. Iterate the model.

## Bottom line

For real work:

- **Always benchmark a strategy's returns against an honest factor model** — at minimum Fama-French 3 + MOM. If the alpha isn't significant against this, you don't have alpha; you have a factor tilt.
- **Compute factor loadings on rolling windows** — they change. A Kalman filter (Module 7) gives the cleanest time-varying betas.
- **For sizing, use the risk decomposition** — factor and idiosyncratic risk are different beasts.

Continue to **[Momentum and mean reversion](02-momentum-meanrev.md)**.
