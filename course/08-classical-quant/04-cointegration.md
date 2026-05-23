# Cointegration and pairs trading

Two random walks have no business being related, yet some pairs of stocks track each other so closely that their *difference* is stationary. Cointegration is the formal name for this phenomenon, and pairs trading is the strategy that monetises it.

This chapter ties together a lot of the course so far — the Kalman filter (Module 7), the OU process (last chapter), proper stationarity tests (Module 7), and the data adapter (Module 0) — into one complete, runnable pairs trading strategy.

## What cointegration means

Two non-stationary series $X_t, Y_t$ are **cointegrated** if there exists a constant $\beta$ such that

$$
Z_t = Y_t - \beta X_t
$$

is stationary. Then $Z_t$ is the **spread**, which mean-reverts to a (potentially time-varying) level, and $\beta$ is the **hedge ratio**.

The economic story: if X and Y are driven by a common factor (same sector, similar business, shared customers), the relationship persists. Idiosyncratic shocks pull them apart; over time, the common factor pulls them back.

## Engle-Granger — the textbook test

Two-step:

1. Regress $Y$ on $X$ to get $\hat\beta$.
2. Test the residual $\hat Z_t = Y_t - \hat\beta X_t$ for stationarity using ADF.

```python
from statsmodels.tsa.stattools import coint

stat, pvalue, _ = coint(y_prices, x_prices)
print(f"EG cointegration p-value: {pvalue:.4f}")
```

Reject the null → cointegrated. The test is asymmetric in the choice of dependent variable; flip $X$ and $Y$ and you'll get a slightly different p-value. In practice, take the more conservative (larger) of the two p-values.

## Johansen — for portfolios of more than two

When you have $k > 2$ candidate assets, the cointegrating vector is multidimensional. Johansen's test gives you both the *number* of cointegrating relationships and the vectors themselves.

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# prices: T x k DataFrame
result = coint_johansen(prices, det_order=0, k_ar_diff=1)
print(result.lr1)             # trace statistics
print(result.cvt[:, 1])       # 95% critical values
# The largest eigenvalue's eigenvector is the most-mean-reverting linear combination
beta = result.evec[:, 0]
spread = prices @ beta
```

For a portfolio of 4–10 ETFs in the same sector (XLF + KBE + KBWR + IAT, for example), Johansen typically finds 1–3 cointegrating relationships. The first eigenvector is the most stable.

## Computing a time-varying hedge ratio with the Kalman filter

OLS gives a single $\hat\beta$. Real hedge ratios drift — the two stocks' beta to the market changes, the relationship shifts. The Kalman filter from Module 7, chapter 3 estimates a slowly-varying $\beta_t$ online:

```python
import numpy as np
import pandas as pd

class KalmanPairs:
    def __init__(self, delta: float = 1e-4, R: float = 1.0):
        self.x = np.zeros((2, 1))                   # [intercept, slope]
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * (delta / (1 - delta))
        self.R = np.array([[R]])

    def update(self, a: float, b: float) -> tuple[float, float]:
        H = np.array([[1.0, b]])
        z = np.array([[a]])
        self.P = self.P + self.Q
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        return float(y), float(S)
```

The `update(a, b)` call returns the standardised spread (innovation) `y` and its variance `S`. The `z = y / sqrt(S)` is your trading signal.

## A complete pairs strategy

```python
import numpy as np
import pandas as pd
from engine.data import YFinanceFeed, ParquetCache

feed = ParquetCache(YFinanceFeed(), root="data/bars")

# Two consumer staples that have historically tracked
ko  = feed.bars("KO",  "2014-01-01", "2024-12-31")["close"]
pep = feed.bars("PEP", "2014-01-01", "2024-12-31")["close"]
df = pd.concat({"KO": ko, "PEP": pep}, axis=1).dropna()

# Kalman-filtered spread
kp = KalmanPairs(delta=1e-4, R=1.0)
rows = []
for a, b in zip(df["KO"], df["PEP"]):
    y, S = kp.update(a, b)
    rows.append((float(kp.x[0]), float(kp.x[1]), y, S ** 0.5))
state = pd.DataFrame(rows, index=df.index, columns=["alpha", "beta", "spread", "z_sd"])
state["z"] = state["spread"] / state["z_sd"]

# Trading rule: enter at |z| > 2, exit at |z| < 0.5
entry, exit_th = 2.0, 0.5
position = pd.Series(0, index=state.index, dtype=int)
for i in range(1, len(state)):
    z = state["z"].iat[i]
    prev = position.iat[i - 1]
    if prev == 0:
        if z >  entry: position.iat[i] = -1                # short KO, long PEP
        elif z < -entry: position.iat[i] = +1               # long KO, short PEP
    else:
        if abs(z) < exit_th: position.iat[i] = 0
        else: position.iat[i] = prev

# P&L: long KO, short beta_t * PEP per unit position
ret_ko  = df["KO"].pct_change().fillna(0)
ret_pep = df["PEP"].pct_change().fillna(0)
pair_ret = position.shift(1) * (ret_ko - state["beta"].shift(1) * ret_pep)
sharpe = pair_ret.mean() / pair_ret.std() * np.sqrt(252)
print(f"KO-PEP pairs Sharpe (gross): {sharpe:.2f}")
```

On a ten-year sample, expect Sharpe ~0.5-0.8 *gross of trading costs*. Net of realistic borrow + commissions, it's often less. The Kalman version typically outperforms a rolling-OLS hedge by 0.2-0.3 Sharpe because it adapts to slow changes in the relationship.

## How to find pairs at scale

1. **Same sector, similar size, same exchange** — narrow the universe.
2. **Pre-filter on correlation** — only test pairs with rolling correlation > 0.8.
3. **Engle-Granger on the candidates** — apply with FDR correction (Module 6) because you're running many tests.
4. **Verify with a Kalman fit** — check that the fitted $\beta_t$ is stable, not wildly jumping around. Unstable betas mean the cointegration is a fluke.
5. **Out-of-sample test** — fit on first 60% of data, test on the last 40%.

Steps 1-3 typically take a few hundred candidates from thousands of pairs. Steps 4-5 winnow further to perhaps 10-30 tradable pairs from any reasonable universe.

## A worked example: scan for cointegrated pairs

```python
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

def scan_pairs(prices: pd.DataFrame, p_thresh: float = 0.01) -> pd.DataFrame:
    rows = []
    cols = prices.columns
    for a, b in combinations(cols, 2):
        sub = prices[[a, b]].dropna()
        if len(sub) < 250:
            continue
        try:
            _, p, _ = coint(sub[a], sub[b])
        except Exception:
            continue
        if p < p_thresh:
            corr = sub.pct_change().corr().iloc[0, 1]
            rows.append({"a": a, "b": b, "p": p, "corr": corr})
    return pd.DataFrame(rows).sort_values("p")
```

Run this on a sector ETF universe and you typically get a handful of statistically-cointegrated pairs at the 1% level. **Apply FDR correction** (Benjamini-Hochberg) before deciding which ones are real.

## The risk-management addendum

Pairs strategies sound idyllic — dollar-neutral, market-neutral — until a pair *breaks*. KMart and Walmart cointegrated for years; then KMart went bankrupt. The risks:

1. **Single-name credit blowup.** Hard stop per pair. If $|z| > 4$ for more than $N$ days, exit and don't re-enter for a quarantine period.
2. **Borrow / hard-to-borrow.** Some names are illiquid to short. Pre-screen.
3. **Earnings.** Don't carry pairs through earnings; the idiosyncratic move dominates the cointegration.
4. **Margin calls.** Run the portfolio with conservative leverage; pairs blow up in clusters during stress.

The course's Module 16 wraps the pairs strategy with explicit risk overlays. Don't ship without them.

## Pitfalls

!!! warning "Cointegration is sample-dependent"
    Two random walks pass the Engle-Granger test at 5% about 5% of the time. With 1000 candidate pairs, expect ~50 spurious "cointegrated" results. FDR or Bonferroni.

!!! warning "Hedge ratio is the regression slope, not the market beta"
    A common bug: using the market betas of A and B as the hedge ratio. The correct hedge ratio is the slope of A on B from the cointegration regression.

!!! warning "Walk-forward, always"
    Cointegration found in-sample often disappears out-of-sample. Always reserve a holdout (Module 9 has more on this).

!!! warning "Spread bias from log vs simple prices"
    For dollar-neutral pairs, the spread is in price units. For ratio-neutral, log prices. Pick one consistently and document the choice.

## Bottom line

A complete pairs-trading workflow:

- **Filter** candidates by sector, size, exchange, correlation > 0.8.
- **Test** cointegration with Engle-Granger; apply FDR.
- **Fit** time-varying $\beta_t$ with a Kalman filter.
- **Trade** the standardised spread with $\pm 2$/$\pm 0.5$ entry/exit.
- **Risk-manage** with hard stops, no earnings overlap, conservative leverage.

For multi-asset cointegration baskets, swap Engle-Granger for Johansen and use the leading eigenvector as your spread.

Continue to **[Portfolio construction: Markowitz, Black-Litterman, risk parity](05-portfolio.md)**.
