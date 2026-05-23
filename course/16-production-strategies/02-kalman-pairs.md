# Kalman pairs trading at production grade

Module 8 chapter 4 introduced the Kalman-filtered pairs strategy as a research idea. This chapter takes it to production: a complete implementation with cointegration screening, walk-forward retraining, risk controls (hard stops, earnings filters), and capacity awareness.

## The pipeline

```
universe ──► sector / size / liquidity filter
                          │
                          ▼
              correlation pre-screen (rolling)
                          │
                          ▼
                Engle-Granger test with FDR
                          │
                          ▼
                Kalman filter — hedge ratio
                          │
                          ▼
       z-score of spread innovation ──► trade rule
                          │
                          ▼
                risk overlay (stops, earnings)
                          │
                          ▼
                portfolio aggregation
```

## Universe filter

Don't even consider pairs across:

- **Different sectors** — economic relationship is too weak.
- **Vastly different market caps** — liquidity asymmetry kills you.
- **Different exchanges** — settlement, timing, holiday calendars complicate.

A reasonable starting universe: same sector (GICS level 2), market cap within a 3× ratio, both with average daily volume > $50M.

## Correlation pre-screen

Even within the filtered universe, only test cointegration on pairs with rolling correlation > 0.8. This cuts the test count by 10-100× and avoids spurious cointegration on unrelated names.

```python
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint
from scipy.stats import false_discovery_control


def correlation_screen(prices: pd.DataFrame, threshold: float = 0.80,
                       window: int = 252) -> list[tuple[str, str]]:
    """Return pairs with rolling correlation > threshold over the last window."""
    rets = prices.pct_change().tail(window)
    corr = rets.corr()
    pairs = []
    for a, b in combinations(prices.columns, 2):
        if corr.at[a, b] > threshold:
            pairs.append((a, b))
    return pairs


def cointegration_fdr(prices: pd.DataFrame, pairs: list[tuple[str, str]],
                      alpha_fdr: float = 0.10) -> pd.DataFrame:
    """Run Engle-Granger on each candidate pair, apply FDR correction."""
    rows = []
    for a, b in pairs:
        sub = prices[[a, b]].dropna()
        if len(sub) < 252:
            continue
        try:
            stat, p, _ = coint(sub[a], sub[b])
            rows.append({"a": a, "b": b, "p": p, "n": len(sub)})
        except (ValueError, KeyError):
            continue
    if not rows:
        return pd.DataFrame(columns=["a", "b", "p", "n", "p_adj", "reject"])
    df = pd.DataFrame(rows)
    df["p_adj"] = false_discovery_control(df["p"].values, method="bh")
    df["reject"] = df["p_adj"] < alpha_fdr
    return df.sort_values("p_adj")
```

The FDR correction (Benjamini-Hochberg) is essential — you're running thousands of tests; without correction, ~5% are spurious at the conventional p<0.05 cutoff.

## Kalman pairs trading loop

The Module 8 Kalman is the workhorse. Wrap it in a strategy class:

```python
import numpy as np
import pandas as pd


class KalmanPairsState:
    def __init__(self, delta: float = 1e-4, R: float = 1.0):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * (delta / (1 - delta))
        self.R = np.array([[R]])

    def update(self, a: float, b: float) -> tuple[float, float, float, float]:
        H = np.array([[1.0, b]])
        z = np.array([[a]])
        self.P = self.P + self.Q
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        return float(self.x[0]), float(self.x[1]), float(y), float(S ** 0.5)


def kalman_pairs_signals(a_close: pd.Series, b_close: pd.Series,
                          entry_z: float = 2.0, exit_z: float = 0.5,
                          hard_stop_z: float = 4.0,
                          delta: float = 1e-4):
    state = KalmanPairsState(delta=delta)
    rows = []
    position = 0
    for ts, a, b in zip(a_close.index, a_close.values, b_close.values, strict=True):
        alpha, beta, spread, sd = state.update(a, b)
        z = spread / sd if sd > 0 else 0.0
        new_pos = position
        # Hard stop on extreme moves first
        if abs(z) > hard_stop_z:
            new_pos = 0
        else:
            if position == 0:
                if z > entry_z: new_pos = -1
                elif z < -entry_z: new_pos = +1
            else:
                if abs(z) < exit_z: new_pos = 0
        position = new_pos
        rows.append((ts, alpha, beta, spread, sd, z, position))
    return pd.DataFrame(rows, columns=["ts", "alpha", "beta", "spread", "sd", "z", "position"]).set_index("ts")
```

## Risk overlays

### Hard stop on extreme z

Beyond entry/exit thresholds, exit unconditionally if `|z| > 4`. This catches the case where the cointegration relationship breaks (one name is going to zero).

### No earnings overlap

```python
def filter_earnings(positions: pd.Series, earnings_dates: pd.Series,
                     window_days: int = 2) -> pd.Series:
    """Zero out positions within `window_days` of any earnings date."""
    out = positions.copy()
    for ed in earnings_dates:
        mask = (positions.index >= ed - pd.Timedelta(days=window_days)) & \
               (positions.index <= ed + pd.Timedelta(days=window_days))
        out.loc[mask] = 0
    return out
```

For pairs strategies, the idiosyncratic earnings move dominates the cointegration. Sit it out.

### Borrow cost in the cost model

```python
def pair_pnl_with_borrow(pos: pd.Series, ret_a: pd.Series, ret_b: pd.Series,
                          beta: pd.Series, borrow_rate_ann: float = 0.005) -> pd.Series:
    """Realistic pair P&L: long A, short beta * B, paying borrow on the short leg."""
    daily_borrow = borrow_rate_ann / 252
    leg_a = pos * ret_a
    leg_b = -pos * beta.shift(1) * ret_b
    borrow = -np.abs(pos * beta.shift(1)) * daily_borrow
    return leg_a + leg_b + borrow
```

## A worked example

```python
from engine.data import YFinanceFeed, ParquetCache

feed = ParquetCache(YFinanceFeed(), root="data/bars")
ko = feed.bars("KO", "2018-01-01", "2024-12-31")["close"]
pep = feed.bars("PEP", "2018-01-01", "2024-12-31")["close"]
df = pd.concat({"KO": ko, "PEP": pep}, axis=1).dropna()

signals = kalman_pairs_signals(df["KO"], df["PEP"],
                                entry_z=2.0, exit_z=0.5, hard_stop_z=4.0, delta=1e-4)

ret_ko = df["KO"].pct_change().fillna(0)
ret_pep = df["PEP"].pct_change().fillna(0)
pair_pnl = pair_pnl_with_borrow(signals["position"].shift(1).fillna(0),
                                 ret_ko, ret_pep, signals["beta"], borrow_rate_ann=0.005)
sharpe = pair_pnl.mean() / pair_pnl.std() * np.sqrt(252)
print(f"KO-PEP Sharpe (with borrow): {sharpe:.2f}")
print(f"Max drawdown: {((1+pair_pnl).cumprod() / (1+pair_pnl).cumprod().cummax() - 1).min():.2%}")
```

Expect Sharpe 0.5-0.8 on a representative period after borrow and modest transaction costs.

## Portfolio of pairs

A single pair has high idiosyncratic risk. Running 20-50 pairs (each ~1-2% of capital, vol-targeted) gives a diversified portfolio with ~0.8-1.2 Sharpe — the kind of strategy that ships at multi-billion shops.

```python
def portfolio_pairs(pair_returns: pd.DataFrame, vol_target: float = 0.10):
    """Equal-vol weight each pair; cap total leverage."""
    vol = pair_returns.rolling(60).std() * np.sqrt(252)
    weights = (vol_target / vol).clip(upper=3.0)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)        # gross-exposure-normalised
    return (weights.shift(1) * pair_returns).sum(axis=1)
```

## Walk-forward retraining

Refit the universe filter and the cointegration list **quarterly**. Pairs come and go; the universe of viable candidates changes.

For each quarter:

1. Re-screen the universe (correlation, sector, liquidity).
2. Re-test cointegration with FDR.
3. Compare the new "active" list to the previous one.
4. For pairs that left the list, close positions over the quarter.
5. For new pairs, start trading at small size, scale up over a month.

This avoids "fire all positions on day 1 of the new quarter" turnover spikes.

## Capacity

A pairs strategy has limited capacity per pair (because the spread snaps back quickly when too much capital is in). Estimate as:

$$
\text{Capacity per pair} \approx \frac{\text{average daily turnover in the spread}}{10}
$$

For most liquid equity pairs, that's $5-50M per pair. Across 30 pairs, $150M-1.5B total. Beyond that, your own flow moves the spread and the alpha disappears.

## Pitfalls

!!! warning "Cointegration that disappears"
    A pair cointegrated for 5 years may stop cointegrating tomorrow. Always run a rolling cointegration test and exit pairs that fail recent tests.

!!! warning "Selection bias in pair candidates"
    Picking the pair with the lowest in-sample p-value is selection bias. Walk-forward test before trading.

!!! warning "Survivorship bias in the universe"
    If your universe is "today's S&P 500", you've removed companies that went bankrupt. Several real pairs strategies in the literature evaporate when run on a point-in-time universe.

!!! warning "Hard-to-borrow specials"
    Some names have astronomical borrow costs (e.g., short-squeeze candidates). Pre-filter to "borrowable at <100bp/year" or include borrow cost in the trade's expected return.

## Bottom line

A production-grade Kalman pairs strategy:

- **Filtered universe**: same sector, similar size, liquid, easy-to-borrow.
- **FDR-corrected cointegration screen**, quarterly refresh.
- **Kalman-filtered hedge ratio**, online.
- **2σ entry / 0.5σ exit / 4σ hard stop**.
- **Earnings filter**: no positions within 2 days of either name's earnings.
- **Borrow cost in P&L**.
- **Portfolio of 20-50 pairs**, vol-targeted at 10% per pair, gross-leverage-capped.

Capacity: $100M-$1B+ depending on universe liquidity. Sharpe expected: 0.8-1.2 in production.

Continue to **[Gradient-boosted classifier with conformal sizing](03-conformal-gbm.md)**.
