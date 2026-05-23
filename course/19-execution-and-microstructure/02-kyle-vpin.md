# Kyle's lambda and VPIN — measuring informed flow

When you trade, some of the volume around you is from informed traders (who know something) and some is from uninformed traders (liquidity demanders). Distinguishing the two matters: trading alongside informed flow loses money; trading against uninformed flow makes money.

Two foundational metrics: **Kyle's lambda** (the marginal price impact per unit volume) and **VPIN** (volume-synchronised probability of informed trading). Both have been operationalised in the published literature; both inform execution.

## Kyle's lambda — recap and estimation

Kyle (1985). The basic idea: if all your trading impact were caused by informed traders moving the equilibrium price, the regression coefficient of price changes on signed order flow gives you lambda.

$$
\Delta P_t = \lambda \cdot Q_t + \varepsilon_t
$$

where $Q_t$ is signed order flow (positive for buys, negative for sells).

High lambda → illiquid market, each share moves the price a lot.

**Estimating in practice**: signed-order-flow via the Lee-Ready algorithm (Module 18 chapter 4); regress over 1-minute bars; report lambda in basis-points per million-share volume.

```python
import pandas as pd
import numpy as np


def lee_ready_sign(trade_price: float, mid_price: float, tick_test: int = 0) -> int:
    """Lee-Ready 1991. Sign = +1 if trade above mid, -1 below; tick rule on the midpoint."""
    if trade_price > mid_price: return +1
    if trade_price < mid_price: return -1
    return tick_test                       # at mid: use last tick direction


def kyles_lambda(trades: pd.DataFrame, midpoints: pd.Series, freq: str = "1min") -> dict:
    """Trades: ts (DatetimeIndex), price, size.
       midpoints: ts (DatetimeIndex), value.
       Returns lambda + diagnostic stats."""
    # Sign each trade
    signed = []
    last_dir = 0
    for ts, row in trades.iterrows():
        mid = midpoints.asof(ts)
        if pd.isna(mid): continue
        d = lee_ready_sign(row["price"], mid, last_dir)
        if d != 0: last_dir = d
        signed.append((ts, d * row["size"]))
    signed_df = pd.DataFrame(signed, columns=["ts", "signed_size"]).set_index("ts")
    # Resample to bars
    bars = signed_df.resample(freq).sum()
    bars["mid_at_close"] = midpoints.resample(freq).last()
    bars["dp"] = bars["mid_at_close"].diff()
    bars = bars.dropna()
    if len(bars) < 5:
        return {"lambda": float("nan"), "r_squared": float("nan"), "n_bars": 0}
    # OLS
    x = bars["signed_size"].values
    y = bars["dp"].values
    lam = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else float("nan")
    yhat = lam * x
    r2 = 1 - np.var(y - yhat) / np.var(y) if np.var(y) > 0 else 0
    return {"lambda": float(lam), "r_squared": float(r2), "n_bars": len(bars),
            "lambda_bps_per_M": float(lam * 1e6 / bars["mid_at_close"].mean() * 10000)}
```

Lambda translates to: **"trading 1M shares of this name moves the price by X bps in expectation."** For SPY, X is around 0.5-1 bps. For a mid-cap stock, 10-30 bps. For a micro-cap, 100+ bps.

## VPIN

Easley, López de Prado, O'Hara (2012). A real-time estimator of the probability that recent flow is dominated by informed traders.

Method:

1. Group trades into **equal-volume buckets** (not equal-time). A typical bucket is 1/50th of the day's average volume.
2. For each bucket, classify each trade as buy or sell (Lee-Ready).
3. Compute the bucket's **order imbalance**: |buy_volume − sell_volume| / total_volume.
4. **VPIN** = rolling-average of bucket imbalances over the last N buckets.

A high VPIN indicates that recent flow is one-sided — typically a sign of informed pressure.

```python
import pandas as pd
import numpy as np


def vpin(trades: pd.DataFrame, midpoints: pd.Series,
          n_buckets_per_day: int = 50, window_n: int = 50) -> pd.Series:
    """VPIN over volume buckets. trades: ts, price, size."""
    daily_volume = trades["size"].sum()
    bucket_size = daily_volume / n_buckets_per_day
    # Sign each trade
    last_dir = 0
    signs = []
    for ts, row in trades.iterrows():
        mid = midpoints.asof(ts)
        if pd.isna(mid): signs.append(0); continue
        d = lee_ready_sign(row["price"], mid, last_dir)
        if d != 0: last_dir = d
        signs.append(d)
    trades = trades.assign(sign=signs)
    trades["cum_volume"] = trades["size"].cumsum()
    trades["bucket"] = (trades["cum_volume"] / bucket_size).astype(int)
    # Per-bucket buy / sell volumes
    grouped = trades.groupby("bucket").apply(
        lambda g: pd.Series({
            "buy": g.loc[g["sign"] == +1, "size"].sum(),
            "sell": g.loc[g["sign"] == -1, "size"].sum(),
        })
    )
    grouped["imbalance"] = (grouped["buy"] - grouped["sell"]).abs() / (grouped["buy"] + grouped["sell"]).replace(0, 1)
    return grouped["imbalance"].rolling(window_n).mean()
```

## When VPIN spikes

A VPIN spike (>90th percentile of recent history) historically precedes:

- Vol spikes within minutes-to-hours.
- Flash-crash-like events at the extreme.
- Big-news arrivals (sometimes the news hits the tape after VPIN already moved — informed trading ahead of the public announcement).

Use as a **real-time risk-off signal**: when VPIN exceeds threshold, reduce position sizing, widen scopes, pause aggressive execution.

## Combining the two

Kyle's lambda is structural (it describes the asset's liquidity). VPIN is temporal (it describes the *current* informed-flow regime).

A good execution algo uses both:

- **High lambda + low VPIN**: illiquid name in a calm period. Trade carefully but no urgency.
- **High lambda + high VPIN**: illiquid name in informed-flow regime. PAUSE — wait it out or trade against (very dangerous).
- **Low lambda + high VPIN**: liquid name in informed-flow regime. Execute carefully — the spread is small but the next print might be big.
- **Low lambda + low VPIN**: liquid name in calm regime. Trade freely.

## A worked example

```python
from engine.data import YFinanceFeed

# Pretend trade/quote data
trades = pd.DataFrame({...})            # ts, price, size
midpoints = pd.Series(...)              # ts, mid

kl = kyles_lambda(trades, midpoints, freq="1min")
print(f"Kyle's lambda: {kl['lambda_bps_per_M']:.2f} bps/M shares")

v = vpin(trades, midpoints)
print(f"Recent VPIN: {v.iloc[-1]:.2%}")
print(f"VPIN P90 (recent): {v.tail(200).quantile(0.9):.2%}")

# Decision
if v.iloc[-1] > v.tail(200).quantile(0.9):
    print("HIGH VPIN — risk-off; pause aggressive execution")
```

## Pitfalls

!!! warning "Lee-Ready accuracy"
    The Lee-Ready sign isn't perfect — about 75-85% correct on US equities. For tiny tick sizes or high-frequency markets, accuracy drops. Modern methods (BVC, Easley-O'Hara) improve but require more data.

!!! warning "Survivorship in lambda time series"
    A lambda estimate computed during exchange-meltdown moments looks like infinity. Filter outliers and use robust regression (e.g., M-estimator) instead of OLS.

!!! warning "VPIN's window choice"
    The rolling window (`window_n`) and bucket size affect VPIN substantially. For SPY-like underlyings, 50 buckets per day × 50-bucket window is reasonable. For thinner names, fewer buckets.

!!! warning "Causality vs prediction**
    VPIN spikes often *coincide* with vol moves rather than purely *precede* them. Don't expect a 10-minute warning; it's more like 30 seconds to 2 minutes.

## Bottom line

Kyle's lambda and VPIN are the working microstructure metrics for execution and risk:

- **Kyle's lambda** for *structural* liquidity (asset-level cost of trading).
- **VPIN** for *temporal* informed-flow detection (current regime).
- Combine them — informed regimes on illiquid names are the most dangerous to trade through.

Continue to **[Realised kernels and pre-averaging](03-realised-kernels.md)**.
