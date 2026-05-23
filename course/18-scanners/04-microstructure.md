# Microstructure scanners

Microstructure scanners operate at higher frequencies than the daily-bar scanners in chapters 2-3. They look at order-book dynamics, trade-tape signatures, and intraday flow imbalances to identify short-horizon trading opportunities. They require minute-or-tick data; most retail platforms don't provide it.

This chapter sketches three classics: Kyle's lambda, VPIN, and queue-position scanners. Module 19 (Execution and Microstructure) goes deeper into the underlying theory.

## 1. Kyle's lambda

Kyle (1985). The **price impact per unit of volume**. A high Kyle's lambda means the underlying is illiquid (each trade moves the price a lot); low means liquid.

The empirical estimator: regress price changes on signed volume over a short window.

$$
\Delta P_t = \lambda \cdot \text{SignedVolume}_t + \varepsilon_t
$$

Signed volume = (buy volume - sell volume), where you infer "buy" vs "sell" from the trade's relationship to the prevailing quote (Lee-Ready algorithm).

```python
import pandas as pd
import numpy as np
from engine.scanners import scanner


def estimate_kyles_lambda(trades: pd.DataFrame, quotes: pd.DataFrame, window_minutes: int = 30) -> float:
    """Trades: ts, price, size. Quotes: ts, bid, ask. Sample at 1-min frequency."""
    # Lee-Ready: buy if trade > midpoint, sell otherwise
    quotes_aligned = pd.merge_asof(trades.sort_values("ts"), quotes.sort_values("ts"), on="ts")
    quotes_aligned["mid"] = (quotes_aligned["bid"] + quotes_aligned["ask"]) / 2
    quotes_aligned["signed_size"] = np.where(quotes_aligned["price"] > quotes_aligned["mid"],
                                              +quotes_aligned["size"], -quotes_aligned["size"])
    minute_bars = quotes_aligned.resample("1min", on="ts").agg(
        signed_volume=("signed_size", "sum"),
        last_price=("price", "last"),
    ).dropna()
    minute_bars["dp"] = minute_bars["last_price"].diff()
    minute_bars = minute_bars.dropna()
    # Linear regression
    x = minute_bars["signed_volume"].values
    y = minute_bars["dp"].values
    if len(x) < 5: return float("nan")
    return float(np.cov(x, y)[0, 1] / np.var(x))


@scanner("kyles-lambda")
class KylesLambdaScanner:
    def __init__(self, percentile_threshold: float = 95.0):
        self.percentile_threshold = percentile_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            try:
                trades = fetch_trades(sym, asof - pd.Timedelta(hours=2), asof)
                quotes = fetch_quotes(sym, asof - pd.Timedelta(hours=2), asof)
                lam = estimate_kyles_lambda(trades, quotes, window_minutes=30)
                if np.isfinite(lam):
                    rows.append({"symbol": sym, "lambda": lam})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty: return df
        threshold = df["lambda"].quantile(self.percentile_threshold / 100)
        return df[df["lambda"] > threshold].sort_values("lambda", ascending=False)
```

High Kyle's lambda symbols are **illiquid** — trade in them with size will move the price significantly. Use as a **warning** for execution algos.

## 2. VPIN — Volume-synchronised Probability of Informed Trading

Easley, López de Prado, O'Hara (2012). Estimates the probability that the current flow contains informed traders (vs uninformed). A rising VPIN warns of an impending vol spike.

```python
def compute_vpin(trades: pd.DataFrame, volume_bucket_size: int, n_buckets: int = 50) -> pd.Series:
    """VPIN over rolling volume buckets."""
    trades = trades.sort_values("ts").reset_index(drop=True)
    trades["cum_volume"] = trades["size"].cumsum()
    bucket_id = (trades["cum_volume"] // volume_bucket_size).astype(int)
    trades["bucket"] = bucket_id
    # For each bucket, estimate buy/sell volume imbalance
    # (Simplified: use trade-size weighted normal approximation)
    grouped = trades.groupby("bucket").agg(
        buy_vol=("size", lambda x: x[trades.loc[x.index, "side"] == "BUY"].sum()),
        sell_vol=("size", lambda x: x[trades.loc[x.index, "side"] == "SELL"].sum()),
    )
    grouped["imbalance"] = (grouped["buy_vol"] - grouped["sell_vol"]).abs()
    grouped["total"] = grouped["buy_vol"] + grouped["sell_vol"]
    grouped["vpin"] = grouped["imbalance"] / grouped["total"]
    return grouped["vpin"].rolling(n_buckets).mean()


@scanner("vpin-spike")
class VPINSpikeScanner:
    def __init__(self, baseline_pct: float = 80.0):
        self.baseline_pct = baseline_pct

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            try:
                trades = fetch_trades(sym, asof - pd.Timedelta(hours=4), asof)
                volume_bucket = trades["size"].sum() / 50          # 50 buckets in 4 hours
                vpin = compute_vpin(trades, volume_bucket_size=volume_bucket)
                if vpin.empty: continue
                current = vpin.iloc[-1]
                baseline = vpin.quantile(self.baseline_pct / 100)
                if current > baseline:
                    rows.append({"symbol": sym, "vpin": float(current),
                                  "baseline_pct": self.baseline_pct, "baseline_value": float(baseline)})
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("vpin", ascending=False)
```

VPIN spikes precede vol spikes more often than not. Use as a real-time risk-off signal.

## 3. Queue position deterioration

For market-making strategies, the **queue position** of your resting orders matters. If you're at the top of the queue, you fill on the next adverse print; if you're at the back, you're effectively decorative.

```python
@scanner("queue-deterioration")
class QueueDeteriorationScanner:
    """For your own resting limits, flag when queue position has deteriorated significantly."""
    def __init__(self, queue_drop_threshold: float = 0.5):
        self.threshold = queue_drop_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            my_orders = fetch_resting_orders(sym, asof)
            for order in my_orders:
                # Estimate current queue position from book updates
                pos = estimate_queue_position(order, asof)
                if pos["fraction_ahead"] > self.threshold:
                    rows.append({"symbol": sym, "order_id": order.id,
                                  "side": order.side, "price": order.price,
                                  "queue_fraction_ahead": pos["fraction_ahead"]})
        return pd.DataFrame(rows).sort_values("queue_fraction_ahead", ascending=False)
```

The scanner output is a list of orders to **cancel and re-place** (because they're too far back to fill on favourable prints).

## 4. Spread compression / expansion

For execution algos, the bid-ask spread tells you when liquidity is plentiful (compressed) or scarce (expanded). Scanners on spread state are valuable for choosing when to deploy passive vs aggressive execution.

```python
@scanner("spread-expansion")
class SpreadExpansionScanner:
    def __init__(self, expansion_factor: float = 3.0, baseline_minutes: int = 30):
        self.expansion_factor = expansion_factor
        self.baseline_minutes = baseline_minutes

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            try:
                quotes = fetch_quotes(sym, asof - pd.Timedelta(minutes=self.baseline_minutes), asof)
                quotes["spread_bps"] = (quotes["ask"] - quotes["bid"]) / quotes["bid"] * 10000
                current = quotes["spread_bps"].iloc[-1]
                baseline = quotes["spread_bps"].median()
                if current / baseline > self.expansion_factor:
                    rows.append({"symbol": sym, "current_spread_bps": current,
                                  "baseline_spread_bps": baseline,
                                  "expansion_ratio": current / baseline})
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("expansion_ratio", ascending=False)
```

When spreads expand 3-10× the baseline, liquidity has evaporated. **Pause execution algos**; reduce position sizing.

## Combining with the production scanners

Microstructure signals are highest-value when **paired with structural events**: an unusual options volume flag + a VPIN spike + a spread expansion = high-conviction "something is happening, position carefully."

```python
def hot_microstructure_today(universe, asof):
    flagged = {}
    for scanner_name in ["unusual-options-volume", "vpin-spike", "spread-expansion", "hawkes-spike"]:
        df = registry.get(scanner_name)().scan(universe, asof)
        for sym in df["symbol"].unique():
            flagged.setdefault(sym, set()).add(scanner_name)
    # Symbols flagged by 3+ microstructure scanners → highest-attention
    return [s for s, sources in flagged.items() if len(sources) >= 3]
```

## Pitfalls

!!! warning "Data quality at high frequency"
    Microstructure scanners are very sensitive to data quality. Out-of-order timestamps, missed quotes, or feed delays produce false signals. Use vendor-grade SIP/feeds, not Yahoo.

!!! warning "Compute cost at scale"
    Real-time VPIN on a thousand-symbol universe is a serious engineering exercise. Often implemented in C++/Rust with Python for the analytics layer.

!!! warning "Backtesting microstructure scanners"
    Truly historical L2/L3 data is expensive ($10k-100k/year per market). Many published "microstructure backtests" use proxies that aren't quite the same as live conditions.

!!! warning "Symbol-specific calibration"
    Spread baselines, queue depths, lambdas — all vary dramatically by symbol. Single-threshold scanners flag the wrong symbols. Always per-symbol calibration.

## Bottom line

Microstructure scanners are:

- **High-value signals** for intraday strategies and execution.
- **Data-hungry** — require real-time L2/L3.
- **Calibration-heavy** — per-symbol thresholds, not universal.
- **Complementary** to structural scanners.

For most retail and small-fund work, microstructure scanners are aspirational. For execution-sensitive shops and HFT, they're table stakes.

Continue to **[Composing scanners into a daily report](05-composition.md)**.
