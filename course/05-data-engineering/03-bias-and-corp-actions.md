# Survivorship bias, corporate actions, point-in-time

This is the most important chapter in Module 5 — and the one most beginners skip. The biases hiding in your historical data are exactly the kind that make backtests look great and live trading lose money. Internalise them before you write your first real backtest.

## Survivorship bias

A historical index — "the S&P 500 today" — does not contain the companies that *were* in the S&P 500 ten years ago but went bankrupt, got acquired, or were delisted. If your backtest universe is "today's S&P 500", you're testing only the survivors. That's a free 1–3% per year of fake outperformance.

### How big is the bug?

A frequently cited figure: a long-only momentum strategy on "today's S&P 500" outperforms the same strategy on a survivorship-bias-free universe by **2–4% annualised**. Over 20 years that's a 50–100% backtest-vs-reality gap.

### The fix

You need a **historical universe** — a record of which symbols were in the index on each date. Sources:

- **CRSP** (commercial, the gold standard for US equities).
- **Bloomberg / FactSet / S&P** indices include historical constituents.
- **Polygon.io** has historical S&P 500 membership for US equities (paid).
- **Wikipedia + Wayback Machine** if you're scrappy.

Once you have it, build a **point-in-time universe**:

```python
# universe.parquet — partitioned by date
# columns: date, symbol, in_universe (boolean)

universe = pl.scan_parquet("data/universe/*.parquet")
# For a backtest day t, the universe is everything `in_universe` on t-1 (no look-ahead)
todays_universe = universe.filter(pl.col("date") == prior_business_day(t)).select("symbol")
```

The course shows the production-grade version in Module 9 (Backtesting).

## Corporate actions: splits and dividends

A stock split (2:1 means double the shares, half the price) and a dividend both change the raw price series in ways that look like big moves but aren't.

### Two adjustment philosophies

**Total-return adjusted** prices reinvest dividends and bake in splits, so the series represents what a buy-and-hold investor would have actually earned. Yahoo's `Adj Close` is this.

**Split-adjusted but not dividend-adjusted** is the right series for backtests where dividends are handled separately (so you can model dividend-paying stocks correctly). Polygon and most professional vendors offer both.

For most signal research, use **total-return adjusted prices to compute returns**, then verify your strategy's P&L includes dividends consistently.

### The bug pattern

```python
# UNSAFE — `close` is the raw close, ignoring the dividend that paid yesterday
returns = df["close"].pct_change()
```

If a $1 dividend paid on a $100 stock, the close drops by $1 on ex-date. Your computed return is -1% — that's the *negative* of what holders actually earned (zero, because they got the dollar). Use the adjusted series, or compute returns from total-return data:

```python
returns = df["adj_close"].pct_change()        # treats dividends as reinvested
```

### Stock splits: the silent disaster

A 2:1 split on day t makes day t's raw close half of day (t-1)'s raw close. Without adjustment, your strategy sees a -50% return and trades wildly.

Yahoo's `yfinance` library returns split-adjusted prices by default. Polygon offers both raw and adjusted series; *always specify which you want*. The cache (chapter 1) stores whatever the upstream returned — when in doubt, drop the adjusted close and use only the raw + a separate adjustment series.

## Point-in-time fundamentals

Fundamentals (earnings, balance sheet, sector classifications) are restated frequently. Without point-in-time discipline, you'll have access to numbers in your backtest that nobody had when the decision was made.

### Two killer cases

1. **Earnings restatements.** A company reports EPS of $1.20 in Q1, then restates it to $0.85 six months later. A "current" fundamentals database has $0.85 for that quarter — but at the time, traders saw $1.20.

2. **Index reconstitution.** S&P adds or removes companies on specific dates. The right backtest uses the membership *as it was on day t-1*, not the current membership.

### The fix

Store fundamentals **as snapshots**, one per release:

```
data/fundamentals/
  asof=2024-Q1-released-2024-04-25/
    data.parquet
  asof=2024-Q1-restated-2024-08-15/
    data.parquet
```

To answer "what did we know on 2024-06-30?":

```python
released = pl.scan_parquet("data/fundamentals/**/*.parquet").filter(
    pl.col("release_date") <= pl.date(2024, 6, 30)
)
latest = released.group_by(["symbol", "fiscal_period"]).agg(
    pl.col("eps").last().alias("eps_pit"),
)
```

The `release_date <= asof` filter is the entire point-in-time discipline in one line.

## Look-ahead bias from your own code

Not all bias comes from data. The most insidious comes from your own pipeline:

### Bug 1: signal aligned to the same day's bar

```python
signal = momentum(returns)
pnl = signal * returns                 # WRONG — using today's signal on today's return
```

`signal` was computed from the close of day t. You can only act on the next day. The fix:

```python
pnl = signal.shift(1) * returns
```

This single `.shift(1)` is the difference between a backtest that's a fantasy and one that's honest.

### Bug 2: rolling statistics using future bars

```python
ma = df["close"].rolling(20, center=True).mean()   # BUG — half the window is in the future
```

`center=True` is *only* for offline smoothing, never for signals.

### Bug 3: train/test contamination

```python
scaler = StandardScaler().fit(returns)             # uses all data including the test set
returns_scaled = scaler.transform(returns)
```

Train statistics must be computed on training data only. The right way is **time-series CV** with purged splits, covered in Module 9.

### Bug 4: training labels leaking via overlapping windows

If your labels are "return over the next 5 days" and your training samples overlap by 4 days, neighbouring training rows share most of their labels. K-fold CV will give optimistic scores because the "test" rows are nearly identical to "train" rows.

The fix is **purging and embargoing** — when splitting, remove training rows whose label horizon overlaps the test window. Module 9 implements this from scratch.

## A practical PIT-correct workflow

When you build a backtest, every input series should have an explicit "this was known by..." timestamp. The discipline:

1. **Bars** — known by the bar's `close + 1 second` (for daily bars, the next session's open).
2. **Fundamentals** — known by the `release_date`.
3. **Index membership** — known by the day the index publisher announces it (S&P typically announces 5 days in advance).
4. **Features computed from bars** — known by the most recent bar's known-by time. A rolling 20-day mean as of close on day t is known by t+1's session open.
5. **Signals** — known one step after the latest input's known-by.

If every series carries this metadata, your backtest cannot leak. In code, the discipline becomes:

```python
features = compute_features(bars).shift(1)    # known after close, used next day
signal = model.predict(features)              # signal computed from yesterday's features
pnl = signal * bars["return"]                 # acted on today
```

Two `.shift(1)`s (one implicit in computing features from already-closed bars, one explicit on the signal) cover most of it.

## Common bugs by symptom

| Symptom | Likely bug |
|---|---|
| Backtest Sharpe > 4 on a simple signal | look-ahead, or survivorship |
| Strategy outperforms in-sample, fails OOS | overfit + no purging |
| Strategy works on QQQ, fails on real index | survivorship bias in the universe |
| Strategy underperforms after dividends paid | unadjusted prices |
| Strategy spikes on specific dates | unadjusted splits |
| ML model F1 > 0.9 on next-day predictions | leaking labels via overlapping windows |

A Sharpe ratio that looks too good to be true almost always is. Module 9 builds the discipline that catches these reliably.

## Bottom line

Three rules:

1. **Use a point-in-time universe.** Today's index is not yesterday's index.
2. **Adjust your prices, and know which adjustment you used.** Total-return for return series; raw for execution prices.
3. **Shift your signals by one bar.** Today's signal acts on tomorrow's bar.

Get these three right and you've eliminated 80% of the silent killers. The other 20% is in Module 9.

Continue to **[Options chain storage and queries](04-options-chains.md)**.
