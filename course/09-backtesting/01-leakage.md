# The leakage bestiary

A non-exhaustive catalogue of the ways backtests lie. Every one of these has happened to me; every one of these has happened at funds you've heard of. Internalising the list is half of the defence.

## 1. The same-bar signal

```python
# WRONG
signal = momentum(returns)
pnl = signal * returns
```

You computed `signal` from `returns[:t]` (because that's how `momentum` looks back), then applied it to `returns[t]`. The signal *knows* today's return. The single most common bug, and the one that produces "too-good-to-be-true" Sharpes of 3 or more on otherwise reasonable strategies.

Fix:

```python
pnl = signal.shift(1) * returns
```

The course's vectorised backtester (`engine.backtest.vectorized_backtest`) shifts by default so this bug is structurally impossible if you use it.

## 2. Centred rolling windows

```python
ma = close.rolling(window=20, center=True).mean()    # uses 10 future bars
sig = (close > ma).astype(int)
```

`center=True` puts half the window in the future. Useful for offline smoothing of *finished* historical data; never for signals.

Same trap: smoothing splines, wavelet denoising on the whole series, Hodrick-Prescott filters. Anything that "uses both sides" leaks unless explicitly causalised.

## 3. Volatility scaling with today's vol

```python
vol = returns.rolling(60).std()                       # includes today
position = signal / vol                                # uses today's vol to size today
```

The vol estimate ends at today's bar. The position is then applied to today's return. Subtle leak. The fix is the same `.shift(1)`:

```python
position = signal / vol.shift(1)
```

## 4. Cross-validation that crosses time

Standard scikit-learn `KFold` shuffles indices and splits randomly. For time-series labels with autocorrelation, "random" folds let test samples sit next to train samples — labels overlap, models memorise.

Use **purged k-fold with embargo**:

```python
from engine.backtest import PurgedKFold

cv = PurgedKFold(n_splits=5, purge=label_horizon, embargo=label_horizon)
for train_idx, test_idx in cv.split(len(returns)):
    ...
```

The `purge` removes train samples whose label horizon overlaps the test window. The `embargo` blocks training resuming immediately after the test window. We cover this in detail in chapter 4.

## 5. Survivorship bias

Universe = "current S&P 500". Past constituents that went to zero are absent. Strategy that picks losers and shorts them looks great because the historical losers — who would have continued to lose — were already removed. Module 5, chapter 3 is the data-side fix; the backtest side is "use a point-in-time universe always."

## 6. Look-ahead in fundamentals

Earnings, revenue, balance sheet items — all reported with a lag and frequently restated. Using the *current* fundamentals database to backtest a strategy from 5 years ago gives you data that nobody had at the time. The fix is the same as survivorship: point-in-time storage with explicit `release_date`.

## 7. Over-optimistic fills

```python
# implicit assumption: you fill at the close of the bar in which you decided
fill_price = bar.close
```

In live trading, you decided at the close, but your order didn't fill at the close — it filled somewhere on the next bar's first prints, almost always at a worse price. The fix:

```python
fill_price = next_bar.open
# Better: model intrabar slippage explicitly (chapter 6)
```

## 8. Strategy-on-strategy selection bias

You generated 1000 strategies, picked the best. The best one's *backtested* Sharpe is, by construction, much higher than its *true* Sharpe. The Deflated Sharpe Ratio (chapter 5) is the formal correction; the practical discipline is "never deploy your best in-sample strategy without out-of-sample verification first."

## 9. Implicit re-training

```python
def signal(close):
    return some_model.predict(features(close))
```

If `some_model` was trained on the full history including the part you're testing, the test is contaminated. **Always retrain models within the CV fold**.

## 10. Forward-fill that fills with the future

```python
# YOU intended: forward-fill within each day's first available value
prices.fillna(method="ffill")
```

`ffill` fills with the most recent non-NaN value *before* the NaN — that's fine. But people often instead `pad` or `interpolate`, which can pull future values backward. Read the docs of your fill method.

## 11. Returns vs adjusted returns

We covered this in Module 5: a stock split looks like a -50% return if you use raw close. A dividend looks like a -1% return. Strategies that "buy on -50% moves" look great in backtests run on raw prices. They are buying nothing real.

Always: total-return adjusted prices for return computation; raw prices for execution prices.

## 12. Universe leak

Your "tradeable universe" was filtered by "average volume in 2024 > 1M shares". A name that was illiquid in 2018 but is liquid in 2024 is in your 2018 backtest universe but wouldn't have been tradeable. The universe filter, like fundamentals, must be point-in-time.

## 13. Holiday and corporate-action gaps

Markets closed; index doesn't change. Your code computes a return between two adjacent bars across the gap and gets a misleading number. The fix is to compute returns only between *trading days* in the same regime — your index should be the trading calendar.

## 14. Bid-ask vs mid

For execution prices, use bid-ask. For analysis, use mid. Mixing them silently — analysing on mid, "filling" at mid — assumes a zero-spread world.

## 15. Margin / borrow / hard-to-borrow

Backtest says you can short LULU at -2% on Tuesday. In practice that morning LULU is hard-to-borrow at 8% per annum carrying cost. For long-short strategies, **factor borrow into the cost model**. Most equity-shorting strategies' backtested edge is partially or entirely borrow cost.

## 16. Earnings overlap

Your strategy holds a name through earnings. Earnings move the stock 5%. The strategy's "edge" is half just being on the right side of earnings. Filter out positions that would span an earnings announcement — your strategy isn't an earnings predictor.

## 17. The same bug in your factor and your test

You built a momentum factor with a subtle look-ahead. You then "controlled" your strategy's performance with that factor in a Fama-MacBeth regression. The control absorbs the same fictitious alpha, making the strategy look like it has none. (Or, the opposite: the factor's look-ahead amplifies a weak signal.) The fix: build factor controls with the same discipline you'd apply to a strategy.

## 18. Multiple-testing without correction

You tested 50 hyperparameters. The best one had a Sharpe of 1.3. The honest interpretation: "in a sample of 50 random strategies under the null, you'd expect the best to have a Sharpe in the 1.0-1.5 range." Module 6 has the DSR formalism; the practical rule is **walk-forward, no peeking, no tuning on the holdout**.

## 19. Inflated capacity

A backtest with $10M of capital might be tradeable; the same backtest with $1B is fantasy because your orders move the market. Estimate **capacity** — at what AUM does the marginal Sharpe drop to your target? The Almgren-Chriss model (chapter 6) gives one estimate; bootstrap-and-resize is another.

## 20. The "I just ran a backtest" fallacy

A single backtest is a single sample from a distribution. The point estimate is almost worthless on its own. Bootstrap the Sharpe, report a CI, and verify with walk-forward. Module 6 covers the statistical machinery.

## The checklist

For every backtest you ship:

- [ ] Signal lagged 1 bar before applying to returns.
- [ ] All rolling/EWM is causal (no `center=True`).
- [ ] Point-in-time universe and fundamentals.
- [ ] Total-return-adjusted prices.
- [ ] Realistic slippage (chapter 6).
- [ ] Fills at next-bar open (or worse).
- [ ] Borrow cost modelled for shorts.
- [ ] Purged + embargoed CV with horizon-sized purge.
- [ ] Walk-forward over a 30%+ holdout untouched until the final test.
- [ ] DSR computed against the trial set.
- [ ] Capacity estimate.
- [ ] Bootstrap CI for Sharpe.

If any of those is missing, don't deploy.

Continue to **[A vectorised backtester in 30 lines](02-vectorized-engine.md)**.
