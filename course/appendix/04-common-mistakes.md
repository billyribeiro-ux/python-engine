# Common mistakes you'll make

A consolidated list of the failure modes from across the course. You'll make some of these. The goal is to make them less catastrophic by knowing what to look for.

## Data and engineering

1. **Storing dates as strings**. Always coerce to `pd.Timestamp` / `datetime64[ns, tz]` at the boundary.
2. **Mixing timezones**. Always UTC. Coerce at the data adapter.
3. **Object-dtype columns in pandas**. Kill any column with `dtype("object")` unless you really need it.
4. **`df.append` in a loop**. Build a list, then `pd.concat`.
5. **`iterrows`** for anything non-trivial. Vectorise or use `itertuples`.
6. **In-place mutation** (`df.sort_values(..., inplace=True)`). Return a new frame.
7. **Forgetting to sort before `merge_asof`**. The error message is clear; the bug is silent if you ignore it.
8. **Cache invalidation**. A cache that doesn't know to refresh on new data lies to you.

## Backtesting and ML

9. **Same-bar signal**. `signal × returns` without `signal.shift(1)`. The single most common bug; immediately invalidates the backtest.
10. **Centred rolling windows** (`center=True`). Uses future data.
11. **Survivorship bias** in the universe.
12. **Restated fundamentals** instead of point-in-time.
13. **Random k-fold CV** on time-series labels. Use `PurgedKFold`.
14. **No purge gap** between train and test. Even with kfold, contiguous train/test leak through label horizons.
15. **Hyperparameter tuning on the holdout**. Selection bias.
16. **No DSR adjustment** for a parameter sweep. The "winner" is upward-biased.
17. **Sample-Sharpe sizing**. Use a deflated or shrunk Sharpe for Kelly.
18. **Calibration on training data**. Calibrate inside the CV fold.
19. **Forgetting to standardise inside a Pipeline**. Scalers fit on the full data leak test stats.

## Options-specific

20. **Pricing in days when the formula wants years**. `T = days / 365`.
21. **Vol as 20 instead of 0.20**. Almost always a typo bug.
22. **Black-Scholes on American options**. Use binomial.
23. **IV from last-trade prices**. Use bid-ask mid.
24. **Forgetting put-call parity** in surface stripping. Recovered surface is asymmetric where it shouldn't be.
25. **Discrete dividends ignored** for single-stock options near a div date.
26. **Pin risk** — closing options the day after expiry instead of the day before.
27. **Mixing settlement times** for AM- vs PM-settled options.

## Strategy and risk

28. **Sizing for backtest Sharpe**. Live will be 20-40% lower; size for the lower.
29. **Vol-targeting + drawdown overlay stacked carelessly**. The strategy can hibernate forever during turbulent regimes.
30. **No earnings filter** for stock pairs / drift strategies.
31. **No borrow cost** for short legs.
32. **No hard stop** on extreme spread moves in pairs trading.
33. **Trading too small a universe** → no diversification.
34. **Trading too large a universe** → spurious signals after correction.
35. **Re-tuning after every bad day**. Over-tuning into noise.

## Execution

36. **Market orders on illiquid names**. Use limits + a time stop.
37. **Naive impact assumption** (linear cost) for big orders. Use square-root.
38. **Re-quoting on every tick**. Use a minimum-move threshold.
39. **No queue-position tracking** for resting limit orders.
40. **Trading at the open or close** without considering microstructure noise.

## Deployment

41. **Skipping paper trading**. Live is the worst place to discover a bug.
42. **No idempotency keys**. One TCP retry = doubled position.
43. **No position reconciliation on startup**.
44. **No kill switches**, or kill switches that depend on the strategy code working.
45. **Logs without structure**. Plain `print()` is unsearchable at 3am.
46. **No daily reports**. By the time someone notices the drift, it's a month deep.
47. **Re-tuning a live strategy mid-day**. The bug is usually elsewhere.

## Statistics

48. **Reporting Gaussian VaR** for fat-tailed returns. Under-estimates the tail.
49. **Bootstrap on time-series** without block-sampling. Underestimates variance.
50. **P-value < 0.05 → real**. Only after multiple-testing correction.
51. **Correlation = causation**. Many "predictive" signals are confounded by a common cause.
52. **Cointegration in-sample**. Out-of-sample test is mandatory.

## Software

53. **No tests**. Trading software with no tests is throwing dice.
54. **No version pinning**. Library updates break things subtly.
55. **Magic-number hyperparameters scattered through code**. Centralise in config.
56. **Print-debugging instead of logging**. Print evaporates; logs persist.

## Behaviour

57. **Mid-incident strategy tinkering**. Pause first; investigate; act.
58. **Hiding losses from investors**. Always proactive.
59. **Believing the strategy is special**. Most strategies have decay; budget for it.

The complete list is longer. Every quant has their own catalogue of "I will not do that again." Yours will grow over time.

Continue to **[What I'd skip if I had to](05-what-to-skip.md)**.
