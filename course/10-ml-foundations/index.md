# Module 10 — ML Foundations for Trading

Standard ML tutorials don't prepare you for financial data. The signal-to-noise ratio is two orders of magnitude lower than in most ML benchmarks, the labels are autocorrelated, the distribution is non-stationary, and your prediction's calibration matters as much as its rank order. This module is what you need to know to do ML on markets without getting hurt.

Pages:

1. **[The bias-variance picture in low-SNR finance](01-bias-variance.md)** — why "more data" doesn't always help, and why you should bias hard toward simple models.
2. **[Feature engineering for markets](02-feature-engineering.md)** — fractional differentiation, triple-barrier labels, meta-labeling; uses `engine.features` directly.
3. **[Time-series cross-validation in practice](03-time-series-cv.md)** — putting `PurgedKFold` and `WalkForward` (Module 9) to work for ML.
4. **[Calibration — Platt, isotonic, and why you need it](04-calibration.md)** — turning model scores into probabilities you can size against.
5. **[Conformal prediction with MAPIE](05-conformal.md)** — honest prediction intervals on the next bar's move.
6. **[An end-to-end ML signal pipeline](06-pipeline.md)** — features → labels → CV → fit → calibrate → conformal-size → backtest.

Start with **[The bias-variance picture in low-SNR finance](01-bias-variance.md)**.
