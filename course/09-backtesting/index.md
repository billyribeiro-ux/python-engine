# Module 9 — Backtesting that doesn't lie

A backtest is a hypothesis: "had I run this strategy in the past, this is what would have happened." Almost every retail backtest tested against the same period it was designed on is a lie of some flavour — leakage, survivorship, optimistic costs, selection bias. This module is the practical discipline of running backtests that, when you put the strategy live, agree with what you saw.

Pages:

1. **[The leakage bestiary](01-leakage.md)** — every common way backtests lie, with examples.
2. **[A vectorised backtester in 30 lines](02-vectorized-engine.md)** — the simplest correct engine; ships in `engine.backtest`.
3. **[An event-driven engine](03-event-driven.md)** — when vectorised isn't enough; matching engines, order types, partial fills.
4. **[Cross-validation that doesn't leak](04-cv.md)** — purged k-fold, embargo, walk-forward; ships in `engine.backtest.cv`.
5. **[Combinatorial purged CV and the Deflated Sharpe Ratio](05-cpcv-dsr.md)** — the López de Prado backtesting hygiene.
6. **[Slippage and the cost of trading](06-slippage.md)** — linear, square-root, and Almgren-Chriss models.

Start with **[The leakage bestiary](01-leakage.md)**.
