# Module 16 — Production Strategies

This module is the **Production track**: the strategies that actually ship at real quant funds, year after year, decade after decade. They are not glamorous. They are not new. They print money because they capture genuine risk premia or persistent behavioural inefficiencies — and they survive deployment because everyone running them takes correctness seriously.

Every strategy in this module is built from pieces you already have: the universal data adapter (Module 0), feature engineering (Module 10), the backtest engine (Module 9), the model layers (Modules 10-11), and the options machinery (Modules 14-15). The novelty is the *combination* — and the discipline of layering risk overlays, regime conditioning, and conformal sizing on top of a base signal.

Pages:

1. **[HMM regime overlay on momentum](01-hmm-regime.md)** — when to trade and when not to.
2. **[Kalman pairs trading at production grade](02-kalman-pairs.md)** — with proper risk management.
3. **[Gradient-boosted next-day classifier with conformal sizing](03-conformal-gbm.md)** — the ML-driven workhorse.
4. **[Vol-targeted carry across asset classes](04-carry.md)** — equity carry, FX carry, term-premium carry.
5. **[Post-earnings drift with bias controls](05-pead.md)** — one of the oldest documented anomalies, still alive.
6. **[Calendar and diagonal spreads on IV term-structure](06-calendar-spreads.md)** — an options-only production strategy.

Start with **[HMM regime overlay on momentum](01-hmm-regime.md)**.
