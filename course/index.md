# The Python Engine

> A distinguished-engineer course on Python, machine learning, and trading — covering both stocks and options.

This course exists because most "Python for trading" material is one of two things: a beginner tutorial wrapped in TradingView screenshots, or a wall of math that never tells you what to actually type. This one is the third thing — what a senior or principal engineer at a real quant shop would teach you over a long sabbatical. It is meant to be **read end-to-end at least once** before being used as a reference, because the chapters compound.

## Two tracks, one course

Every advanced module ships in two flavours:

- **Production track** — the methods that genuinely ship at quant funds today. Robust, well-understood, the boring stuff that prints money. HMMs for regime overlays, Kalman pairs, gradient boosting with conformal sizing, deep hedging at the conservative end.
- **Frontier track** — the methods almost no retail trader has touched and many funds haven't either. Hawkes processes for self-exciting event clusters, topological data analysis (persistent homology) on rolling correlation matrices, transfer-entropy networks for lead-lag discovery, neural SDEs, RL for optimal execution and sizing, diffusion models for synthetic chain stress tests, signature methods on rough paths, GNNs on cross-asset graphs, meta-learning for fast adaptation across symbols.

You can read either track in isolation, but you'll get more from doing both: the frontier is a lot less scary when you have shipped a vanilla version first.

## How to read this site

- Use the **sidebar** to navigate. Every code block has a **copy** button in the top right.
- Press <kbd>/</kbd> or click the **search** icon to fast-jump to any concept.
- Toggle **dark mode** with the moon/sun icon in the header.
- A **single PDF** of the entire course is built with `make pdf` and lives at `site/pdf/python-engine-course.pdf` after the build.

## The course map

### Foundations

- **[Module 0 — Orientation](00-orientation/index.md)** — how to read this, environment setup, the universal data adapter, conventions.
- **[Module 1 — Python Foundations](01-python-foundations/index.md)** — the data model, descriptors, dunder methods, generators, async/threads/processes, typing, dataclasses.
- **[Module 2 — Python Hacks](02-python-hacks/index.md)** — iteration tricks, `collections`, profiling, plugin patterns.
- **[Module 3 — NumPy Mastery](03-numpy-mastery/index.md)** — strides, broadcasting, `einsum`, sliding windows, numerical stability.

### Data and analytics (Phase 2)

- Module 4 — pandas + polars
- Module 5 — Data engineering for markets
- Module 6 — Statistics + probability for traders
- Module 7 — Time series

### Quant core (Phase 3)

- Module 8 — Classical quant (factor models, momentum, pairs, OU)
- Module 9 — Backtesting that doesn't lie
- Module 10 — ML foundations for trading

### Modern ML and DL (Phases 4–5)

- Module 11 — Modern ML (XGBoost, LightGBM, conformal, stacking)
- Module 12 — Deep learning for time series (PatchTST, N-HiTS, Neural SDEs)
- Module 13 — Reinforcement learning
- Module 14 — Options foundations
- Module 15 — Vol surface + advanced options (SVI, SABR, Heston, deep hedging)

### Strategies and infrastructure (Phases 6–7)

- Module 16 — Production strategies
- Module 17 — Frontier strategies
- Module 18 — Scanners
- Module 19 — Execution and microstructure
- Module 20 — Risk and portfolio
- Module 21 — Deployment
- Appendix — math refreshers, reading list, glossary, common mistakes

## A note on honesty

Three things this course will not do:

1. **Promise returns.** Markets are non-stationary low-SNR adversarial systems. Any course that promises specific edges is selling something. We teach you to find your own.
2. **Hide the hard parts.** Backtest leakage, multiple-testing, slippage, sim-to-real gaps, regime breaks — they get their own chapters. You will read them.
3. **Pretend the exotic methods are universally better.** They aren't. A gradient-boosted classifier with proper CV will beat a poorly-tuned transformer 9 times out of 10. We do both, and we tell you when each one is the right tool.

Let's go. Start at **[Module 0 — Orientation](00-orientation/index.md)**.
