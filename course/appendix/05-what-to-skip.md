# What I'd skip if I had to

Opinions on chapter priorities. If you're starting from zero and have limited time, this is the order I'd suggest. If you're already a working quant, you can skim aggressively.

## The mandatory reading

If you read nothing else, these are the chapters that change how you work:

- **Module 0 — Orientation, entire**. The data adapter is the seam everything depends on.
- **Module 1, chapter 1 (data model)** and **chapter 4 (concurrency)**. Without these, the rest of Python feels like magic.
- **Module 3 — NumPy entirely**. Pandas/Polars assume NumPy fluency.
- **Module 9 — Backtesting entirely**. The leakage bestiary is the most important chapter in the course.
- **Module 10 — ML Foundations entirely**. Especially the bias-variance-in-low-SNR mindset.
- **Module 20, chapter 1 (drawdowns)** and **chapter 3 (Kelly trap)**.
- **Module 21, chapter 1 (paper trading)**.

That's about 25 chapters — roughly a focused weekend of reading. After it, you have the L7+ engineering mindset.

## The strategy-specific deep dives

Pick based on what you're actually trying to ship:

**You want a multi-asset systematic macro program (TS momentum, carry, etc.)**:
- Module 8 chapters 2 (momentum) and 6 (sizing).
- Module 16 chapter 1 (HMM overlay) and chapter 4 (carry).

**You want to do equity pairs trading**:
- Module 7 chapter 3 (Kalman) and Module 8 chapter 4 (cointegration).
- Module 16 chapter 2 (Kalman pairs production).

**You want to do ML-driven equity prediction**:
- Module 10 entire.
- Module 11 chapters 1 (GBM) and 2 (stacking).
- Module 16 chapter 3 (GBM + conformal).

**You want to trade options**:
- Modules 14 and 15 entire.
- Module 16 chapter 6 (calendar spreads).
- Module 18 chapter 3 (frontier scanners including GEX).

**You want to do market making or execution**:
- Module 19 entire.
- Module 13 chapter 4 (Avellaneda-Stoikov as RL).
- Module 13 chapter 5 (Almgren-Chriss).

## What I'd skip on a first read

Genuinely advanced material that doesn't pay back unless you have a specific need:

- **Module 7 chapter 5 (Wavelets and Hilbert)** — interesting but rarely needed.
- **Module 11 chapter 4 (Survival analysis)** — useful only if you have explicit trade-duration questions.
- **Module 11 chapter 5 (Causal inference)** — important conceptually; in practice, most quants get by without explicit DoWhy/EconML.
- **Module 12 chapter 4 (Mamba/SSM)** — frontier; you can revisit when SSMs are mainstream.
- **Module 12 chapter 6 (Diffusion models)** — likewise.
- **Module 13 chapter 6 (Offline RL)** — for serious RL deployments only.
- **Module 17 entire** — read after Module 16 is producing real results; otherwise it's research material.
- **Module 19 chapter 6 (RL execution)** — for serious institutional execution.

## What I'd revisit annually

- **Module 9 chapter 1 (leakage bestiary)** — re-read once a year to keep the checklist fresh.
- **Module 21 chapter 6 (postmortems)** — re-read after any incident.

## What I'd read at the end as a "victory lap"

Once you have a strategy live and earning:

- **Module 17 entire** — the frontier becomes relevant when you're hunting for incremental edge.
- **Appendix's reading list** — the books are where the depth lives.

## The honest take

The course is long because the field is large. Almost nobody needs every module. Pick the path that matches your goal; skip the rest; come back when your goal changes.

The Python and engineering chapters compound — they pay off in every project. The strategy chapters are specific — they pay off when you're working on that strategy.

## And finally

Trading is hard. Most strategies that look promising don't survive deployment. The hit rate for "research a new strategy → ship it live" is maybe 1 in 5, even at top firms. Don't be discouraged when several attempts fail; that's the field, not you.

The discipline that wins is **slow, methodical, leakage-resistant research, paired with conservative deployment, kill switches, and honest postmortems**. Glamorous techniques matter less than that discipline.

Good luck.

## End of the course

You've reached the end. **The full course is now 21 modules, 130+ pages, and roughly 130,000 words of dense, working content** — about the length of two technical books. Every code block in the `engine` package has tests; every chapter's principal examples have been verified to run.

The repository at `claude/python-trading-ml-course-3VRQB` is the canonical artefact. Build the HTML site with `make serve`. Build the PDF with `make pdf`. Bookmark the search. Come back to it.

Now go build something.
