# Module 18 — Scanners

A **scanner** answers the question "out of my universe, which symbols look interesting *right now*?". Every quant shop has scanners; the difference between an OK one and a great one is what you scan for. This module is the catalogue of production scanners (the ones that ship at real funds) and frontier scanners (the ones that exploit research from Module 17).

The course's `engine.scanners` package provides the framework: a `Scanner` Protocol, a `@scanner("name")` decorator, and a `registry` that auto-collects them. Every scanner in this module follows the same shape.

Pages:

1. **[The Scanner framework](01-framework.md)** — Protocol, registry, the runtime loop.
2. **[Production scanners](02-production.md)** — unusual options volume, IV rank, post-earnings drift candidates, ETF NAV-arb, sector relative-strength.
3. **[Frontier scanners](03-frontier.md)** — dealer GEX inflection, vol-smile dislocation (SVI residuals), Hawkes intensity, TDA fragility, transfer-entropy lead-lag.
4. **[Microstructure scanners](04-microstructure.md)** — Kyle's lambda, VPIN spikes, queue-position deterioration.
5. **[Composing scanners into a daily report](05-composition.md)** — the daily-watchlist pattern.

Start with **[The Scanner framework](01-framework.md)**.
