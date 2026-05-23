# Module 5 — Data Engineering for Markets

This module turns the universal `Feed` adapter from Module 0 into a real data layer: a partitioned Parquet store, async batch loaders, correct handling of corporate actions and survivorship bias, an options-chain storage layout that scales, and a websocket pattern for live feeds.

Pages:

1. **[A partitioned Parquet cache](01-parquet-cache.md)** — caching feed responses so Yahoo (and you) don't get throttled.
2. **[Async batch loaders with backoff](02-async-batch.md)** — pulling hundreds of symbols in parallel without melting your or the vendor's API.
3. **[Survivorship bias, corporate actions, and point-in-time data](03-bias-and-corp-actions.md)** — the bugs that quietly destroy backtests.
4. **[Options chain storage and queries](04-options-chains.md)** — partitioning by expiry, the columns you actually need, the columns you must compute.
5. **[Live data — websockets and order events](05-live-data.md)** — keeping a hot path responsive while a backtester sleeps.

Start with **[A partitioned Parquet cache](01-parquet-cache.md)**.
