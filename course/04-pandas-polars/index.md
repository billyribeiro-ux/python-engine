# Module 4 — pandas + polars

You will spend most of your data-engineering and research time inside one of these two libraries. Pandas is the lingua franca of quantitative Python; polars is the faster, stricter, increasingly serious competitor. Knowing both — and knowing when each is right — is table-stakes for L7+ work.

Pages:

1. **[Index, MultiIndex, and the seven sins of pandas](01-index-multiindex.md)** — the index is half of pandas; using it well changes everything.
2. **[Joins, `merge_asof`, and tick alignment](02-joins-asof.md)** — how to combine market data without breaking it.
3. **[Rolling, expanding, and `ewm`](03-rolling.md)** — moving statistics done correctly.
4. **[GroupBy: `apply` vs `agg` vs `transform`](04-groupby.md)** — the three operations, when each is right, and the performance trap.
5. **[Polars from a pandas user's perspective](05-polars.md)** — expressions, lazy frames, when to switch.
6. **[Arrow, Parquet, DuckDB — the fast triangle](06-arrow-parquet-duckdb.md)** — the storage and query layer underneath modern data work.

Start with **[Index, MultiIndex, and the seven sins of pandas](01-index-multiindex.md)**.
