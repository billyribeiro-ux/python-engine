# Composing scanners into a daily report

You have 15 scanners running. Each produces 10-50 candidate symbols. Across all of them, you have hundreds of "hits" but only time to look at a dozen. The composition layer turns scanner output into a prioritised, deduplicated, actionable daily report.

This chapter is the final synthesis of Module 18.

## The pattern

```
all scanners (production + frontier + microstructure)
                       │
                       ▼
        unified candidate table per scanner
                       │
                       ▼
        per-symbol aggregation (count, weighted score)
                       │
                       ▼
        promotion to watchlist (top-N)
                       │
                       ▼
        per-symbol context page (one screen per candidate)
                       │
                       ▼
        human review (or auto-trade)
```

## The aggregation layer

```python
import pandas as pd
import numpy as np
from collections import defaultdict
from engine.scanners import registry


def daily_aggregation(universe: list[str], asof: pd.Timestamp,
                        scanner_names: list[str] | None = None,
                        weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Run every scanner; aggregate per symbol; rank.

    weights: optional per-scanner weight. Default = 1.0.
    """
    if scanner_names is None:
        scanner_names = registry.names()
    if weights is None:
        weights = {n: 1.0 for n in scanner_names}

    per_symbol = defaultdict(lambda: {"score": 0.0, "sources": [], "metadata": []})
    for name in scanner_names:
        try:
            scn = registry.get(name)()
            df = scn.scan(universe, asof)
            if df.empty or "symbol" not in df.columns:
                continue
            for _, row in df.iterrows():
                sym = row["symbol"]
                per_symbol[sym]["score"] += weights.get(name, 1.0)
                per_symbol[sym]["sources"].append(name)
                per_symbol[sym]["metadata"].append(row.drop("symbol").to_dict())
        except Exception as e:
            print(f"scanner {name} failed: {e}")

    out = pd.DataFrame([
        {"symbol": sym, "score": data["score"],
         "n_sources": len(data["sources"]),
         "sources": ", ".join(data["sources"]),
         "metadata": data["metadata"]}
        for sym, data in per_symbol.items()
    ]).sort_values(["score", "n_sources"], ascending=False)
    return out
```

The output: one row per symbol with all the sources that flagged it. Sorted by weighted score.

## Custom scoring

For different strategies, weight scanners differently:

```python
# For a directional momentum book
momentum_weights = {
    "sector-rs": 2.0,
    "pead-candidates": 2.0,
    "unusual-options-volume": 1.5,
    "te-leadlag": 1.0,
    "hawkes-spike": 0.5,
}

# For a vol-trading book
vol_weights = {
    "iv-rank": 2.0,
    "implied-vs-realised": 2.0,
    "svi-dislocation": 1.5,
    "gex-inflection": 1.5,
    "vpin-spike": 1.0,
    "spread-expansion": -1.0,    # negative: spread expansion is a risk-off signal
}

# For execution
execution_weights = {
    "spread-expansion": -2.0,
    "vpin-spike": -1.5,
    "kyles-lambda": -1.0,        # high lambda = avoid trading
}
```

## Per-candidate context pages

For each high-score symbol, generate a one-screen "context page":

```python
def candidate_context(symbol: str, aggregated_row: pd.Series, asof: pd.Timestamp) -> str:
    """One-screen markdown summary for human review."""
    lines = [
        f"# {symbol}",
        f"**Score**: {aggregated_row['score']:.1f} (across {aggregated_row['n_sources']} scanners)",
        f"**Sources**: {aggregated_row['sources']}",
        "",
        "## Per-scanner details",
    ]
    for meta in aggregated_row["metadata"]:
        lines.append("- " + ", ".join(f"{k}={v}" for k, v in meta.items()))
    return "\n".join(lines)
```

Generate the markdown, save to a daily folder, push to your team's channel. The human reviewer makes the trade decision (or routes to an auto-trader).

## End-to-end daily script

```python
import pandas as pd
import json
from pathlib import Path
from engine.data import YFinanceFeed, ParquetCache
from engine.scanners import registry


def run_daily_report(universe: list[str], output_dir: str = "daily_reports"):
    asof = pd.Timestamp.utcnow().normalize()
    feed = ParquetCache(YFinanceFeed(), root="data/bars")     # warmed cache
    aggregated = daily_aggregation(universe, asof,
                                     scanner_names=registry.names(),
                                     weights=momentum_weights)
    out_dir = Path(output_dir) / asof.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the aggregated table
    aggregated.head(50).to_csv(out_dir / "top_50.csv", index=False)
    # Save context pages for the top 20
    top = aggregated.head(20)
    for _, row in top.iterrows():
        md = candidate_context(row["symbol"], row, asof)
        (out_dir / f"{row['symbol']}.md").write_text(md)
    # Slack/email digest (skeleton)
    digest = "\n".join(f"- {r['symbol']} (score {r['score']:.1f}): {r['sources']}"
                       for _, r in top.iterrows())
    print(digest)


if __name__ == "__main__":
    run_daily_report(["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
```

## Asynchronous scanning at scale

For a 500-symbol universe with 15 scanners, sequential execution takes minutes. Async parallelisation cuts to seconds:

```python
import asyncio


async def run_scanner_async(name, universe, asof):
    cls = registry.get(name)
    return name, await asyncio.to_thread(cls().scan, universe, asof)


async def parallel_aggregate(universe, asof, scanner_names):
    tasks = [run_scanner_async(n, universe, asof) for n in scanner_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(r for r in results if isinstance(r, tuple))


# Run
asyncio.run(parallel_aggregate(universe, asof, registry.names()))
```

## Risk filters on the aggregated output

Before promoting any candidate to a trade:

- **Liquidity check**: ADV > some minimum.
- **Borrow check**: if you'd short, is the name borrowable?
- **Earnings check**: any earnings within N days?
- **Recent move check**: skip if the symbol moved > 10% today (chase risk).

```python
def filter_for_tradeability(candidates: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    out = []
    for _, row in candidates.iterrows():
        sym = row["symbol"]
        adv = fetch_adv(sym)
        borrow = fetch_borrow_rate(sym)
        earnings = fetch_next_earnings(sym, asof)
        recent_move = fetch_today_return(sym, asof)
        if adv < 50_000_000: continue
        if abs(recent_move) > 0.10: continue
        if earnings and (earnings.date - asof.date()).days < 3: continue
        out.append({**row.to_dict(), "adv": adv, "borrow": borrow,
                     "earnings_in_days": (earnings.date - asof.date()).days if earnings else None})
    return pd.DataFrame(out)
```

## Monitoring over time

Save the daily aggregated output to a Parquet store. After 6 months, you have a record of:

- Which scanners produced the most hits.
- Which scanner combinations produced the best ex-post P&L.
- Which symbols were flagged repeatedly without ever paying off (filter them out).

This longitudinal data lets you refine the weighting and the scanner set empirically.

## Pitfalls

!!! warning "Selection bias from look-back tuning"
    Don't tune weights on historical hit rates → forward-walk these. Otherwise the daily report looks great in backtest and underperforms live.

!!! warning "Scanner dependence**
    Two scanners that share an input (both use options chain data, say) aren't independent. A "5 scanners flagged it" symbol where 3 of those use the same data isn't 5× the evidence — it's closer to 2-3×.

!!! warning "Information staleness"
    Daily reports computed at midnight UTC are stale by US market open. Refresh once more at the open for highest-quality signals.

!!! warning "Alert fatigue**
    A daily report with 100 candidates means nobody reads any. Cap at 10-20 entries and trust the prioritisation.

## End of Module 18

You now have the full scanner stack: framework, production scanners, frontier scanners, microstructure scanners, and the composition layer that turns them into actionable daily output. The next module — **Execution and Microstructure** — goes deeper into the underlying order-book mechanics that the microstructure scanners only scratched.

Continue to **[Module 19 — Execution and Microstructure](../19-execution-and-microstructure/index.md)**.
