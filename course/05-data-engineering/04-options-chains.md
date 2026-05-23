# Options chain storage and queries

Options data is a different beast from equity bars. The data is multidimensional (strike × expiry × type × time), the volumes are dramatically larger (a single underlying can have 10,000+ active contracts), and the queries are different (you almost never want "all strikes, all expiries" — you want a *slice*).

This chapter is the blueprint for storing options chains so the queries that matter are fast and the queries that don't are tolerable.

## The schema

A canonical row in our store:

| Column | Type | Notes |
|---|---|---|
| `asof` | `datetime64[ns, UTC]` | snapshot time |
| `underlying` | `string` | e.g. "SPY" |
| `expiry` | `datetime64[ns]` | naive date (the close of trading that day) |
| `right` | `string` | "C" or "P" |
| `strike` | `float64` | the strike price |
| `bid` | `float64` | best bid |
| `ask` | `float64` | best ask |
| `last` | `float64` | last trade price |
| `volume` | `int64` | shares (1 contract = 100 shares) |
| `open_interest` | `int64` | |
| `implied_volatility` | `float64` | as reported by the vendor, or recomputed |
| `delta` | `float64` | if vendor provides; else computed |
| `gamma` | `float64` | |
| `vega` | `float64` | |
| `theta` | `float64` | |
| `rho` | `float64` | |
| `spot` | `float64` | underlying mid at `asof` |

The greeks are optional but worth storing — recomputing them at query time is expensive when you do it for millions of rows.

## Partitioning

The right answer depends on access patterns. For backtesting strategies that hold to expiry:

```
data/chains/
  underlying=SPY/
    expiry=2024-06-21/
      data.parquet              # every snapshot of every strike for this expiry
    expiry=2024-07-19/
      data.parquet
```

For real-time scanning that pulls today's chain across many tickers:

```
data/chains/
  asof=2024-06-03/
    underlying=SPY/
      data.parquet
    underlying=QQQ/
      data.parquet
```

The principle: **partition by the column(s) you filter on most**. Test query patterns against the layout before committing to it — a wrong choice can mean 100× slowdown.

## A query layer

```python
import polars as pl

def chain_for(underlying: str, asof_date: str, expiry: str | None = None):
    q = pl.scan_parquet(f"data/chains/underlying={underlying}/**/*.parquet")
    q = q.filter(pl.col("asof").dt.date() == pl.date(asof_date))
    if expiry:
        q = q.filter(pl.col("expiry") == pl.date(expiry))
    return q.collect()
```

Polars pushes the filters into the Parquet scan, so a query for "SPY chain on 2024-06-03 for the 2024-06-21 expiry" reads ~1 MB even if your total store is 100 GB.

## The columns you must compute

Some metadata you should compute once and store. Three high-value examples:

### Days to expiry

```python
df = df.with_columns(
    dte = (pl.col("expiry").cast(pl.Date) - pl.col("asof").cast(pl.Date)).dt.total_days()
)
```

Every scanner ever written filters on DTE. Store it.

### Moneyness

```python
df = df.with_columns(
    moneyness = pl.col("strike") / pl.col("spot"),                # 1.0 = ATM
    log_moneyness = (pl.col("strike") / pl.col("spot")).log(),
)
```

`log_moneyness` is what vol-surface models actually use.

### Mid price

```python
df = df.with_columns(
    mid = (pl.col("bid") + pl.col("ask")) / 2,
    spread_bps = ((pl.col("ask") - pl.col("bid")) / pl.col("mid") * 10_000),
)
```

The spread in basis points is the only honest sign of whether an option is tradeable. ATM SPY options run 2–5 bps; far OTM weeklies run 200+ bps.

## A worked example: today's full SPY chain

```python
from engine.data import YFinanceFeed

feed = YFinanceFeed()
chain = feed.option_chain("SPY")
calls, puts = chain.calls, chain.puts
print(f"Spot: {chain.spot:.2f}")
print(f"Expiries available: {sorted(calls['expiry'].unique())[:6]}...")

# Quick scan: 30-day call IVs around the money
import pandas as pd
today = pd.Timestamp.utcnow()
near = calls[(calls["expiry"] - today).dt.days.between(20, 40)]
atm = near.iloc[(near["strike"] - chain.spot).abs().argsort()].head(10)
print(atm[["strike", "bid", "ask", "implied_volatility"]])
```

This is what every options scanner starts with. Modules 14-15 take it from "get the chain" to "fit a vol surface and find dislocations."

## Computing missing greeks

When the vendor doesn't provide greeks, you compute them from the implied vol via Black-Scholes. The course's `engine.options` (Module 14) implements this:

```python
# Module 14 preview
from engine.options.bs import greeks

g = greeks(spot=475.0, strike=480.0, t=20/252, r=0.045, q=0.013,
           sigma=0.18, kind="call")
print(g.delta, g.gamma, g.vega)
```

`vega → 0` for deeply ITM/OTM options is the numerical edge case that bites every IV solver — Module 14 covers Brent's method as the robust fallback.

## A worked example: gamma exposure (GEX)

Dealer gamma exposure is one of the most important second-order signals in options markets. The idea: market makers hedge their book; their hedging flows are predictable from the chain.

```python
import numpy as np

def gex(chain: pd.DataFrame, spot: float, contract_size: int = 100) -> float:
    """Sum of gamma × open_interest × contract_size for the chain.
    Convention: dealers are short calls, long puts. Signs flip accordingly."""
    call_gamma = chain.loc[chain["right"] == "C", "gamma"] * chain.loc[chain["right"] == "C", "open_interest"]
    put_gamma  = chain.loc[chain["right"] == "P", "gamma"] * chain.loc[chain["right"] == "P", "open_interest"]
    # Dealers are short calls (+ exposure to upside vol), long puts (- exposure)
    return contract_size * spot * spot * 0.01 * (call_gamma.sum() - put_gamma.sum())
```

A positive GEX environment dampens moves (dealers buy dips, sell rips). A negative GEX environment amplifies them (dealers chase). Module 18 (Scanners — Frontier) builds the full dealer-positioning scanner.

## Storage volumes

A rough sanity check for SPY:

- 10 expiries × 200 strikes × 2 rights = 4,000 contracts per snapshot.
- One snapshot per minute over the trading day = ~390 snapshots.
- Per day: 1.56M rows.
- Per row: ~120 bytes uncompressed.
- Per day uncompressed: ~190 MB.
- After Parquet + zstd: ~30 MB.
- Per year (~252 trading days): ~7.5 GB.

For one symbol. Across the universe of options-active tickers (a few thousand), you're at multi-TB territory. Storage planning matters.

## Pitfalls

!!! warning "Implied vol from `last` vs `mid`"
    Vendors report IV computed from the last trade price. The last trade can be hours old and meaningless. Always recompute IV from the bid-ask mid for any analytical work.

!!! warning "American vs European pricing"
    Equity options (SPY, AAPL) are American. Index options (SPX, NDX) are European. They price differently for ITM puts (early exercise). Use a binomial / LSMC for American (Module 14); BS is fine for European.

!!! warning "Settlement times — AM vs PM"
    Quarterly index options settle at the *open*, not the close. Weekly index options settle at the close. Mixing them silently breaks gamma calculations near expiry.

!!! warning "Expiration time"
    "Expiry 2024-06-21" means market close on that date for equity options, but it's sometimes the open for index options (AM settlement). Store the expiry *timestamp*, not just the date, if you care about hourly mechanics.

## Bottom line

- Schema: spot, OHLCV, OI, IV, greeks, computed `dte`/`moneyness`/`mid`/`spread_bps`.
- Partition by your dominant filter — `(underlying, expiry)` for backtests, `(asof, underlying)` for live scanners.
- Always work from the **bid-ask mid**, never the last.
- Compute greeks once at ingest; don't pay the BS cost on every query.
- Plan for tens of GB per liquid symbol per year.

Continue to **[Live data — websockets and order events](05-live-data.md)**.
