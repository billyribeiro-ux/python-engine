# Production scanners

The scanners in this chapter are the ones that genuinely show up in production at quant funds and active hedge funds — boring, robust, slowly-decaying alpha sources. Most run daily on a universe of 500-5000 names and produce a watchlist of 20-50 candidates for human review or fully-automated trading.

Each is presented with the working implementation pattern. They're all built on top of `engine.scanners`.

## 1. Unusual options volume

**The signal**: today's options volume is much higher than recent average, suggesting informed positioning.

**The metric**: `today_volume / 20_day_average_volume`. Combined with `volume / open_interest` (high values = new positioning, not exit).

```python
import pandas as pd
import numpy as np
from engine.scanners import scanner


@scanner("unusual-options-volume")
class UnusualOptionsVolumeScanner:
    def __init__(self, vol_ratio_threshold: float = 3.0, vol_oi_threshold: float = 0.5):
        self.vol_ratio_threshold = vol_ratio_threshold
        self.vol_oi_threshold = vol_oi_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            chain_today = fetch_chain(sym, asof)
            chain_history = fetch_chain_history(sym, asof - pd.Timedelta(days=20), asof)
            avg_volume = chain_history.groupby("strike")["volume"].mean()
            for _, row in chain_today.iterrows():
                avg = avg_volume.get(row["strike"], 0)
                if avg == 0:
                    continue
                ratio = row["volume"] / avg
                vol_oi = row["volume"] / row["open_interest"] if row["open_interest"] > 0 else float("inf")
                if ratio > self.vol_ratio_threshold and vol_oi > self.vol_oi_threshold:
                    rows.append({"symbol": sym, "strike": row["strike"], "expiry": row["expiry"],
                                  "right": row["right"], "volume": row["volume"],
                                  "volume_ratio": ratio, "vol_oi": vol_oi})
        return pd.DataFrame(rows).sort_values("volume_ratio", ascending=False)
```

This produces a list of contracts (not just symbols) that traded much more than usual. Often correlates with upcoming corporate actions or insider-positioning.

## 2. IV Rank / IV Percentile

**The signal**: current implied volatility is unusually low or high vs the past year. Useful for vol-selling (high IV) or vol-buying (low IV) strategies.

**Metric**: `IV_rank = (current_IV - min_IV_1yr) / (max_IV_1yr - min_IV_1yr) * 100`. Percentile is similar but uses the rank.

```python
@scanner("iv-rank")
class IVRankScanner:
    def __init__(self, lookback_days: int = 252, min_rank: float = 80.0):
        self.lookback_days = lookback_days
        self.min_rank = min_rank

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            iv_history = fetch_atm_iv_history(sym, asof - pd.Timedelta(days=self.lookback_days), asof)
            if len(iv_history) < 60:
                continue
            current = iv_history.iloc[-1]
            iv_rank = (current - iv_history.min()) / (iv_history.max() - iv_history.min()) * 100
            iv_percentile = (iv_history < current).mean() * 100
            if iv_rank >= self.min_rank or iv_rank <= (100 - self.min_rank):
                rows.append({"symbol": sym, "iv": current, "iv_rank": iv_rank,
                              "iv_percentile": iv_percentile})
        return pd.DataFrame(rows).sort_values("iv_rank", ascending=False)
```

For vol-selling strategies, target high IV rank (>80). For vol-buying (calendar spreads, debit spreads), low IV rank (<20).

## 3. Implied move vs realised move gap

**The signal**: the options market is over- or under-pricing the upcoming move (typically around earnings).

**Metric**: `implied_move = ATM_straddle_price / spot`. Compare to historical realised moves of similar lengths.

```python
@scanner("implied-vs-realised")
class ImpliedVsRealisedScanner:
    def __init__(self, lookback_days: int = 252, gap_threshold: float = 0.5):
        self.lookback_days = lookback_days
        self.gap_threshold = gap_threshold

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            spot = fetch_spot(sym, asof)
            expiry = fetch_next_expiry(sym, asof, days_min=5, days_max=15)
            if not expiry:
                continue
            T = (expiry - asof).days / 365
            atm_call, atm_put = fetch_atm_straddle(sym, expiry)
            implied_move = (atm_call + atm_put) / spot
            # Historical realised moves of similar duration
            returns = fetch_close(sym, asof - pd.Timedelta(days=self.lookback_days), asof).pct_change()
            window = int((expiry - asof).days)
            realised_moves = returns.rolling(window).std() * np.sqrt(window)
            avg_realised = realised_moves.mean()
            gap = implied_move - avg_realised
            if abs(gap) / avg_realised > self.gap_threshold:
                rows.append({"symbol": sym, "expiry": expiry, "implied_move": implied_move,
                              "avg_realised_move": avg_realised, "gap_pct": gap / avg_realised * 100})
        return pd.DataFrame(rows).sort_values("gap_pct", key=abs, ascending=False)
```

Positive gap → implied > realised → vol-selling candidate. Negative gap → implied < realised → vol-buying candidate.

## 4. Post-earnings drift candidates

**The signal**: a stock that just announced earnings with a large SUE, likely to drift in the announcement's direction (Module 16 chapter 5).

```python
@scanner("pead-candidates")
class PEADCandidatesScanner:
    def __init__(self, sue_threshold: float = 1.5, days_since_earnings_max: int = 5):
        self.sue_threshold = sue_threshold
        self.max_days = days_since_earnings_max

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            earnings = fetch_last_earnings(sym, asof)
            if not earnings:
                continue
            days_since = (asof.date() - earnings.date.date()).days
            if days_since > self.max_days or days_since < 0:
                continue
            sue = (earnings.actual - earnings.consensus) / earnings.consensus_std
            if abs(sue) > self.sue_threshold:
                rows.append({"symbol": sym, "earnings_date": earnings.date, "sue": sue,
                              "days_since": days_since, "direction": int(np.sign(sue))})
        return pd.DataFrame(rows).sort_values("sue", key=abs, ascending=False)
```

Filter to a small list of candidates per day, then verify each against the PEAD strategy entry rules from Module 16.

## 5. Sector relative strength

**The signal**: a sector ETF is outperforming or underperforming its peers, suggesting persistent flow.

```python
@scanner("sector-rs")
class SectorRSScanner:
    def __init__(self, lookback: int = 60, threshold_z: float = 2.0):
        self.lookback = lookback
        self.threshold_z = threshold_z

    def scan(self, universe, asof):
        # universe should be sector ETFs (XLF, XLK, XLY, ...)
        prices = {sym: fetch_close(sym, asof - pd.Timedelta(days=self.lookback*2), asof)
                   for sym in universe}
        df = pd.concat(prices, axis=1).dropna()
        returns = df.pct_change(self.lookback).iloc[-1]
        mean_ret = returns.mean()
        std_ret = returns.std()
        z = (returns - mean_ret) / std_ret
        rows = [{"symbol": sym, "return_60d": ret, "z": z[sym]}
                 for sym, ret in returns.items() if abs(z[sym]) > self.threshold_z]
        return pd.DataFrame(rows).sort_values("z", key=abs, ascending=False)
```

Output: sectors with abnormally high or low 60-day return. Use as a "where's the money flowing?" signal.

## 6. ETF NAV-arbitrage

**The signal**: an ETF's premium/discount vs its NAV exceeds historical norms — arbitrage candidate (or, more often, a hedge candidate).

```python
@scanner("etf-nav-arb")
class ETFNavArbScanner:
    def __init__(self, threshold_bps: float = 30.0):
        self.threshold_bps = threshold_bps

    def scan(self, universe, asof):
        rows = []
        for sym in universe:
            market_price = fetch_last_price(sym, asof)
            nav = fetch_etf_nav(sym, asof)
            if not nav or market_price is None:
                continue
            premium_bps = (market_price - nav) / nav * 10000
            if abs(premium_bps) > self.threshold_bps:
                rows.append({"symbol": sym, "market": market_price, "nav": nav,
                              "premium_bps": premium_bps})
        return pd.DataFrame(rows).sort_values("premium_bps", key=abs, ascending=False)
```

For pure NAV-arb, you need creation-redemption access (institutional). For retail, large premiums can warn of avoiding the ETF or shorting it.

## Composing into a watchlist

Each morning, run all scanners against the universe; aggregate the resulting symbols (with sources):

```python
import pandas as pd
from engine.scanners import registry


def daily_watchlist(universe: list[str], asof: pd.Timestamp,
                     scanner_names: list[str] | None = None) -> pd.DataFrame:
    if scanner_names is None:
        scanner_names = registry.names()
    results = []
    for name in scanner_names:
        scn = registry.get(name)()
        df = scn.scan(universe, asof)
        df["source"] = name
        results.append(df[["symbol", "source"] +
                          [c for c in df.columns if c not in ("symbol", "source")]])
    combined = pd.concat(results, ignore_index=True)
    # Symbols appearing in multiple scanners → "convergence" signals
    counts = combined.groupby("symbol").size().sort_values(ascending=False)
    combined["scanner_count"] = combined["symbol"].map(counts)
    return combined.sort_values(["scanner_count", "symbol"], ascending=[False, True])
```

Symbols flagged by 3+ scanners on the same day are the highest-conviction setups.

## Pitfalls

!!! warning "Stale data"
    Scanner output is only as good as the data it ran on. Stale option chains, EOD prices presented as live — these all silently corrupt scanner output.

!!! warning "Universe selection"
    A scanner finding "10 great candidates" out of the S&P 500 is meaningful. The same scanner finding 10 out of all listed equities (~3000 names) may just be picking noise. Always know your universe.

!!! warning "Threshold-tuning bias"
    If you tune the scanner threshold to maximise historical backtest hits, you've selection-biased into a fragile setup. Use a single threshold across all symbols and validate walk-forward.

!!! warning "Backtest as decision-tool, not as predictor"
    Scanners don't predict the future; they prioritise where to spend attention. The strategy's edge comes from what you do *with* the flagged symbols, not from the flag itself.

## Bottom line

Production scanners are:

- Boring, robust, slowly-decaying.
- All built on the same Protocol-based framework.
- Most valuable when combined (symbols flagged by multiple scanners are highest conviction).
- Daily-frequency for liquid universes; intraday for higher-frequency strategies.

Continue to **[Frontier scanners](03-frontier.md)**.
