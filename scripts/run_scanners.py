"""Daily scanner report runner.

Iterates the scanner registry, runs each scanner against a configurable
universe, aggregates per-symbol scores, and writes a CSV + Markdown digest
to ``daily_reports/YYYY-MM-DD/``. Module 18 chapter 5 walks through the
pattern conceptually; this is the deployable version.

Usage:
    python scripts/run_scanners.py --universe SPY QQQ IWM
    python scripts/run_scanners.py --scanner-names test-zscore
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data import ParquetCache, YFinanceFeed
from engine.scanners import registry, scanner

DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "EFA",
    "EEM",
    "VNQ",
    "UUP",
]


# ----------------------------------------------------------------------------
# A minimal set of demo scanners so `run_scanners.py` works out of the box.
# Real deployments would import scanners from a `strategies/` package; the
# decorator pattern auto-registers them at import time.
# ----------------------------------------------------------------------------


@scanner("zscore-momentum-demo")
class ZScoreMomentumDemo:
    """Flags symbols whose 60-day return z-score exceeds a threshold.
    Demo scanner so this runner does something useful out of the box."""

    def __init__(self, lookback: int = 60, threshold: float = 1.5):
        self.lookback = lookback
        self.threshold = threshold

    def scan(self, universe, asof: pd.Timestamp) -> pd.DataFrame:
        cache = ParquetCache(YFinanceFeed(), root="data/bars")
        rows = []
        for sym in universe:
            try:
                window_start = asof - pd.Timedelta(days=self.lookback * 3)
                bars = cache.bars(sym, window_start, asof)
                if bars.empty or len(bars) < self.lookback:
                    continue
                returns = bars["close"].pct_change().dropna().tail(self.lookback)
                z = float((returns.iloc[-1] - returns.mean()) / returns.std())
                if abs(z) > self.threshold:
                    rows.append(
                        {
                            "symbol": sym,
                            "z": z,
                            "direction": int(np.sign(z)),
                            "lookback_days": self.lookback,
                        }
                    )
            except Exception as exc:  # pragma: no cover
                print(f"  [zscore] {sym} failed: {exc!r}")
        if not rows:
            return pd.DataFrame(columns=["symbol", "z", "direction", "lookback_days"])
        return pd.DataFrame(rows).sort_values("z", key=abs, ascending=False)


@scanner("vol-regime-demo")
class VolRegimeDemo:
    """Flags symbols where realised vol differs sharply from its 1-year norm."""

    def __init__(self, short: int = 20, long: int = 252, ratio_threshold: float = 1.5):
        self.short = short
        self.long = long
        self.ratio_threshold = ratio_threshold

    def scan(self, universe, asof: pd.Timestamp) -> pd.DataFrame:
        cache = ParquetCache(YFinanceFeed(), root="data/bars")
        rows = []
        for sym in universe:
            try:
                window_start = asof - pd.Timedelta(days=self.long * 2)
                bars = cache.bars(sym, window_start, asof)
                if bars.empty or len(bars) < self.long:
                    continue
                rets = bars["close"].pct_change().dropna()
                short_vol = float(rets.tail(self.short).std() * np.sqrt(252))
                long_vol = float(rets.tail(self.long).std() * np.sqrt(252))
                ratio = short_vol / long_vol if long_vol > 0 else float("nan")
                if np.isfinite(ratio) and (
                    ratio > self.ratio_threshold or ratio < 1 / self.ratio_threshold
                ):
                    rows.append(
                        {
                            "symbol": sym,
                            "short_vol": short_vol,
                            "long_vol": long_vol,
                            "vol_ratio": ratio,
                        }
                    )
            except Exception as exc:  # pragma: no cover
                print(f"  [vol-regime] {sym} failed: {exc!r}")
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(
            "vol_ratio", key=lambda s: (s - 1).abs(), ascending=False
        )


# ----------------------------------------------------------------------------
# Aggregation + reporting
# ----------------------------------------------------------------------------


def aggregate(
    universe: list[str],
    asof: pd.Timestamp,
    scanner_names: list[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or {n: 1.0 for n in scanner_names}
    per_symbol: dict[str, dict] = defaultdict(lambda: {"score": 0.0, "sources": [], "metadata": []})
    for name in scanner_names:
        try:
            scn = registry.get(name)()
        except KeyError:
            print(f"  [aggregate] unknown scanner {name!r}; skipping")
            continue
        try:
            df = scn.scan(universe, asof)
        except Exception as exc:  # pragma: no cover
            print(f"  [aggregate] {name} failed: {exc!r}")
            continue
        if df.empty or "symbol" not in df.columns:
            continue
        for _, row in df.iterrows():
            sym = row["symbol"]
            per_symbol[sym]["score"] += weights.get(name, 1.0)
            per_symbol[sym]["sources"].append(name)
            per_symbol[sym]["metadata"].append(row.drop(labels="symbol").to_dict())
    if not per_symbol:
        return pd.DataFrame(columns=["symbol", "score", "n_sources", "sources"])
    out = pd.DataFrame(
        [
            {
                "symbol": sym,
                "score": data["score"],
                "n_sources": len(data["sources"]),
                "sources": ", ".join(data["sources"]),
                "metadata": json.dumps(data["metadata"], default=str),
            }
            for sym, data in per_symbol.items()
        ]
    )
    return out.sort_values(["score", "n_sources"], ascending=False)


def write_report(aggregated: pd.DataFrame, asof: pd.Timestamp, output_dir: Path) -> None:
    day_dir = output_dir / asof.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(day_dir / "report.csv", index=False)
    lines = [
        f"# Scanner report — {asof.strftime('%Y-%m-%d')}",
        "",
        f"Symbols flagged: **{len(aggregated)}**",
        "",
        "| Symbol | Score | Sources |",
        "|--------|------:|---------|",
    ]
    for _, row in aggregated.head(50).iterrows():
        lines.append(f"| {row['symbol']} | {row['score']:.1f} | {row['sources']} |")
    (day_dir / "report.md").write_text("\n".join(lines))
    print(f"  [report] wrote {day_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", nargs="+", default=DEFAULT_UNIVERSE, help="symbols to scan")
    parser.add_argument(
        "--scanner-names",
        nargs="+",
        default=None,
        help="subset of registered scanners to run (default: all)",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="run for this date (default: today UTC)",
    )
    parser.add_argument(
        "--output-dir",
        default="daily_reports",
        help="root directory for per-day report subdirs",
    )
    args = parser.parse_args()

    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.utcnow().normalize()
    scanner_names = args.scanner_names or registry.names()

    print(f"[scanners] asof={asof.date()} universe={args.universe} scanners={scanner_names}")
    aggregated = aggregate(args.universe, asof, scanner_names)
    print(f"[scanners] {len(aggregated)} flagged symbols across {len(scanner_names)} scanners")
    if not aggregated.empty:
        print(aggregated.head(20).to_string(index=False))
    write_report(aggregated, asof, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
