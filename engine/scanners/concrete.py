"""Two concrete scanners on real, swappable data.

Both implement the ``Scanner`` protocol, both auto-register, both run
end-to-end against any ``Feed`` (so the default ``YFinanceFeed`` works
out of the box). These are deliberately the simplest production-shape
scanners that aren't toys:

- :class:`ZScoreMomentumScanner` — short-horizon mean-reversion via the
  z-score of N-day returns.
- :class:`RealisedVolRankScanner` — annualised realised vol percentile
  rank across the universe (a stock-side stand-in for the "IV-rank"
  scanner described in Module 18 chapter 1).

Both take a ``Feed`` at construction so the scanning logic stays unit-
testable: pass a feed backed by an in-memory dict in tests, pass a
``YFinanceFeed`` in production.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import scanner


@scanner("zscore-momentum")
@dataclass
class ZScoreMomentumScanner:
    """Rank symbols by the standardised N-day return.

    ``feed`` must implement ``bars(symbol, start, end, interval) -> DataFrame``
    with at least a 'close' column. Symbols are ranked by ``|z|`` (largest
    dislocations first). ``direction`` is ``"long"``, ``"short"``, or
    ``"both"``.
    """

    feed: object  # duck-typed Feed; declared loose to avoid a circular import
    lookback_days: int = 20
    window_days: int = 252
    direction: str = "both"
    name: str = field(default="zscore-momentum", init=False)

    def scan(self, universe: Sequence[str], asof: pd.Timestamp) -> pd.DataFrame:
        start = asof - pd.Timedelta(days=self.window_days * 2)
        rows: list[dict[str, float | str]] = []
        for sym in universe:
            try:
                df = self.feed.bars(  # type: ignore[attr-defined]
                    sym, start=start, end=asof, interval="1d"
                )
            except Exception:
                continue
            if df.empty or len(df) < self.lookback_days + 2:
                continue
            ret = df["close"].pct_change(self.lookback_days).dropna()
            if len(ret) < 30:
                continue
            mu, sigma = float(ret.mean()), float(ret.std(ddof=1))
            if sigma <= 0:
                continue
            last = float(ret.iloc[-1])
            z = (last - mu) / sigma
            if self.direction == "long" and z >= 0:
                continue
            if self.direction == "short" and z <= 0:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "last_return": last,
                    "mean": mu,
                    "std": sigma,
                    "zscore": z,
                    "abs_zscore": abs(z),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("abs_zscore", ascending=False).reset_index(drop=True)


@scanner("rvol-rank")
@dataclass
class RealisedVolRankScanner:
    """Rank symbols by the percentile of *recent* annualised vol vs its
    own trailing history.

    Conceptually the stock-side analogue of an "IV rank" scanner: where
    in its own range is each name's current vol sitting? Use the top of
    the ranking for vol-selling candidates and the bottom for vol-buying.
    """

    feed: object
    short_window_days: int = 20
    history_days: int = 252
    name: str = field(default="rvol-rank", init=False)

    def scan(self, universe: Sequence[str], asof: pd.Timestamp) -> pd.DataFrame:
        start = asof - pd.Timedelta(days=self.history_days * 2 + 30)
        rows: list[dict[str, float | str]] = []
        for sym in universe:
            try:
                df = self.feed.bars(  # type: ignore[attr-defined]
                    sym, start=start, end=asof, interval="1d"
                )
            except Exception:
                continue
            if df.empty or len(df) < self.history_days + self.short_window_days + 2:
                continue
            ret = df["close"].pct_change().dropna()
            rvol = ret.rolling(self.short_window_days).std(ddof=1) * np.sqrt(252)
            rvol = rvol.dropna()
            if len(rvol) < self.history_days:
                continue
            history = rvol.iloc[-self.history_days :]
            current = float(rvol.iloc[-1])
            rank = float((history < current).mean())  # percentile in [0, 1]
            rows.append(
                {
                    "symbol": sym,
                    "rvol_current": current,
                    "rvol_min": float(history.min()),
                    "rvol_max": float(history.max()),
                    "rvol_rank": rank,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("rvol_rank", ascending=False).reset_index(drop=True)
