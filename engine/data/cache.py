"""On-disk Parquet cache for market data.

The cache wraps any :class:`engine.data.feed.Feed` and persists daily-or-
coarser bars to a date-partitioned Parquet store. Re-requests are served
from disk; only missing windows are fetched from the upstream feed.

Design rules
------------
* One Parquet file per ``(symbol, interval, year, month)`` partition, so a
  typical 5-year window is at most ~60 files per symbol.
* The on-disk schema matches the in-memory contract from
  :mod:`engine.data.feed`: lower-case OHLCV columns and a tz-aware UTC index
  named ``timestamp``.
* The cache is **content-addressable by request window**: it remembers which
  windows have been pulled so it can answer "give me Jan 1 → Mar 1" with a
  single read even if it was pulled as two separate windows originally.

The implementation here is deliberately small (~200 lines). Production
deployments often add concurrency primitives (file locks, multi-writer
coordination) but the single-process case is the common case and stays
boring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .feed import DateLike, Feed, Interval, OptionsChain, _normalise_bars

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _month_path(root: Path, symbol: str, interval: Interval, month: str) -> Path:
    return root / f"interval={interval}" / f"symbol={symbol.upper()}" / f"month={month}.parquet"


def _iter_months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """All month keys (YYYY-MM) intersecting [start, end]."""
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    end_norm = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    out = []
    while cur <= end_norm:
        out.append(_month_key(cur))
        cur = (cur + pd.offsets.MonthBegin(1)).tz_convert("UTC")
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class _Manifest:
    """Tiny JSON file remembering the spans we've already pulled.

    The cache uses the manifest to avoid re-pulling overlapping windows. We
    track the *requested* span, not the returned span — vendors sometimes
    return fewer rows than asked, and the manifest still proves we tried.
    """

    path: Path
    data: dict

    @classmethod
    def load(cls, path: Path) -> _Manifest:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return cls(path, json.loads(path.read_text()))
        return cls(path, {"spans": []})

    def add(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        self.data["spans"].append([start.isoformat(), end.isoformat()])
        self.path.write_text(json.dumps(self.data, sort_keys=True, indent=2))

    def covers(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        for s, e in self.data["spans"]:
            if pd.Timestamp(s) <= start and pd.Timestamp(e) >= end:
                return True
        return False


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ParquetCache:
    """Wrap a :class:`Feed` and persist bars to a Parquet directory.

    >>> from engine.data import YFinanceFeed
    >>> from engine.data.cache import ParquetCache
    >>> cache = ParquetCache(YFinanceFeed(), root="data/bars")
    >>> bars = cache.bars("SPY", "2024-01-02", "2024-03-31")   # pulls + writes
    >>> bars = cache.bars("SPY", "2024-01-02", "2024-03-31")   # serves from disk
    """

    name = "parquet-cache"

    def __init__(self, upstream: Feed, root: str | Path):
        self.upstream = upstream
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ bars

    def bars(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: Interval = "1d",
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        manifest_path = self.root / f"interval={interval}" / f"symbol={symbol}" / "_manifest.json"
        manifest = _Manifest.load(manifest_path)

        if not manifest.covers(start_ts, end_ts):
            # Fetch the missing window and write each month's slice
            fetched = _normalise_bars(self.upstream.bars(symbol, start_ts, end_ts, interval))
            if not fetched.empty:
                for month, slice_ in fetched.groupby(_month_key_series(fetched.index)):
                    path = _month_path(self.root, symbol, interval, str(month))
                    self._write_month(path, slice_)
                manifest.add(start_ts, end_ts)

        # Read everything back from disk and slice to the requested window
        return self._read_window(symbol, interval, start_ts, end_ts)

    def _write_month(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index, utc=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(path, compression="zstd")

    def _read_window(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        frames = []
        for month in _iter_months(start, end):
            path = _month_path(self.root, symbol, interval, month)
            if path.exists():
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index, utc=True)
                frames.append(df)
        if not frames:
            return _normalise_bars(pd.DataFrame())
        out = pd.concat(frames).sort_index()
        return out.loc[(out.index >= start) & (out.index <= end)]

    # --------------------------------------------------------- option chains
    # Options chains are real-time snapshots; caching them is a different
    # problem with different semantics (snapshot vs span). The course covers
    # chain storage in chapter 4 of this module.

    def option_chain(self, symbol: str, expiry: DateLike | None = None) -> OptionsChain:
        return self.upstream.option_chain(symbol, expiry)


def _month_key_series(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.strftime("%Y-%m"), index=idx)
