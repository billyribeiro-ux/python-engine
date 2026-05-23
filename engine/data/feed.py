"""Universal data feed abstraction.

The whole course runs against one tiny interface: :class:`Feed`. Every
concrete vendor adapter implements it. Examples ship with :class:`YFinanceFeed`
so anyone can run the course with zero credentials, but you can swap in
Polygon / Alpaca / Tradier / IBKR by changing one import line.

Design rules
------------
* Frames returned by ``bars`` always have columns ``open, high, low, close,
  volume`` (lower-case) and a tz-aware ``DatetimeIndex`` named ``timestamp``.
* Times are UTC unless the vendor only gives local exchange time, in which
  case they're localised to ``America/New_York`` and documented per-adapter.
* Missing prints are dropped — never forward-filled silently. Filling is a
  modelling decision and lives in :mod:`engine.features`, not here.
* Adapters are stateless w.r.t. requests; auth keys live in env vars so the
  same object is safe to share across threads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

Interval = Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
DateLike = str | date | datetime | pd.Timestamp


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar. Mostly useful in event-driven contexts; for
    vectorised work we hand around DataFrames directly."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OptionsChain:
    """A snapshot of one options chain.

    The DataFrames have a stable schema across vendors:

    * ``calls`` / ``puts``: ``strike, last, bid, ask, volume, open_interest,
      implied_volatility, in_the_money, expiry``
    * ``spot``: the underlying mid at the time of the snapshot.
    """

    symbol: str
    asof: pd.Timestamp
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Feed(Protocol):
    """The single seam between your code and the market.

    A ``Feed`` is anything that can answer two questions:

    1. *"Give me historical bars for symbol X between dates A and B at this
       interval."*  → :meth:`bars`
    2. *"Give me the live options chain for symbol X at expiry E."*
       → :meth:`option_chain` (raises ``NotImplementedError`` if the vendor
       doesn't carry options data).
    """

    name: str

    def bars(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: Interval = "1d",
    ) -> pd.DataFrame: ...

    def option_chain(self, symbol: str, expiry: DateLike | None = None) -> OptionsChain: ...


# ---------------------------------------------------------------------------
# Helpers shared across adapters
# ---------------------------------------------------------------------------


_CANONICAL_COLS = ["open", "high", "low", "close", "volume"]


def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with canonical columns and a tz-aware UTC index.

    Vendors disagree on capitalisation, on whether the index is named, and on
    timezone handling. This function is the chokepoint that hides all of that.
    """
    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLS).rename_axis("timestamp")

    df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
    # yfinance returns 'adj close'; we drop it: total-return adjustments
    # are a modelling decision, not a data one.
    df = df[[c for c in _CANONICAL_COLS if c in df.columns]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df.sort_index()


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class YFinanceFeed:
    """Free, anonymous, frequently rate-limited — perfect for the course.

    Notes
    -----
    * ``yfinance`` is a scraper of public Yahoo endpoints; flaky behaviour is
      to be expected. Use it for learning, not for live trading.
    * Intraday intervals are restricted by Yahoo to short look-back windows.
      The adapter passes your request through verbatim and lets ``yfinance``
      raise if the window is too long.
    """

    name = "yfinance"

    def bars(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: Interval = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf  # local import: optional dep

        raw = yf.download(
            symbol,
            start=pd.Timestamp(start).date(),
            end=pd.Timestamp(end).date(),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        # yfinance returns a MultiIndex when you ask for >1 symbol; we ask
        # for one, but it sometimes wraps anyway depending on version.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return _normalise_bars(raw)

    def option_chain(self, symbol: str, expiry: DateLike | None = None) -> OptionsChain:
        import yfinance as yf

        t = yf.Ticker(symbol)
        expiries = t.options
        if not expiries:
            raise ValueError(f"No options listed for {symbol}")
        chosen = str(pd.Timestamp(expiry).date()) if expiry is not None else expiries[0]
        if chosen not in expiries:
            raise ValueError(
                f"Expiry {chosen} not available for {symbol}. "
                f"Available: {expiries[:8]}{'...' if len(expiries) > 8 else ''}"
            )

        oc = t.option_chain(chosen)
        spot = float(t.history(period="1d")["Close"].iloc[-1])
        rename = {
            "strike": "strike",
            "lastPrice": "last",
            "bid": "bid",
            "ask": "ask",
            "volume": "volume",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_volatility",
            "inTheMoney": "in_the_money",
        }

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            out = df.rename(columns=rename)[list(rename.values())].copy()
            out["expiry"] = pd.Timestamp(chosen)
            return out

        return OptionsChain(
            symbol=symbol,
            asof=pd.Timestamp.utcnow(),
            spot=spot,
            calls=_clean(oc.calls),
            puts=_clean(oc.puts),
        )


# ---------------------------------------------------------------------------
# Stubs for paid vendors — populated in later phases of the course
# ---------------------------------------------------------------------------


class _NotImplementedFeed:
    """Base class for the stubs. Each docstring tells the reader exactly what
    they need to do to wire up real credentials."""

    name = "stub"
    _vendor_doc: str = ""

    def __init__(self, **_: object) -> None:
        msg = (
            f"{type(self).__name__} is a placeholder. "
            f"See {self._vendor_doc} in the course for setup."
        )
        raise NotImplementedError(msg)

    def bars(self, *_: object, **__: object) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def option_chain(self, *_: object, **__: object) -> OptionsChain:  # pragma: no cover
        raise NotImplementedError


class PolygonFeed(_NotImplementedFeed):
    """Polygon.io adapter.

    Set ``POLYGON_API_KEY`` and the full implementation lands in Module 5
    (Data Engineering). Polygon gives you historical options chains with
    greeks and tick-level equities, which is what you want for serious work.
    """

    name = "polygon"
    _vendor_doc = "course/05-data-engineering/polygon.md"


class AlpacaFeed(_NotImplementedFeed):
    """Alpaca Markets adapter.

    Uses ``alpaca-py``. Equities are free at IEX-level. Set
    ``ALPACA_API_KEY_ID`` and ``ALPACA_API_SECRET_KEY``.
    """

    name = "alpaca"
    _vendor_doc = "course/05-data-engineering/alpaca.md"


class TradierFeed(_NotImplementedFeed):
    """Tradier adapter.

    Tradier is the go-to free options data feed for retail, and supports paper
    trading. Set ``TRADIER_TOKEN`` and ``TRADIER_ACCOUNT_ID``.
    """

    name = "tradier"
    _vendor_doc = "course/05-data-engineering/tradier.md"


class IBKRFeed(_NotImplementedFeed):
    """Interactive Brokers adapter via ``ib_insync``.

    Requires TWS or IB Gateway running locally; the adapter connects to
    127.0.0.1:7497 (paper) by default. Best free option for live trading.
    """

    name = "ibkr"
    _vendor_doc = "course/05-data-engineering/ibkr.md"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def default_feed() -> Feed:
    """Return whatever feed the env asks for, falling back to Yahoo.

    Set ``PYTHON_ENGINE_FEED=polygon`` (etc.) to switch. Examples in the
    course use this so a single env var rewires the whole project.
    """
    name = os.getenv("PYTHON_ENGINE_FEED", "yfinance").lower()
    table: dict[str, type[Feed]] = {
        "yfinance": YFinanceFeed,
        "polygon": PolygonFeed,
        "alpaca": AlpacaFeed,
        "tradier": TradierFeed,
        "ibkr": IBKRFeed,
    }
    if name not in table:
        raise ValueError(f"Unknown feed {name!r}. Known: {sorted(table)}")
    return table[name]()
