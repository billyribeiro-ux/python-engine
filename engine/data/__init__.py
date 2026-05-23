"""Universal market-data adapter.

The ``Feed`` protocol is the single seam through which every example in the
course gets its data. Swap the implementation and the rest of the codebase
keeps working unchanged.

Examples
--------
>>> from engine.data import YFinanceFeed
>>> feed = YFinanceFeed()
>>> bars = feed.bars("SPY", "2024-01-01", "2024-02-01")
>>> bars.columns.tolist()
['open', 'high', 'low', 'close', 'volume']
"""

from __future__ import annotations

from .cache import ParquetCache
from .feed import (
    AlpacaFeed,
    Bar,
    Feed,
    IBKRFeed,
    OptionsChain,
    PolygonFeed,
    TradierFeed,
    YFinanceFeed,
    default_feed,
)

__all__ = [
    "AlpacaFeed",
    "Bar",
    "Feed",
    "IBKRFeed",
    "OptionsChain",
    "ParquetCache",
    "PolygonFeed",
    "TradierFeed",
    "YFinanceFeed",
    "default_feed",
]
