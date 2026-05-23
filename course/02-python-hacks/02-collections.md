# Collections you forgot you had

The `collections` module is one of the highest-value parts of the standard library. It exists precisely so you don't write the wrong data structure by hand.

## `Counter` — when you need histograms

`Counter` is a `dict` subclass for counting hashable things.

```python
from collections import Counter

ticks_per_symbol = Counter(t.symbol for t in stream)
print(ticks_per_symbol.most_common(5))
# [('SPY', 12345), ('QQQ', 9876), ...]

# Arithmetic on counters!
yesterday = Counter({"SPY": 100, "QQQ": 80})
today     = Counter({"SPY": 120, "TLT": 30})
print(today - yesterday)   # Counter({'SPY': 20, 'TLT': 30})
print(today + yesterday)   # combined counts
print(today & yesterday)   # min per key
```

A real use: bucketing fills by side and symbol with a `(symbol, side)` key.

```python
fills_by = Counter((f.symbol, f.side) for f in fills)
```

## `deque` — O(1) append and pop from both ends

A `deque` is a double-ended queue with O(1) `append`, `appendleft`, `pop`, `popleft`. Lists are O(n) for left-side operations.

```python
from collections import deque

# Bounded ring buffer for the last N values — perfect for rolling features
recent = deque(maxlen=20)
for px in prices:
    recent.append(px)
    rolling_mean = sum(recent) / len(recent) if recent else 0.0
```

A `deque` with `maxlen` is the canonical Python ring buffer: appending past capacity quietly drops from the other side. Use it for rolling stats in *streaming* code (in batch, use `pandas.Series.rolling`).

`deque` also has `rotate(n)`, which is the cleanest way to implement a cyclic shift:

```python
d = deque([1, 2, 3, 4, 5])
d.rotate(2)        # deque([4, 5, 1, 2, 3])
d.rotate(-1)       # deque([5, 1, 2, 3, 4])
```

## `defaultdict` — no more `setdefault` ceremony

```python
from collections import defaultdict

# Group ticks by symbol — one line.
groups = defaultdict(list)
for t in stream:
    groups[t.symbol].append(t)

# Nested counters
heatmap = defaultdict(lambda: defaultdict(int))
for t in stream:
    heatmap[t.symbol][t.minute] += 1
```

For most "group items by key" patterns, `defaultdict(list)` beats both `dict.setdefault` and a hand-rolled loop.

!!! tip "When `Counter` vs `defaultdict(int)`"
    Both count occurrences. `Counter` has the extra methods (`most_common`, arithmetic). `defaultdict(int)` is slightly faster on bare incrementing because no method-lookup overhead. Default to `Counter` for clarity.

## `ChainMap` — layered configuration

`ChainMap` glues multiple dicts together. Lookups walk the chain; the first dict with the key wins.

```python
from collections import ChainMap

defaults  = {"timeout": 30, "retries": 3, "host": "api.example.com"}
env_over  = {"host": "internal.example.com"}
flags     = {"retries": 5}

config = ChainMap(flags, env_over, defaults)
config["host"]     # "internal.example.com"   (env_over wins over defaults)
config["retries"]  # 5                         (flags wins)
config["timeout"]  # 30                        (only in defaults)

# A new map for a scoped override — leaves the chain unchanged
local = config.new_child({"retries": 1})
local["retries"]   # 1
config["retries"]  # 5
```

For configurations layered as CLI > env > file > defaults, this is the cleanest mental model in Python. Most projects roll their own (and get it slightly wrong). `ChainMap` is right out of the box.

## `OrderedDict` — still useful (sometimes)

Python 3.7+ guarantees regular `dict`s preserve insertion order. So `OrderedDict` is mostly redundant — *except* for `move_to_end()`, which is the cleanest way to implement an LRU:

```python
from collections import OrderedDict

class LRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: OrderedDict[str, object] = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)   # mark recently used
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)   # evict oldest
```

For production caches, use `functools.lru_cache`. But the OrderedDict + `move_to_end` is a great interview answer and a perfectly fine implementation when you need more control.

## `namedtuple` and `NamedTuple` — typed tuples without ceremony

For lightweight return values, `typing.NamedTuple` is great:

```python
from typing import NamedTuple

class BollingerBand(NamedTuple):
    mid: float
    upper: float
    lower: float

def bbands(prices, window=20, k=2.0) -> BollingerBand:
    mid = prices.rolling(window).mean().iloc[-1]
    std = prices.rolling(window).std().iloc[-1]
    return BollingerBand(mid, mid + k * std, mid - k * std)

b = bbands(close)
b.upper      # named access
mid, up, lo = b   # also tuple-unpacks
```

If you need methods or mutability, prefer a `dataclass`. NamedTuples are immutable, light, and unpack-friendly.

## `SimpleNamespace` — quick attribute bag

When you want `obj.x` instead of `obj["x"]` and you don't want to define a class:

```python
from types import SimpleNamespace

ctx = SimpleNamespace(symbol="SPY", price=475.30, qty=100)
ctx.symbol         # 'SPY'
ctx.price = 475.40 # mutable
```

Good for ad-hoc data carriers in tests and scripts. For production, use a dataclass.

## `frozenset` — hashable sets

You can put a `frozenset` in a dict key or another set, which makes it the right way to memoise on "the set of features I used":

```python
universe_key = frozenset(("SPY", "QQQ", "IWM"))
cache[universe_key] = compute(universe_key)
```

`frozenset({1, 2, 3}) == frozenset({3, 2, 1})` is True — order doesn't matter for set equality.

## `dict.fromkeys` — empty-dict initialisation idiom

```python
weights = dict.fromkeys(["SPY", "QQQ", "IWM"], 0.0)
# {'SPY': 0.0, 'QQQ': 0.0, 'IWM': 0.0}
```

Cleaner than a dict comprehension when the value is constant.

## A trading-specific worked example: order book level summary

A toy aggregator that ingests level-2 updates and maintains the top-of-book summary using exactly the tools above:

```python
from collections import defaultdict
from dataclasses import dataclass
from heapq import nlargest, nsmallest

@dataclass(slots=True)
class Book:
    bids: dict[float, int]   # price -> total size at that level
    asks: dict[float, int]

    def __init__(self):
        self.bids = defaultdict(int)
        self.asks = defaultdict(int)

    def update(self, side: str, price: float, delta: int) -> None:
        book = self.bids if side == "BID" else self.asks
        book[price] += delta
        if book[price] <= 0:
            del book[price]

    def top(self, n: int = 5):
        best_bids = nlargest(n, self.bids.items())     # by price descending
        best_asks = nsmallest(n, self.asks.items())    # by price ascending
        return best_bids, best_asks

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks: return None
        return min(self.asks) - max(self.bids)
```

`defaultdict(int)`, `nlargest`, `nsmallest` — three tiny pieces of the standard library, and we have a working order book in 30 lines.

Continue to **[Performance and profiling](03-performance.md)**.
