# Iteration and lookup hacks

A small collection of patterns that, once you've seen them, you'll reach for every week.

## `bisect` — binary search you should be using

If you have a sorted list and you find yourself doing `for i, x in enumerate(xs): if x > threshold: ...`, stop. `bisect` does it in O(log n):

```python
from bisect import bisect_left, bisect_right, insort

xs = [10, 20, 20, 30, 40]
bisect_left(xs, 20)    # 1  — first position where 20 would go
bisect_right(xs, 20)   # 3  — first position past existing 20s
insort(xs, 25)         # xs == [10, 20, 20, 25, 30, 40]
```

A real use: snapping a timestamp to the most-recent bar in a sorted index.

```python
from bisect import bisect_right
def latest_bar(timestamps_ns: list[int], when: int) -> int | None:
    """Return the index of the latest bar at or before `when`."""
    i = bisect_right(timestamps_ns, when) - 1
    return i if i >= 0 else None
```

Pandas's `searchsorted` is the vectorised equivalent for big arrays; `bisect` wins when you have a scalar query against a list.

## `heapq` — top-N without sorting everything

The right tool when you need the top (or bottom) N items from a stream:

```python
import heapq

# Top 5 by volume from a long iterator
top5 = heapq.nlargest(5, ticks, key=lambda t: t.size)

# Streaming median via two heaps (lo: max-heap as negatives, hi: min-heap)
class StreamingMedian:
    def __init__(self):
        self.lo, self.hi = [], []   # max-heap (negate), min-heap
    def push(self, x: float) -> None:
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)
        # rebalance
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    @property
    def median(self) -> float:
        if len(self.lo) > len(self.hi): return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

That's a robust running-median over an unbounded stream in ~20 lines. The naive "store everything and sort" is O(n log n) per update; this is O(log n).

## The walrus, used well

The walrus operator `:=` assigns inside an expression. Used badly it produces clever one-liners that nobody can read. Used well, it removes redundant lookups:

```python
# bad — double lookup
while chunk := f.read(8192):
    if len(f.read(8192)) > 0: process(chunk)

# good — assign once, use it
while chunk := f.read(8192):
    process(chunk)
```

Another good use is in comprehensions where the same expensive computation appears twice:

```python
positions = [pos for sym in symbols if (pos := lookup(sym)) is not None]
```

Without walrus you'd call `lookup(sym)` twice. With it, once.

## Dispatch tables, not chains of `if`

When you have a long `if/elif` over a value, lift it into a dict:

```python
# before
def handle(event):
    if event.kind == "ACK":     return on_ack(event)
    elif event.kind == "FILL":  return on_fill(event)
    elif event.kind == "REJECT": return on_reject(event)
    ...

# after
HANDLERS = {
    "ACK": on_ack,
    "FILL": on_fill,
    "REJECT": on_reject,
}

def handle(event):
    handler = HANDLERS.get(event.kind, on_unknown)
    return handler(event)
```

Three wins: it's O(1) instead of O(n), it can be extended at runtime, and the table itself is documentation.

## `match` for structural cases

Python 3.10's structural pattern matching is more powerful than most people give it credit for. It's not just a switch:

```python
def handle(event):
    match event:
        case {"kind": "FILL", "side": "BUY",  "qty": int(qty)} if qty > 0:
            book.long(qty, event["price"])
        case {"kind": "FILL", "side": "SELL", "qty": int(qty)} if qty > 0:
            book.short(qty, event["price"])
        case {"kind": "CANCEL", "order_id": str(oid)}:
            book.cancel(oid)
        case _:
            log.warning("unknown event: %r", event)
```

Patterns can match shapes, types, sub-structures, and add guards. Used judiciously, this is the cleanest way to handle heterogeneous event streams.

## `else` on loops

If a `for` or `while` completes without `break`, its `else` clause runs. Useful for "find this thing, else complain":

```python
for order in book:
    if order.id == target_id:
        order.cancel(); break
else:
    log.warning("no order %s in book", target_id)
```

You meet it once, look up what it does (because it's surprising), then use it for the rest of your career.

## Unpacking that goes further than you think

```python
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

# In function calls
def fn(a, b, c, d, e): ...
args = [1, 2]; kwargs = {"c": 3, "d": 4, "e": 5}
fn(*args, **kwargs)

# In dict literals
defaults = {"timeout": 30, "retries": 3}
config = {**defaults, "retries": 5}   # override one key, keep others
```

## `dict.setdefault` and `dict.get` with sentinel

Two underused dict idioms:

```python
# Group items by symbol — no defaultdict required
buckets = {}
for tick in stream:
    buckets.setdefault(tick.symbol, []).append(tick)

# Distinguish "missing" from "None present"
MISSING = object()
val = config.get("optional_key", MISSING)
if val is not MISSING:
    apply(val)
```

The sentinel pattern is correct when `None` is a legitimate value.

## Ternaries, conditional expressions, and `or`

Three ways to express "default if falsy":

```python
x = a if a else b           # explicit
x = a or b                  # short-circuit; bug if a is 0 or "" and you wanted those
x = a if a is not None else b   # the "I really meant None" version
```

Use the third form when 0/empty are legitimate values. The second is fine for "this string or a default".

## Generator expression in `any` / `all`

Short-circuiting all() and any() over a generator is a one-liner test:

```python
if any(o.symbol == "SPY" and o.status == "OPEN" for o in book):
    ...
```

Builds no list, stops at the first match. Same for `all` for "every order is filled":

```python
if all(o.status == "FILLED" for o in order_group):
    finalize(order_group)
```

## `functools.lru_cache` and `functools.cache`

Memoise pure functions trivially:

```python
from functools import lru_cache, cache

@cache                           # unbounded, since 3.9; alias for lru_cache(maxsize=None)
def implied_volatility(price, strike, t, r, kind):
    ...
```

Two caveats: arguments must be hashable, and the cache lives in the function — clear it with `fn.cache_clear()` between tests. For methods, prefer `@cached_property` (Module 1, descriptors) — that one caches per instance.

## `operator` — first-class functions for arithmetic

The `operator` module gives you function forms of every operator. Combined with `map`, `functools.reduce`, or pandas-style `apply`, they're tidy:

```python
from operator import add, mul, attrgetter, itemgetter
from functools import reduce

total = reduce(add, [t.notional for t in trades], 0.0)

# Sort by attribute / dict-key
trades.sort(key=attrgetter("notional"))
records.sort(key=itemgetter("ts"))

# Compose with map
weights = list(map(mul, sizes, prices))
```

`attrgetter` and `itemgetter` are dramatically faster than `lambda x: x.foo` because they're written in C.

## `enumerate` from a custom start

Tiny but recurring: `enumerate(xs, start=1)` for one-based numbering in user-facing output.

## A worked one-liner: bucketing latencies

You have a list of round-trip times in microseconds. You want a quick histogram of P50/P90/P99 plus a count per latency bucket:

```python
import bisect, statistics

buckets = [0, 100, 250, 500, 1000, 2500, 5000, 10000]
counts = [0] * (len(buckets) + 1)
for rtt in latencies:
    counts[bisect.bisect_right(buckets, rtt)] += 1

p50 = statistics.median(latencies)
p90 = statistics.quantiles(latencies, n=10)[8]
p99 = statistics.quantiles(latencies, n=100)[98]
```

`statistics.quantiles` is in stdlib (3.8+) and uses inclusive/exclusive interpolation correctly. For really big arrays use NumPy; for streams up to ~1M items, stdlib is fine.

## Anti-patterns to retire

!!! danger "`x = list(map(lambda v: v*2, xs))`"
    `[v*2 for v in xs]` is shorter, faster, and idiomatic. Use `map` only when the function is already named.

!!! danger "Mutable default argument"
    ```python
    def f(items=[]):   # NO — the list is shared across calls
        items.append(1)
        return items
    f(); f()           # [1, 1]
    ```
    Use `items=None` and `items = [] if items is None else items` inside.

!!! danger "`==` against `None`"
    Use `is None` / `is not None`. The `==` comparison may call `__eq__` (pandas's `Series.__eq__` returns a Series, not a bool — surprise traceback).

!!! danger "`list += iter`"
    `lst += gen()` exhausts the generator into the list, which is usually fine but never makes the operation lazy. If you want lazy, `itertools.chain`.

Continue to **[Collections you forgot you had](02-collections.md)**.
