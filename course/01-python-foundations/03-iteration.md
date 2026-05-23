# Iteration, generators, coroutines

If the data model is the *structural* foundation of Python, iteration is the *temporal* foundation. Almost every interesting Python construct — `for`, comprehensions, `yield`, generators, async, even `with` — is iteration in disguise. Understanding it deeply pays for itself the day you process your first multi-billion-row dataset.

## The iteration protocol

A Python iterable is anything `iter()` can ask for an iterator from. An iterator is anything `next()` can call for the next element (or `StopIteration` to end). Two dunders, total:

```python
# An iterable
class Range:
    def __init__(self, n): self.n = n
    def __iter__(self): return RangeIterator(self.n)

# An iterator
class RangeIterator:
    def __init__(self, n): self.n = n; self.i = 0
    def __iter__(self): return self                # iterators are themselves iterable
    def __next__(self):
        if self.i >= self.n: raise StopIteration
        self.i += 1
        return self.i - 1

for x in Range(3): print(x)
# 0
# 1
# 2
```

That's a complete custom iterator, and it's also more code than any sensible person would write today. Generators replace this whole pattern with three keystrokes.

## Generators

A function that contains `yield` is a generator. Calling it does not run the body — it returns a *generator object*. Each call to `next()` on that object runs the body until the next `yield`, then pauses.

```python
def range_gen(n):
    i = 0
    while i < n:
        yield i
        i += 1

for x in range_gen(3): print(x)
# 0 1 2
```

Same behaviour, one quarter of the code, and dramatically faster (generators are implemented in C as a single frame object).

### `yield from`

`yield from` delegates to another iterable. It's both a convenience and the mechanism that lets generators compose:

```python
def flatten(seq):
    for x in seq:
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

list(flatten([1, [2, [3, [4, 5]], 6], 7]))
# [1, 2, 3, 4, 5, 6, 7]
```

For trading code, `yield from` is the natural fit for **streaming concatenation** — reading multiple Parquet partitions, multiple log files, multiple sessions, all as a single logical stream.

```python
from pathlib import Path
import pyarrow.parquet as pq

def stream_partitions(root: Path):
    """Stream rows from every .parquet file under `root`, in sorted order."""
    for path in sorted(root.glob("*.parquet")):
        table = pq.read_table(path)
        yield from table.to_pylist()
```

## Generator pipelines

Generators chain. The output of one is the input of the next. This is the Python equivalent of Unix pipes, and it's how you build memory-bounded pipelines over arbitrarily large data.

```python
def read_ticks(path):
    with open(path) as f:
        for line in f:
            ts, px, sz = line.strip().split(",")
            yield int(ts), float(px), int(sz)

def above_size(ticks, n):
    for ts, px, sz in ticks:
        if sz >= n: yield ts, px, sz

def into_bars(ticks, bucket_ns):
    bucket = None; o = h = l = c = v = 0
    for ts, px, sz in ticks:
        b = ts // bucket_ns
        if b != bucket:
            if bucket is not None:
                yield bucket * bucket_ns, o, h, l, c, v
            bucket, o, h, l, c, v = b, px, px, px, px, sz
        else:
            h = max(h, px); l = min(l, px); c = px; v += sz
    if bucket is not None:
        yield bucket * bucket_ns, o, h, l, c, v

# usage
ticks = read_ticks("ticks.csv")
big_ticks = above_size(ticks, n=100)
bars = into_bars(big_ticks, bucket_ns=60_000_000_000)   # 1-minute bars

for bar in bars:
    print(bar)
```

That pipeline never materialises the whole tick stream in memory. Each generator pulls one element at a time from its source. You can run it on a 100 GB file with a 12 MB resident footprint.

!!! tip "Pipelines vs DataFrames"
    Generator pipelines win when the data is too big for memory or when latency matters per-element (live feeds). DataFrames win when the data fits and you want bulk vectorised ops. **Real systems use both**: generators upstream to land Arrow / Parquet, DataFrames downstream for analytics.

## Comprehensions are sugar for generators

```python
squares = [x*x for x in range(10)]          # list
sq_set  = {x*x for x in range(10)}          # set
sq_map  = {x: x*x for x in range(10)}       # dict
sq_gen  = (x*x for x in range(10))          # generator expression — lazy!
```

The first three eagerly build a collection. The fourth returns a generator object. Use the generator form when you're feeding the values into a reducer (`sum`, `max`, `any`, `join`) — there's no point materialising:

```python
# good — O(1) memory
total = sum(px * sz for px, sz in trades)

# bad — builds a temp list of N tuples
total = sum([px * sz for px, sz in trades])
```

## `enumerate`, `zip`, and the `itertools` toolkit

Three patterns you should never re-implement:

```python
for i, x in enumerate(xs, start=1):
    ...

for x, y in zip(xs, ys, strict=True):   # strict=True is new in 3.10 — raises on length mismatch
    ...

import itertools as it

list(it.chain([1,2], [3,4]))                 # [1,2,3,4]
list(it.islice(range(100), 5, 10))           # [5,6,7,8,9]
list(it.groupby(sorted(xs), key=...))        # streaming group-by
list(it.combinations("ABCD", 2))             # AB AC AD BC BD CD
list(it.product([1,2], "ab"))                # Cartesian product
list(it.accumulate([3,1,4,1,5], max))        # [3,3,4,4,5]   running max
list(it.takewhile(lambda x: x < 5, [1,3,5,2]))   # [1,3]
list(it.dropwhile(lambda x: x < 5, [1,3,5,2]))   # [5,2]
```

`it.pairwise` (3.10+) is a particular gem — overlapping pairs without zipping the same list twice:

```python
list(it.pairwise([1,2,3,4]))
# [(1,2), (2,3), (3,4)]
```

Use it everywhere you find yourself comparing consecutive elements.

## Coroutines: `send`, `throw`, `close`

A generator is a one-way pipe (out). A *coroutine* is two-way: the consumer can push values back in with `send()`. That's the original coroutine protocol; `asyncio` is built on top.

```python
def averager():
    total, count = 0, 0
    avg = None
    while True:
        value = yield avg
        if value is None:
            break
        total += value
        count += 1
        avg = total / count

a = averager()
next(a)              # prime
print(a.send(10))    # 10.0
print(a.send(20))    # 15.0
print(a.send(30))    # 20.0
a.close()
```

You almost never write hand-rolled coroutines today — `asyncio` and structured concurrency replace them — but understanding `send/throw/close` is what makes async make sense. We'll come back to this in the next chapter.

## Iterators in the wild: pandas's `itertuples`

For row-wise iteration over a DataFrame, `df.itertuples()` is the right tool. It returns a generator of `NamedTuple` rows — far faster than `iterrows()`, which builds a `pd.Series` per row.

```python
for row in df.itertuples(index=True, name="Bar"):
    if row.volume > 1_000_000 and row.close > row.open * 1.02:
        print(row.Index, row.close)
```

This is the right pattern when you *must* iterate (event-driven backtests, online learning loops). For everything else, vectorise.

## A non-obvious pattern: generator-based state machines

Trading logic is full of state machines (orders, positions, sessions). Generators are an under-appreciated way to write them:

```python
def order_lifecycle():
    """Yields the current state; receives external events via send()."""
    state = "PENDING"
    while True:
        event = yield state
        match (state, event):
            case ("PENDING", "ACK"):       state = "WORKING"
            case ("WORKING", "PARTIAL"):   state = "PARTIAL"
            case ("WORKING" | "PARTIAL", "FILL"):  state = "FILLED"; break
            case (_, "CANCEL"):            state = "CANCELED"; break
            case _:
                raise RuntimeError(f"illegal transition: {state} <- {event}")
    yield state   # one last yield with the terminal state


order = order_lifecycle()
print(next(order))           # PENDING
print(order.send("ACK"))     # WORKING
print(order.send("PARTIAL")) # PARTIAL
print(order.send("FILL"))    # FILLED
```

Local variables of the generator hold the state. The event loop just calls `send()` with what just happened. Reads like prose, no enum boilerplate.

## Pitfalls

!!! danger "Iterators are one-shot"
    Once an iterator is exhausted it stays exhausted. If you need to re-iterate, store the source (a list / pandas frame / file path) and call `iter()` again.

!!! warning "Late-bound closures in generator expressions"
    ```python
    gens = [(lambda: i) for i in range(3)]
    [g() for g in gens]   # [2, 2, 2]   — all capture the same `i`
    ```
    Fix with a default argument: `[lambda i=i: i for i in range(3)]`. Or just use a list comprehension over the values directly.

!!! warning "`for/else` on iteration"
    A `for` loop's `else` runs if the loop completes *without `break`*. Genuinely useful for "search" loops; baffling if you didn't know it existed.
    ```python
    for x in xs:
        if predicate(x):
            handle(x); break
    else:
        not_found()
    ```

Continue to **[Async, threads, processes, the GIL](04-concurrency.md)**.
