# The data model

Python's data model is the set of rules that say what an *object* is and how the language interacts with it. Almost every "advanced" feature you'll meet (decorators, context managers, async, comprehensions, even `import`) is a syntactic surface over the same underlying protocol-based model. Internalise the model and the language stops surprising you.

## Everything is an object — say it three times

In Python, every value is an object. Including:

- integers (`int` instances)
- functions (`function` instances)
- classes themselves (`type` instances)
- modules (`module` instances)
- `None` (the unique `NoneType` instance)
- exceptions, generators, coroutines, slices, ellipsis...

Every object has three things: an **identity** (its address in memory, retrievable via `id()`), a **type** (`type(obj)`), and a **value**. The first two are immutable. The third is mutable iff the type says so.

```python
x = 42
print(id(x), type(x), x)
#  140702342234176 <class 'int'> 42

def f(): pass
print(type(f), type(type(f)))
# <class 'function'> <class 'type'>
```

Note the second line. The type of `f` is `function`. The type of `function` is `type` — because classes are themselves objects whose type is `type`. This is **not** academic trivia; it's the foundation that makes metaclasses and class decorators possible.

## Dunder methods are the data model's contract

The data model is expressed through **dunder** methods — names with double leading and trailing underscores. When you write `a + b`, Python looks up `type(a).__add__` and calls it with `b`. When you write `len(x)`, Python looks up `type(x).__len__` and calls it. When you write `for v in xs:`, Python looks up `type(xs).__iter__`.

You almost never call dunders directly. You make your class **support** the operations by *defining* the dunders. That's it. That's the whole protocol-based design.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    cents: int
    ccy: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if self.ccy != other.ccy:
            raise ValueError("currency mismatch")
        return Money(self.cents + other.cents, self.ccy)

    def __mul__(self, k: int) -> "Money":
        return Money(self.cents * k, self.ccy)

    def __repr__(self) -> str:
        return f"${self.cents/100:.2f}"

print(Money(1050) + Money(250))       # $13.00
print(Money(199) * 3)                 # $5.97
```

You wrote two operators and a printable representation. The language now treats `Money` like a first-class numeric type for the operations you cared about — and only for those. The protocol is opt-in.

## The most useful dunders

You won't need them all, but the following are the ones a senior Python engineer should know cold.

### Construction and lifecycle

| Method | When called |
|---|---|
| `__new__(cls, ...)` | Object creation. Returns the new instance. Rarely overridden. |
| `__init__(self, ...)` | Initialiser. Sets attributes on the new instance. |
| `__del__(self)` | Finaliser. Called by the GC. Avoid logic here; use context managers. |
| `__init_subclass__(cls, **kw)` | Called on the *parent* when a subclass is created. The cleanest way to build a plugin registry. |
| `__set_name__(self, owner, name)` | Called when a descriptor is assigned to a class attribute. |

### Representation

| Method | When called |
|---|---|
| `__repr__(self)` | `repr(obj)` and the REPL. Should be unambiguous; ideally `eval`-able. |
| `__str__(self)` | `str(obj)` and `print(obj)`. Human-friendly. Falls back to `__repr__`. |
| `__format__(self, spec)` | `f"{obj:spec}"` and `format(obj, spec)`. |
| `__bytes__(self)` | `bytes(obj)`. |

### Arithmetic and comparison

`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`, `__matmul__` (`@`), and their reflected (`__radd__` etc.) and in-place (`__iadd__` etc.) versions. Comparisons: `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`. `__hash__` if instances are to live in sets/dicts.

!!! warning "Hash and equality must agree"
    If you override `__eq__`, you **must** override `__hash__` (set it to `None` to make instances unhashable, or implement it to be consistent with `__eq__`). The default rule is: equal objects have equal hashes. Violate it and dict lookups silently lie.

### Containers and iteration

| Method | Purpose |
|---|---|
| `__len__` | `len(x)` |
| `__getitem__(self, k)` | `x[k]` |
| `__setitem__(self, k, v)` | `x[k] = v` |
| `__delitem__(self, k)` | `del x[k]` |
| `__contains__(self, v)` | `v in x` (falls back to iteration if absent) |
| `__iter__(self)` | `for v in x` |
| `__next__(self)` | next element of an iterator |
| `__reversed__(self)` | `reversed(x)` |

### Attribute access

| Method | When called |
|---|---|
| `__getattr__(self, name)` | Called *only* when normal lookup fails. |
| `__getattribute__(self, name)` | Called for *every* attribute access. Don't override unless you really mean it. |
| `__setattr__(self, name, value)` | Every attribute assignment. |
| `__delattr__(self, name)` | `del obj.name`. |
| `__dir__(self)` | `dir(obj)`. |

### Callable and context

- `__call__(self, ...)` — `obj(...)`. Makes instances callable.
- `__enter__(self) / __exit__(self, et, ev, tb)` — `with obj:`.
- `__aenter__ / __aexit__` — `async with obj:`.

### Async iteration

- `__aiter__ / __anext__` — `async for v in obj:`.
- `__await__` — `await obj`.

## A worked example: a tick-by-tick replay object

We want an object that:

- holds a DataFrame of ticks,
- iterates them in time order,
- supports `len`, `repr`, slicing by date range,
- works as a context manager that closes its source on exit.

That's a lot of features. Eight dunders later, here it is:

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class TickReplay:
    """An iterable, sliceable, printable replay of tick data from a Parquet file."""
    path: Path
    _df: pd.DataFrame | None = None  # lazy

    def __enter__(self) -> "TickReplay":
        self._df = pd.read_parquet(self.path)
        self._df.index = pd.to_datetime(self._df.index, utc=True)
        return self

    def __exit__(self, *exc) -> None:
        self._df = None

    def _frame(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Use TickReplay inside a `with` block")
        return self._df

    def __len__(self) -> int:
        return len(self._frame())

    def __iter__(self):
        # itertuples avoids per-row Series construction
        yield from self._frame().itertuples(index=True, name="Tick")

    def __getitem__(self, key: slice) -> pd.DataFrame:
        if not isinstance(key, slice):
            raise TypeError("Only slice access supported, e.g. tr['2024-01-02':'2024-01-03']")
        return self._frame().loc[key.start:key.stop]

    def __repr__(self) -> str:
        if self._df is None:
            return f"TickReplay(path={self.path!r}, closed)"
        df = self._df
        return f"TickReplay({self.path.name}: {len(df):,} ticks, {df.index.min()} → {df.index.max()})"
```

```python
# usage
# with TickReplay(Path("ticks.parquet")) as tr:
#     print(tr)                              # TickReplay(ticks.parquet: 1,402,991 ticks, ...)
#     window = tr["2024-01-02":"2024-01-03"]
#     for tick in tr:
#         ...
```

That's the data model paying for itself: the object is **idiomatic Python** from the caller's perspective, because we wrote the protocols the language was already going to look for.

## `__slots__` — when it matters

By default, every Python object has a `__dict__` that holds its attributes. That's flexible but expensive: a dict per instance, plus the cost of dict lookups on attribute access.

`__slots__` opts out of `__dict__`. The class declares its attribute names; the interpreter allocates fixed offsets and uses descriptors for access. Three consequences:

1. **Smaller**. A slotted instance is often 5–10× smaller than the equivalent dict-backed one.
2. **Faster attribute access** (especially under PyPy / free-threaded CPython).
3. **No surprise attributes**. Typos that would silently create new attributes now raise `AttributeError`.

```python
class Tick:
    __slots__ = ("ts", "px", "size")

    def __init__(self, ts, px, size):
        self.ts = ts
        self.px = px
        self.size = size

t = Tick(1_700_000_000, 100.5, 200)
# t.typo = 1   # AttributeError
```

For dataclasses, pass `slots=True`:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Quote:
    ts: int
    bid: float
    ask: float
```

!!! tip "When to use slots"
    Always, for value-object-like classes you create in large quantities (ticks, quotes, orders, bars). The marginal cost is one line. The savings, on a million-row pipeline, is real.

!!! warning "Slots and multiple inheritance"
    Slots interact badly with multiple inheritance when both parents declare slots. The rule: only one base class in a slots-using hierarchy can have non-empty `__slots__` per attribute. If you find yourself fighting this, you're doing OO wrong for the problem.

## The MRO and `super()`

The **method resolution order** (MRO) is the linear order in which Python searches base classes for an attribute. Python uses **C3 linearisation**, which guarantees a consistent, deterministic order (and refuses to compute one when the hierarchy is internally inconsistent).

```python
class A:
    def hello(self): print("A")

class B(A):
    def hello(self):
        print("B"); super().hello()

class C(A):
    def hello(self):
        print("C"); super().hello()

class D(B, C):
    def hello(self):
        print("D"); super().hello()

print(D.__mro__)
# (D, B, C, A, object)

D().hello()
# D
# B
# C
# A
```

Two things to notice:

1. `super()` does **not** mean "the parent class". It means "the next class in the MRO". In `B.hello`, `super().hello()` calls `C.hello`, not `A.hello`, because `C` comes next.
2. The MRO is what makes **cooperative multiple inheritance** possible. Every class calls `super()` and the language threads the calls through the linearisation.

In production trading code, deep inheritance trees are usually a mistake. But you'll meet them in `torch.nn`, in framework code, and in the standard library — understanding the MRO is what stops you breaking those frameworks accidentally.

## The cheap mental model

Whenever Python does something interesting, the rule is:

> *Some piece of syntax desugars to a dunder method call on the object's type.*

- `a + b` → `type(a).__add__(a, b)` (with reflected fallback to `type(b).__radd__`)
- `for x in xs` → `it = type(xs).__iter__(xs); while True: yield type(it).__next__(it)` (until `StopIteration`)
- `with obj as v` → `type(obj).__enter__(obj)` ... `type(obj).__exit__(obj, *)`
- `obj()` → `type(obj).__call__(obj, ...)`
- `obj.x` (when `x` is a descriptor on the class) → `descriptor.__get__(obj, type(obj))`

Once you can see the desugaring, the magic disappears and you start writing types that fit the language instead of fighting it.

Continue to **[Descriptors and dunder methods](02-descriptors.md)**.
