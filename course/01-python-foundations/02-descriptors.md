# Descriptors and dunder methods

A **descriptor** is any object that defines `__get__`, `__set__`, or `__delete__`. Descriptors are how `@property`, `@classmethod`, `@staticmethod`, slots, and even bound methods all work under the hood. They're a small, sharp tool, and once you've seen them you'll keep reaching for them.

This chapter is about descriptors and a handful of other dunder tricks that show up in real engineering code.

## The descriptor protocol, in 30 seconds

When you access `obj.x`:

1. Python looks up `x` on `type(obj)` and its MRO.
2. If the result is a **data descriptor** (defines `__set__` or `__delete__`), it's invoked. Done.
3. Otherwise Python checks `obj.__dict__["x"]`.
4. Otherwise it falls back to a **non-data descriptor** (defines only `__get__`) found in step 1.
5. Otherwise it raises `AttributeError` (or calls `__getattr__` if defined).

```python
class TypedField:
    """A descriptor that enforces a type at assignment time."""

    def __init__(self, type_: type):
        self.type = type_

    def __set_name__(self, owner, name):
        # Called once, when the descriptor is bound to a class attribute.
        self.private = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private)

    def __set__(self, obj, value):
        if not isinstance(value, self.type):
            raise TypeError(f"expected {self.type.__name__}, got {type(value).__name__}")
        setattr(obj, self.private, value)


class Order:
    qty   = TypedField(int)
    price = TypedField(float)

    def __init__(self, qty, price):
        self.qty = qty       # routed through TypedField.__set__
        self.price = price

o = Order(100, 50.25)
# Order(qty="oops", price=10.0)   # TypeError: expected int, got str
```

You wrote a tiny class and got runtime-checked typed fields with private backing storage, for *any* class that uses them. That's the descriptor pattern.

## `property` is just a descriptor

Once you've seen `TypedField`, `@property` is unsurprising:

```python
class Position:
    def __init__(self, qty: int, avg_px: float):
        self._qty = qty
        self._avg_px = avg_px

    @property
    def notional(self) -> float:
        return self._qty * self._avg_px

    @notional.setter
    def notional(self, value: float) -> None:
        # rebalance qty around fixed avg_px (toy example)
        self._qty = int(round(value / self._avg_px))
```

`@property` builds a descriptor with `__get__` calling the getter, `__set__` calling the setter (or raising `AttributeError` if none), and `__delete__` similarly. Nothing more.

## `cached_property` — the lazy-evaluation workhorse

A `cached_property` is a non-data descriptor that computes a value once and stores it in the instance `__dict__`. Subsequent lookups skip the descriptor (because the value lives directly on the instance now).

```python
from functools import cached_property
import pandas as pd

class BarFrame:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @cached_property
    def returns(self) -> pd.Series:
        return self.df["close"].pct_change()

    @cached_property
    def log_returns(self) -> pd.Series:
        import numpy as np
        return np.log(self.df["close"]).diff()
```

The first access to `bf.returns` computes the value and stashes it in `bf.__dict__["returns"]`. The second access doesn't even hit the descriptor — Python finds the cached value in step 3 of the lookup above.

!!! warning "`cached_property` and slots"
    Slotted classes have no `__dict__`, so `cached_property` can't stash anything. Solutions: (a) drop slots; (b) use `functools.lru_cache` on a method; (c) hand-roll a cache attribute in `__slots__`. (a) is usually right.

## `__init_subclass__` — the plugin pattern

Probably the single best dunder for *real* engineering work. Defined on the parent, it runs whenever a subclass is created.

```python
class Scanner:
    _registry: dict[str, type["Scanner"]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        if key in Scanner._registry:
            raise ValueError(f"duplicate scanner: {key}")
        Scanner._registry[key] = cls

    def scan(self, universe):
        raise NotImplementedError


class GammaSqueeze(Scanner, name="gex"):
    def scan(self, universe):
        ...


class IVCrush(Scanner, name="iv-crush"):
    def scan(self, universe):
        ...


# Anywhere later in the codebase:
print(Scanner._registry)
# {'gex': <class GammaSqueeze>, 'iv-crush': <class IVCrush>}
```

You now have an auto-populating plugin registry. No decorator boilerplate. No central list. Add a class, it's registered. The course's scanner framework (Module 18) is built on exactly this pattern.

## `__class_getitem__` — generic syntax on your own types

`list[int]`, `dict[str, float]`, `pd.Series[float]` (in newer pandas) — all of these are calls to `__class_getitem__` on the class. Adding it to your own class lets you write generic-looking types:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: list[T] = []
    def push(self, x: T) -> None:
        self._data.append(x)
    def pop(self) -> T:
        return self._data.pop()

stack: Stack[int] = Stack()
stack.push(1)
```

You don't write `__class_getitem__` by hand — `Generic` does it for you. But knowing the dunder exists means `Stack[int]` stops being magic and starts being a regular method call.

## `__getattr__` vs `__getattribute__`

Confusing names, different jobs.

- **`__getattribute__`** is called on *every* attribute access. Overriding it is rarely a good idea.
- **`__getattr__`** is called *only when normal lookup fails*. This is what you want 95% of the time.

Practical use: a thin wrapper around an external library that forwards unknown attribute access:

```python
class FeedShim:
    """Forward unknown calls to the underlying feed; add timing to known ones."""

    def __init__(self, feed):
        self._feed = feed

    def bars(self, *args, **kwargs):
        import time
        t0 = time.perf_counter()
        out = self._feed.bars(*args, **kwargs)
        print(f"bars({args[0]}) took {(time.perf_counter() - t0)*1000:.1f} ms")
        return out

    def __getattr__(self, name):
        # only triggered when normal lookup failed → forward to the wrapped feed
        return getattr(self._feed, name)
```

## `__slots__` and descriptors

Slots are themselves descriptors. When you declare `__slots__ = ("x", "y")`, Python creates two slot descriptors on the class. That's why slotted attribute access is fast (descriptor call into a fixed memory offset) and why subclassing slotted classes is awkward (the child needs its own slots).

## `__init_subclass__` + `__set_name__` together

The two most underused dunders, often combined to make domain-specific declarations look like data:

```python
from datetime import datetime

class Field:
    def __init__(self, kind: type, required: bool = True):
        self.kind = kind
        self.required = required

    def __set_name__(self, owner, name):
        self.name = name
        owner._fields[name] = self


class Record:
    _fields: dict[str, Field] = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._fields = {}
        # Walk class attributes; descriptors that defined __set_name__
        # already registered themselves. Nothing else to do here.

    def __init__(self, **values):
        for name, field in self._fields.items():
            if field.required and name not in values:
                raise ValueError(f"missing field: {name}")
            v = values.get(name)
            if v is not None and not isinstance(v, field.kind):
                raise TypeError(f"{name}: expected {field.kind}, got {type(v)}")
            setattr(self, name, v)

    def __repr__(self):
        kv = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields)
        return f"{type(self).__name__}({kv})"


class Fill(Record):
    ts:      datetime = Field(datetime)
    symbol:  str      = Field(str)
    qty:     int      = Field(int)
    px:      float    = Field(float)


f = Fill(ts=datetime.utcnow(), symbol="SPY", qty=100, px=475.30)
print(f)
# Fill(ts=datetime.datetime(...), symbol='SPY', qty=100, px=475.3)
```

You've just built a tiny declarative DSL — `pydantic` and `attrs` are this pattern at scale.

## `__matmul__` and `@` — the operator nobody uses

Python's `@` operator desugars to `__matmul__`. It exists for numerical libraries (matrix multiplication: `A @ B`). For a quant codebase, it's perfect for composing strategies or pipelines:

```python
class Pipeline:
    def __init__(self, *steps): self.steps = steps

    def __call__(self, x):
        for s in self.steps:
            x = s(x)
        return x

    def __matmul__(self, other):
        # pipe `self` into `other`: (other @ self) is read "other after self"
        return Pipeline(self, other)


zscore = Pipeline(lambda x: (x - x.mean()) / x.std())
clip   = Pipeline(lambda x: x.clip(-3, 3))

normaliser = clip @ zscore   # zscore then clip — reads bottom-up like math
```

Use sparingly. The operator is foreign to most readers; reserve it for cases where it genuinely makes the code clearer.

## When NOT to use descriptors

Descriptors are a sharp tool. Use them when:

- You need the same attribute behaviour on **many classes**.
- You need the behaviour at **the language level** (typed assignment, lazy computation, immutability).

Avoid them when:

- A plain method or function is sufficient.
- The reader has to read three files to understand a simple attribute access.

The senior-engineer rule: **prefer boring code**. Descriptors are a deliberate departure from boring, and the departure should pay for itself.

Continue to **[Iteration, generators, coroutines](03-iteration.md)**.
