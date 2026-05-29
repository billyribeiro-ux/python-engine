# Typing for engineers

Python's static typing story has gone from "interesting toy" in 2015 to "the way real code is written" in 2026. This chapter assumes you have used `int`, `str`, `list[T]`, and `Optional`, and takes you to the parts that actually matter for engineering work: protocols, generics, type guards, and the small set of escape hatches you need when types get in the way.

## Why bother

Three things, in order:

1. **The IDE knows what your code does.** Autocompletion, refactors, "find usages" — all dramatically better with types.
2. **You catch a class of bugs at write time.** Type errors stand out before you even run the file.
3. **Types are documentation that doesn't go stale.** A `Callable[[pd.Series], pd.Series]` parameter says more than a docstring sentence.

The cost is real (you have to write the types) but small with modern tooling. Pick `mypy --strict` or `pyright` and you're set.

## The modern syntax (3.10+ refresher)

```python
# Built-ins as generics — no more `from typing import List, Dict, Tuple`
nums: list[int] = [1, 2, 3]
counts: dict[str, int] = {"a": 1}
record: tuple[str, int, float] = ("SPY", 100, 475.30)

# Union with |
maybe_int: int | None = None
either: str | int = 0

# TypeAlias
type Price = float                # 3.12+
type Bars = list[tuple[int, Price, Price, Price, Price, int]]
```

For 3.10/3.11, use `TypeAlias` from `typing` or just `Bars = list[...]` (which works in any version — the explicit alias keyword is the new bit).

## `Protocol` — structural typing, done right

Nominal typing says "this is a `Feed` because it inherits from `Feed`". Structural typing says "this is a `Feed` because it has `bars` and `option_chain` methods with the right shapes". Python's `Protocol` is structural typing, made first-class.

```python
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class HasBars(Protocol):
    def bars(self, symbol: str, start, end, interval: str = "1d") -> pd.DataFrame: ...
```

Any class with a matching `bars` method satisfies `HasBars` — whether it inherits from it or not. The course's `Feed` (Module 0) is a Protocol for exactly this reason: third-party vendors don't have to subclass us; they just have to match the shape.

`@runtime_checkable` adds `isinstance()` support. Without it, the check is type-checker-only (which is often what you want, because runtime `isinstance` on a Protocol is comparatively expensive — it checks each declared method).

### `Protocol` for callbacks

A common use is typing callback functions precisely:

```python
class StrategyCb(Protocol):
    def __call__(self, bars: pd.DataFrame, ctx: dict) -> pd.Series: ...

def backtest(strategy: StrategyCb, *args, **kwargs) -> "Result":
    ...
```

`StrategyCb` documents the *signature* of an acceptable strategy callback, including parameter names. `Callable[[pd.DataFrame, dict], pd.Series]` would do the same thing without the names.

## Generics — `TypeVar`, `Generic`, and the new syntax

The old way:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: list[T] = []
    def push(self, x: T) -> None: self._data.append(x)
    def pop(self) -> T: return self._data.pop()
```

The new way (3.12+):

```python
class Stack[T]:
    def __init__(self) -> None:
        self._data: list[T] = []
    def push(self, x: T) -> None: self._data.append(x)
    def pop(self) -> T: return self._data.pop()
```

Same semantics, no `TypeVar` boilerplate. Bounded type variables:

```python
class Sorted[T: (int, float)]:        # T is constrained to int or float
    ...

class CmpSorted[T: Comparable]:       # T is bounded by a Protocol
    ...
```

### `ParamSpec` for decorators that preserve signatures

A decorator that wraps `f(x: int, y: str) -> bool` should produce a function with the same `(x: int, y: str) -> bool` signature, not `(*args, **kwargs) -> Any`. `ParamSpec` is how:

```python
from typing import ParamSpec, TypeVar
from functools import wraps
import time

P = ParamSpec("P")
R = TypeVar("R")

def timed[**P, R](label: str):       # 3.12+ syntax for ParamSpec
    def deco(fn):
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                print(f"{label}: {(time.perf_counter() - t0)*1000:.1f} ms")
        return wrapper
    return deco
```

With `ParamSpec`, the IDE shows the *real* signature of the decorated function.

## `TypeGuard` and `TypeIs` — narrowing types in user-defined checks

`isinstance(x, int)` narrows `x: int | str` to `int` in the `if` branch. You can write your own narrowers:

```python
from typing import TypeGuard

def is_non_empty_str(x: object) -> TypeGuard[str]:
    return isinstance(x, str) and len(x) > 0

def f(x: str | None):
    if is_non_empty_str(x):
        # mypy knows x is str here
        print(x.upper())
```

`TypeIs` (3.13+) is the stricter, two-directional version (`PEP 742`). When the guard returns False, the false branch is narrowed too. Use `TypeIs` when your check is genuinely symmetric.

## `Self` — instead of forward-referencing your own class

The old way:

```python
class Builder:
    def step(self) -> "Builder": ...
```

The new way (3.11+):

```python
from typing import Self

class Builder:
    def step(self) -> Self: ...
```

Subclasses now correctly get *their own type back* from the method, not the parent type. Essential for fluent APIs.

## `Annotated` — extra metadata that survives `mypy`

`Annotated[T, ...]` attaches arbitrary metadata to a type without changing what `T` is. The metadata is invisible to type checkers but readable at runtime via `typing.get_type_hints(..., include_extras=True)`. This is the foundation of FastAPI, pydantic, and msgspec:

```python
from typing import Annotated
from dataclasses import dataclass

@dataclass
class Order:
    symbol: Annotated[str, "uppercase ticker"]
    qty:    Annotated[int, "non-zero, signed"]
    price:  Annotated[float, "limit price; positive"]
```

For trading code, `Annotated` is useful for **unit-tagged numbers**: `Annotated[float, "bps"]` vs `Annotated[float, "pct"]`. The type checker won't catch a mismatch (both are `float`), but the metadata documents the intent and survives into pydantic-style validators.

## `Never` and `NoReturn` — for impossible branches

```python
from typing import Never, assert_never

def handle(event: "ACK | CANCEL | FILL"):
    match event:
        case "ACK":     ...
        case "CANCEL":  ...
        case "FILL":    ...
        case _ as unhandled:
            assert_never(unhandled)   # type-checker error if any case missing
```

`assert_never` is brilliant for exhaustiveness checks in pattern matching. Add a new variant to the type, forget a `case`, mypy will tell you which file to fix.

## `Literal` — values as types

When a parameter has a small fixed set of valid values, `Literal` is far more informative than `str`:

```python
from typing import Literal

Interval = Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]

def bars(symbol: str, interval: Interval = "1d") -> pd.DataFrame:
    ...

bars("SPY", "1d")     # ok
bars("SPY", "2d")     # mypy: Argument of type "Literal['2d']" cannot be assigned to "Interval"
```

We use this pattern throughout the course (notice `Interval` in `engine.data.feed`).

## `dataclasses` and types

```python
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True, kw_only=True)
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    extras: dict[str, float] = field(default_factory=dict)
```

`kw_only=True` (3.10+) forces all fields to be keyword-only — fewer positional-argument bugs in long dataclasses. We cover this and the alternatives in detail in the next chapter.

## `TypedDict` — typing the dictionaries you can't replace

You'll have legacy dicts. `TypedDict` types them without rewriting:

```python
from typing import TypedDict

class TickJson(TypedDict):
    ts: int
    px: float
    sz: int

def process(t: TickJson) -> None:
    print(t["px"])   # mypy knows the keys and their types
```

For nested or partial dicts, look at `total=False` and `NotRequired`.

## Type-checking tools

- **`mypy`** — the original, slow but battle-tested. Run with `--strict`.
- **`pyright`** — Microsoft's. Faster, smarter inference, better Protocol support. Built into VS Code's Pylance.
- **`ty`** (formerly `red-knot`) — Astral's new Rust-based checker. Worth watching; not 1.0 yet at the time of writing.

Pick one, commit to it. Mixing checkers leads to "works in one, fails in the other" headaches.

## Escape hatches (use sparingly)

```python
from typing import Any, cast

x: Any = some_external_thing()     # opt out — the value can be anything
y = cast(int, some_external_thing()) # tell the checker, no runtime cost
```

`Any` should be rare and intentional. Each one is a place where the type system has given up. If you find an `Any` near a frequent bug source, that's where to invest in types next.

## A real example: a typed strategy interface

```python
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class Strategy(Protocol):
    """A strategy reads bars and produces a target position series.

    `bars` is the universe DataFrame indexed by timestamp, columns of symbols.
    The returned Series is indexed identically and bounded in [-1, 1] (Kelly
    fraction or notional fraction).
    """
    name: str

    def signals(self, bars: pd.DataFrame) -> pd.Series: ...


class MomentumXSec:
    name = "momentum-xsec"
    def __init__(self, lookback: int = 60): self.lookback = lookback
    def signals(self, bars: pd.DataFrame) -> pd.Series:
        ret = bars.pct_change(self.lookback)
        ranks = ret.rank(axis=1, pct=True) - 0.5
        return ranks.iloc[-1]                # target position per symbol

assert isinstance(MomentumXSec(), Strategy)   # passes — structural match
```

`Strategy` is a contract, not a base class. Any third party can write a strategy without importing your library.

Continue to **[Dataclasses, attrs, pydantic](06-dataclasses.md)**.
