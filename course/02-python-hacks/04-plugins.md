# Plugin patterns and registries

Most non-trivial trading systems benefit from a plugin architecture: you have a base concept (a strategy, a scanner, a data feed) and many concrete implementations. The code that runs them shouldn't have to know each one explicitly.

This chapter shows the three idiomatic Python patterns for plugins, and when each is right.

## Pattern 1 — the `__init_subclass__` registry

Already covered in Module 1, but worth restating here as the **default** answer.

```python
from typing import ClassVar

class Strategy:
    _registry: ClassVar[dict[str, type["Strategy"]]] = {}

    def __init_subclass__(cls, *, name: str | None = None, **kw):
        super().__init_subclass__(**kw)
        key = name or cls.__name__
        if key in Strategy._registry:
            raise ValueError(f"duplicate strategy: {key}")
        Strategy._registry[key] = cls

    def signals(self, bars):
        raise NotImplementedError


class CrossSectionalMomentum(Strategy, name="xsec-mom"):
    def signals(self, bars):
        ...


class KalmanPairs(Strategy, name="kalman-pairs"):
    def signals(self, bars):
        ...


def make_strategy(name: str) -> Strategy:
    return Strategy._registry[name]()
```

Use when: all plugins live in your codebase or in code you import explicitly. Simplest, no extra dependencies, type-safe.

## Pattern 2 — explicit decorator registry

When `__init_subclass__` feels like too much magic (or when the things you're registering aren't classes — they could be functions), an explicit decorator is just as clean:

```python
from typing import Callable

_SCANNERS: dict[str, Callable] = {}

def scanner(name: str):
    def deco(fn):
        if name in _SCANNERS:
            raise ValueError(f"duplicate scanner: {name}")
        _SCANNERS[name] = fn
        return fn
    return deco


@scanner("gex-inflection")
def gex_inflection(universe, asof):
    """Flag tickers where dealer gamma exposure is near a sign flip."""
    ...

@scanner("iv-crush")
def iv_crush(universe, asof):
    """Flag tickers with high implied move and low realized move."""
    ...
```

The pattern is obvious to a reader: "this function is a scanner named `gex-inflection`." Use when the plugins are functions, when you want the registration site to be visible, or when the team prefers explicit over implicit.

## Pattern 3 — `entry_points` for third-party plugins

When plugins live in **separate packages** that you don't import explicitly, you want Python's [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) mechanism. The package declares a plugin in its `pyproject.toml`; your code loads them at runtime.

In `pyproject.toml` of a third-party `super_scanner` package:

```toml
[project.entry-points."python_engine.scanners"]
super_scanner = "super_scanner.module:SuperScanner"
```

In your loader code:

```python
from importlib.metadata import entry_points

def load_scanners():
    out = {}
    for ep in entry_points(group="python_engine.scanners"):
        out[ep.name] = ep.load()       # imports and returns the class
    return out
```

Now any user can `pip install super-scanner` and your runtime picks it up — no source changes. This is how `pytest` plugins, `flake8` extensions, and `setuptools` plugins work.

Use when: you genuinely have third parties writing extensions, and you want them to ship as separate pip packages.

## Pattern 4 — config-driven instantiation

The "plugin" is a class name + kwargs in YAML/TOML:

```yaml
# strategies.yaml
- kind: xsec-mom
  params: { lookback: 60, top_n: 50 }

- kind: kalman-pairs
  params: { half_life: 22, hedge: "ols" }
```

```python
import yaml

def build_strategies(path: str) -> list[Strategy]:
    spec = yaml.safe_load(open(path))
    return [Strategy._registry[s["kind"]](**s["params"]) for s in spec]
```

Combined with one of the previous patterns, this gives you a strategy lab where you edit YAML, not Python, to compose runs.

## Avoid: silent monkey-patching

It is possible to mutate someone else's class or module from your own at import time. It is almost always a bad idea.

```python
# DON'T do this in production code
import some_lib
original = some_lib.SomeClass.method
def patched(self, *a, **kw):
    print("intercepted")
    return original(self, *a, **kw)
some_lib.SomeClass.method = patched
```

The intercepting effect happens by import side-effect, two engineers away, on a Tuesday. The next person debugging it has no idea where the "intercepted" log line comes from.

**The exception**: testing. `unittest.mock.patch` or `pytest`'s `monkeypatch` fixture is the right way, scoped to the test:

```python
def test_with_patched_feed(monkeypatch):
    monkeypatch.setattr("engine.data.YFinanceFeed.bars", lambda *a, **kw: fixture_df)
    ...
```

The patch dies with the test. That's the difference between a debugging tool and a footgun.

## The full picture: a strategy registry the course uses

Putting it all together, here's the pattern the course uses in `engine.scanners`:

```python
# engine/scanners/base.py
from __future__ import annotations
from typing import ClassVar, Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class Scanner(Protocol):
    name: str
    def scan(self, universe: list[str], asof: pd.Timestamp) -> pd.DataFrame: ...


class _Registry:
    """Singleton-ish registry of scanners."""
    _data: dict[str, type] = {}

    def register(self, name: str, cls: type) -> type:
        if name in self._data:
            raise ValueError(f"duplicate scanner: {name}")
        self._data[name] = cls
        return cls

    def get(self, name: str) -> type:
        if name not in self._data:
            raise KeyError(f"no scanner registered as {name!r}; have {list(self._data)}")
        return self._data[name]

    def names(self) -> list[str]:
        return sorted(self._data)

registry = _Registry()


def scanner(name: str):
    """Decorator: register a Scanner subclass under `name`."""
    def deco(cls: type) -> type:
        assert isinstance(cls, type) and hasattr(cls, "scan"), \
            f"{cls!r} doesn't satisfy the Scanner protocol"
        cls.name = name
        return registry.register(name, cls)
    return deco
```

```python
# engine/scanners/gex.py
from engine.scanners.base import scanner

@scanner("gex-inflection")
class GammaExposureInflection:
    def scan(self, universe, asof):
        ...
```

```python
# usage
from engine.scanners.base import registry

scanner_cls = registry.get("gex-inflection")
results = scanner_cls().scan(universe=["SPY", "QQQ"], asof=pd.Timestamp.utcnow())
```

You get: protocol-based duck typing, an explicit decorator with a useful name, a registry that's also iterable for "list all scanners", and a clean failure mode for duplicates. The same shape works for strategies, feature pipelines, and signal aggregators.

## End of Module 2

You now have a kit of patterns that, once internalised, change how you write Python. The next module is NumPy — where vectorisation, broadcasting, and `einsum` will buy you 100× speedups on the kinds of operations every trading pipeline performs constantly.

Continue to **[Module 3 — NumPy Mastery](../03-numpy-mastery/index.md)**.
