# The Scanner framework

`engine.scanners` is a tiny framework: a Protocol that defines what a scanner looks like, a decorator that registers it by name, and a registry singleton you query at runtime. Forty lines of code, but it's the structural backbone that keeps all the strategies in Module 18 manageable.

## The Protocol

```python
from typing import Protocol, runtime_checkable
from collections.abc import Sequence
import pandas as pd

@runtime_checkable
class Scanner(Protocol):
    name: str
    def scan(self, universe: Sequence[str], asof: pd.Timestamp) -> pd.DataFrame: ...
```

Two things: a `name`, and a `scan(universe, asof)` method that returns a DataFrame of candidates with their per-symbol metric values.

`@runtime_checkable` means `isinstance(obj, Scanner)` works without needing to subclass — duck typing made formal.

## The decorator and registry

```python
class _Registry:
    def __init__(self) -> None:
        self._data: dict[str, type] = {}
    def register(self, name: str, cls: type) -> type:
        if name in self._data:
            raise ValueError(f"duplicate scanner: {name}")
        self._data[name] = cls
        return cls
    def get(self, name: str) -> type:
        if name not in self._data:
            raise KeyError(f"no scanner registered as {name!r}")
        return self._data[name]
    def names(self) -> list[str]:
        return sorted(self._data)

registry = _Registry()


def scanner(name: str):
    def deco(cls: type) -> type:
        if not hasattr(cls, "scan"):
            raise TypeError(f"{cls!r} must implement scan(universe, asof)")
        cls.name = name
        return registry.register(name, cls)
    return deco
```

Total: ~30 lines. Imported as `from engine.scanners import scanner, registry, Scanner`.

## Writing a scanner

```python
import pandas as pd
import numpy as np
from engine.scanners import scanner


@scanner("zscore-momentum")
class ZScoreMomentumScanner:
    """Flags symbols whose recent return z-score exceeds a threshold."""
    def __init__(self, lookback: int = 60, threshold: float = 2.0):
        self.lookback = lookback
        self.threshold = threshold

    def scan(self, universe, asof: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for sym in universe:
            data = fetch_close(sym)        # your data adapter
            returns = data.pct_change().dropna()
            recent = returns.iloc[-self.lookback:]
            z = (recent.iloc[-1] - recent.mean()) / recent.std()
            if abs(z) > self.threshold:
                rows.append({"symbol": sym, "z": float(z), "direction": int(np.sign(z))})
        return pd.DataFrame(rows).sort_values("z", key=abs, ascending=False)
```

That's it. Decorated, registered, callable through the registry.

## Running scanners

```python
from engine.scanners import registry

universe = ["SPY", "QQQ", "IWM", "TLT", "GLD", "AAPL", "MSFT", ...]

for name in registry.names():
    cls = registry.get(name)
    scanner_obj = cls()
    candidates = scanner_obj.scan(universe, asof=pd.Timestamp.utcnow())
    print(f"\n{name}: {len(candidates)} candidates")
    print(candidates.head(10))
```

The runtime is one loop. Adding a new scanner = adding a new file that imports `scanner` and decorates a class.

## A scanner-loader for production

For deployment, a configuration-driven loader:

```python
import yaml
from engine.scanners import registry

def load_scanners_from_yaml(path: str) -> list:
    """YAML: list of {kind: name, params: {...}}."""
    spec = yaml.safe_load(open(path))
    return [registry.get(s["kind"])(**s.get("params", {})) for s in spec]


# scanners.yaml:
# - kind: zscore-momentum
#   params: {lookback: 60, threshold: 2.5}
# - kind: gex-inflection
#   params: {underlying: SPY}
```

This is the pattern most production scanner platforms use: code defines what scanners exist; YAML defines which are active and with what parameters. Change YAML, restart, done.

## Async batch scanning

For 5,000 symbols, sequential scanning is slow. Use the Module 5 async pattern:

```python
import asyncio
from engine.scanners import registry

async def run_scanner_async(scanner_obj, universe, asof):
    return await asyncio.to_thread(scanner_obj.scan, universe, asof)


async def run_all_scanners(scanner_names: list[str], universe, asof):
    tasks = [run_scanner_async(registry.get(name)(), universe, asof) for name in scanner_names]
    return dict(zip(scanner_names, await asyncio.gather(*tasks), strict=True))
```

Now all scanners run in parallel. For CPU-bound scanners, swap `to_thread` for `concurrent.futures.ProcessPoolExecutor`.

## Testing scanners

Every scanner should have a stub-feed test that doesn't require network:

```python
def test_my_scanner_with_stub_data():
    scanner = ZScoreMomentumScanner(lookback=60, threshold=2.0)
    # Hand the scanner a controlled fixture (no real data)
    # ...
    assert "symbol" in scanner.scan(...).columns
```

The course's `engine.scanners` tests (`tests/test_scanners.py`) verify the framework's registry and Protocol semantics. Scanner-specific tests should verify the scanner produces sensible output on known inputs.

## Output schema discipline

Every scanner returns a DataFrame with at least:

- **`symbol`** — string.
- **A score column** — float, sort order.
- **A direction column** — int in {-1, 0, +1}, optional.
- **Metadata columns** — anything else relevant to the alert.

This consistency means downstream consumers (filters, alerting, aggregation) don't have to special-case each scanner.

## Bottom line

The scanner framework is intentionally tiny:

- **`@scanner("name")`** decorator + registry for plugin-style addition.
- **Single `scan(universe, asof) → DataFrame`** Protocol method.
- **Auto-collected** in the registry; iterable; testable.
- **YAML-configurable** at deployment.
- **Async-batchable** for large universes.

Add a new strategy → write a scanner → it shows up in the registry → it runs in the daily batch.

Continue to **[Production scanners](02-production.md)**.
