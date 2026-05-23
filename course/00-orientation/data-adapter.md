# The universal data adapter

Every example in this course gets its data through one tiny interface. That interface is the most important piece of design in the whole codebase, because it's what lets you swap a free flaky source for an expensive reliable one **without changing any of the strategies, scanners, or models** you'll write.

## The interface

```python
# engine/data/feed.py — the heart of the abstraction
class Feed(Protocol):
    name: str
    def bars(self, symbol: str, start, end, interval="1d") -> pd.DataFrame: ...
    def option_chain(self, symbol: str, expiry=None) -> OptionsChain: ...
```

Two methods. That's it. Whatever vendor you point at, the returned shapes are identical.

## Returned shapes

`bars` always returns a DataFrame with:

- **Index**: tz-aware `DatetimeIndex` named `timestamp` (UTC).
- **Columns** (lower-case, in this order): `open, high, low, close, volume`.

`option_chain` returns an `OptionsChain` dataclass with `spot`, `asof`, `calls`, and `puts`. The two chain DataFrames share the columns `strike, last, bid, ask, volume, open_interest, implied_volatility, in_the_money, expiry`.

That's the **contract**. Vendor weirdness — capitalisation, multi-index columns, timezone in local exchange time, weird names like `adjClose` — is hidden inside the adapter and never leaks out.

## Using it

```python
from engine.data import YFinanceFeed

feed = YFinanceFeed()
spy = feed.bars("SPY", "2024-01-01", "2024-06-01", interval="1d")
print(spy.tail())

chain = feed.option_chain("SPY")
print(chain.calls.head())
print(f"Spot at snapshot: {chain.spot:.2f}")
```

## Switching vendors

Two equally valid patterns:

### Pattern 1 — name the class explicitly

```python
from engine.data import PolygonFeed       # change one line
feed = PolygonFeed()                       # your strategy code below is unchanged
```

### Pattern 2 — env-driven default

```python
from engine.data.feed import default_feed
feed = default_feed()                     # respects PYTHON_ENGINE_FEED env var
```

```bash
# Switch globally at the shell level
export PYTHON_ENGINE_FEED=polygon
```

The env-driven pattern is what you want for scanners and live processes, because you can flip a single env var at deployment time and rewire the whole project without code changes. Use the explicit form in notebooks where the intent is clearer.

## Which vendor for which job?

| Vendor | Stocks | Options | Cost | Best for |
|---|---|---|---|---|
| **Yahoo (yfinance)** | EOD + intraday (short window) | yes (chain snapshot, no history) | free | learning, this course |
| **Polygon** | tick + EOD | full chains with greeks, history | paid | research + serious backtesting |
| **Alpaca** | EOD + minute (free tier IEX) | yes (newer) | free / paid | execution + reasonable data |
| **Tradier** | EOD + intraday | full chains, paper trading | freemium | retail options |
| **IBKR** | tick + EOD | full chains, real-time | brokerage | live trading |

The course's working examples all use Yahoo so anyone can run them. When a chapter genuinely needs a paid feed (e.g. options-chain history for a backtest in Module 9), it says so up front and shows you how to fall back to a synthetic version.

## Implementing your own

If your shop has an internal data lake, this is the entire integration:

```python
import pandas as pd
from engine.data.feed import Feed, OptionsChain

class InternalFeed:
    name = "internal"

    def __init__(self, conn):
        self.conn = conn

    def bars(self, symbol, start, end, interval="1d") -> pd.DataFrame:
        q = """
            SELECT timestamp, open, high, low, close, volume
              FROM bars
             WHERE symbol = %(sym)s
               AND timestamp BETWEEN %(start)s AND %(end)s
               AND interval = %(interval)s
             ORDER BY timestamp
        """
        df = pd.read_sql(q, self.conn, params=dict(sym=symbol, start=start, end=end, interval=interval))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def option_chain(self, symbol, expiry=None) -> OptionsChain:
        raise NotImplementedError("Chains live in a different table; wire it up if needed.")

assert isinstance(InternalFeed(...), Feed)   # passes — duck-typed at runtime
```

The `@runtime_checkable` decorator on the `Feed` protocol means `isinstance` works without explicit subclassing. The assert is your test.

## Caching

Yahoo will throttle you. The simplest workaround is on-disk caching with [`joblib`](https://joblib.readthedocs.io/) or a thin Parquet layer. Module 5 builds a proper cache; for now the one-liner is:

```python
import functools, pickle, hashlib, pathlib
CACHE = pathlib.Path(".cache"); CACHE.mkdir(exist_ok=True)

def cached(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(pickle.dumps((fn.__name__, args, kwargs))).hexdigest()
        path = CACHE / f"{key}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        out = fn(*args, **kwargs)
        path.write_bytes(pickle.dumps(out))
        return out
    return wrapper

bars = cached(feed.bars)("SPY", "2024-01-01", "2024-06-01")
```

The proper version in Module 5 understands invalidation, partial windows, and Arrow, but the snippet above is enough to unblock you today.

Continue to **[Course conventions](conventions.md)**.
