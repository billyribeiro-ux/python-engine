# Dataclasses, `attrs`, `pydantic`

You will spend the rest of your Python life writing classes that hold data. Three libraries dominate. They look identical at first glance and behave very differently in production. Choose deliberately.

## TL;DR

| Library | Standard library? | Validation? | Speed | Use when |
|---|---|---|---|---|
| `dataclasses` | yes | no | fastest (it's just code-gen for `__init__`/`__repr__`) | internal value objects, hot paths, slots/frozen |
| `attrs` | no | yes (lightweight) | very fast | richer behaviour (validators, converters), no Pydantic dependency |
| `pydantic v2` | no | yes (strict, type-driven) | fast | data crossing trust boundaries (API requests, configs, vendor JSON) |
| `msgspec` | no | yes | **fastest** for JSON/MessagePack | high-throughput parsing of structured wire data |

For a trading codebase: `dataclasses` for in-memory records, `pydantic` for configs and external data, `msgspec` for hot deserialisation.

## `dataclasses` — the standard library answer

```python
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True, kw_only=True)
class Bar:
    timestamp: pd.Timestamp
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    extras: dict[str, float] = field(default_factory=dict)
```

What the decorator gives you:

- `__init__`, `__repr__`, `__eq__` generated from the annotated fields.
- `slots=True` → no `__dict__`, smaller memory, faster attribute access.
- `frozen=True` → instances are immutable. Assignment raises `FrozenInstanceError`. Also gives you a `__hash__` for free.
- `kw_only=True` → all fields must be passed by keyword.

Performance: a dataclass `__init__` is just regular Python code that the decorator wrote for you. It is exactly as fast as the hand-rolled version.

### Custom logic? Just write it

A dataclass is still a class. Add methods:

```python
@dataclass(slots=True, frozen=True)
class Trade:
    price: float
    qty: int
    side: Literal["BUY", "SELL"]

    @property
    def notional(self) -> float:
        return self.price * self.qty * (1 if self.side == "BUY" else -1)
```

### `__post_init__` for derived fields

`dataclass` lets you hook in after `__init__` runs:

```python
@dataclass(slots=True)
class CashBalance:
    deposits: float
    withdrawals: float
    net: float = 0.0           # filled in post_init

    def __post_init__(self):
        self.net = self.deposits - self.withdrawals
```

For frozen classes, set fields via `object.__setattr__(self, "name", value)`.

### When `dataclasses` is the right answer

- The data never leaves your process trusted.
- You don't need validation beyond "the types are what you typed".
- You care about speed and memory.

That's the bulk of your code.

## `attrs` — the older, sharper sibling

`attrs` predates `dataclasses` and is still excellent. The main reasons to reach for it:

- **Validators and converters** are first-class.
- **`@define` and `@frozen`** with `slots=True` by default.
- **Better APIs** for inheritance, multiple base classes, and class manipulation.

```python
from attrs import define, field, validators

@define(frozen=True, slots=True)
class Position:
    symbol: str = field(validator=validators.matches_re(r"^[A-Z.]{1,8}$"))
    qty:    int = field(converter=int, validator=validators.instance_of(int))
    avg_px: float = field(converter=float, validator=validators.gt(0))
```

If you find yourself writing `__post_init__` in a dataclass every time to validate, switch to `attrs`. The code becomes declarative.

## `pydantic` v2 — for data crossing trust boundaries

`pydantic` is the library you reach for when the data **isn't yours**. API responses, broker order acknowledgements, config files, request bodies from a web framework. The killer feature is **strict type-driven validation, parsed from arbitrary JSON-like input**.

```python
from pydantic import BaseModel, Field, ConfigDict

class OrderAck(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    filled_qty: int = Field(ge=0)
    avg_fill_price: float | None = Field(default=None, gt=0)


# Vendor sends you this JSON:
payload = {"order_id": "X1", "symbol": "SPY", "side": "BUY", "filled_qty": 100,
           "avg_fill_price": 475.30, "extra_field": "ignored"}

ack = OrderAck.model_validate(payload)
```

What just happened:

- All types were validated.
- `extra_field` was ignored (the default; configurable).
- If `filled_qty` were negative, you'd get a structured `ValidationError` listing exactly which field failed and why.

`pydantic` v2 is written in Rust under the hood (`pydantic-core`) and is **roughly an order of magnitude faster than v1**. Performance is no longer a reason to avoid it.

### Configuration

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PE_", env_file=".env")

    feed: Literal["yfinance", "polygon", "alpaca", "tradier", "ibkr"] = "yfinance"
    cache_dir: str = ".cache"
    log_level: str = "INFO"
    polygon_api_key: str | None = None

config = AppConfig()
```

Environment variables, `.env` files, secrets — all driven by the type. Replace your `os.getenv("X", "default")` blocks with this and never look back.

### Tagged unions

For "this can be one of several shapes" payloads:

```python
from typing import Literal
from pydantic import BaseModel, Field
from typing import Annotated

class Limit(BaseModel):
    kind: Literal["LIMIT"]
    price: float

class Market(BaseModel):
    kind: Literal["MARKET"]

class Stop(BaseModel):
    kind: Literal["STOP"]
    trigger: float

Order = Annotated[Limit | Market | Stop, Field(discriminator="kind")]

class Envelope(BaseModel):
    order: Order

env = Envelope.model_validate({"order": {"kind": "STOP", "trigger": 100.0}})
isinstance(env.order, Stop)   # True
```

pydantic looks at the `kind` literal, picks the right model, validates the rest. This is the cleanest way to type a heterogeneous event stream.

## `msgspec` — when bytes are the bottleneck

If you parse millions of messages per second (websocket feeds, kafka topics), `msgspec` is the right tool. It has the same declarative-class API as pydantic, but compiles a specialised parser per class and skips Python-level overhead. Benchmarks routinely show 5–10× over pydantic for JSON, 20×+ for MessagePack.

```python
import msgspec

class Tick(msgspec.Struct, frozen=True, gc=False):
    ts: int
    px: float
    sz: int
    sym: str

decoder = msgspec.json.Decoder(Tick)
tick = decoder.decode(b'{"ts": 1700000000, "px": 475.3, "sz": 100, "sym": "SPY"}')
```

`gc=False` opts out of cyclic GC tracking for these instances — a real win on tight inner loops.

When you're processing > 10 k msgs/s in Python, you want msgspec.

## Choosing without thinking

The simplest decision rule:

1. **Trusted, internal data** → `@dataclass(slots=True, frozen=True)`.
2. **Trusted but needs validators / converters** → `attrs`.
3. **Untrusted external data** → `pydantic`.
4. **High-throughput parsing** → `msgspec`.

Mix them freely. The boundary between "this came from outside" and "this is internal" is where you parse with pydantic/msgspec into a dataclass for everything downstream:

```python
class WireOrder(BaseModel):       # untrusted shape
    ...

@dataclass(slots=True, frozen=True)
class Order:                       # internal shape
    ...

wire = WireOrder.model_validate(payload)
order = Order(symbol=wire.symbol, qty=wire.qty)   # ...plus the remaining fields
```

That's the pattern. The validation boundary is explicit. Everything past it is fast and untyped-string-free.

You now know enough Python to write data systems that don't fall over, are clearly typed, and use the right concurrency for the workload. One more foundations chapter rounds out the module — the language features that landed in the current Python releases.

Continue to **[Modern Python: what changed in 3.12–3.14](07-modern-python.md)**.
