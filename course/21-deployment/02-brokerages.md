# Brokerage integration

For US retail and small-fund systematic trading, three brokers cover most use cases: **Alpaca** (best API), **Tradier** (best options), **Interactive Brokers** (most markets and instruments). This chapter is the working overview, with the code patterns that ship.

## Decision matrix

| Vendor | Stocks | Options | Margin | API | Best for |
|---|---|---|---|---|---|
| **Alpaca** | yes | newer | yes | REST + WebSocket | systematic equity strategies, clean API |
| **Tradier** | yes | excellent | yes | REST | options strategies, lowest options fees |
| **Interactive Brokers** | yes | yes | yes | TWS + REST (deprecated), IB Gateway + ib_insync | multi-asset, multi-region, institutional features |

Pick by what you trade. For pure US equity ETFs and stocks, Alpaca is the gentlest. For options-heavy work, Tradier. For futures, FX, non-US — IBKR.

## Alpaca

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream


# Trading client
trading = TradingClient(API_KEY, SECRET_KEY, paper=True)


def submit_market(symbol: str, qty: int, side: str) -> str:
    order = MarketOrderRequest(
        symbol=symbol, qty=qty,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return trading.submit_order(order_data=order).id


def positions() -> dict:
    return {p.symbol: int(p.qty) for p in trading.get_all_positions()}


# Live data stream
async def stream_quotes(symbols: list[str]):
    stream = StockDataStream(API_KEY, SECRET_KEY)
    async def on_quote(quote):
        await handle_quote(quote)
    stream.subscribe_quotes(on_quote, *symbols)
    await stream.run()
```

The Alpaca Python SDK is well-maintained, type-hinted, and consistent. The paper/live switch is one boolean.

## Tradier

Tradier has a REST API; no official Python SDK but plenty of third-party wrappers. For options strategies:

```python
import requests

class TradierClient:
    def __init__(self, token: str, account_id: str, paper: bool = True):
        self.token = token
        self.account_id = account_id
        self.base = "https://sandbox.tradier.com/v1" if paper else "https://api.tradier.com/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}",
                                      "Accept": "application/json"})

    def options_chain(self, symbol: str, expiration: str) -> list[dict]:
        r = self.session.get(f"{self.base}/markets/options/chains",
                              params={"symbol": symbol, "expiration": expiration})
        return r.json()["options"]["option"]

    def submit_options_order(self, symbol: str, option_symbol: str,
                              side: str, quantity: int, order_type: str = "market") -> dict:
        data = {
            "class": "option",
            "symbol": symbol,
            "option_symbol": option_symbol,
            "side": side,                                     # buy_to_open, sell_to_open, etc.
            "quantity": quantity,
            "type": order_type,
            "duration": "day",
        }
        r = self.session.post(f"{self.base}/accounts/{self.account_id}/orders", data=data)
        return r.json()
```

Tradier's options-chain API is one of the cleanest. Their fee structure ($0.35/contract) is competitive for high-volume traders.

## Interactive Brokers via `ib_insync`

IBKR's classic API is challenging; `ib_insync` wraps it cleanly:

```python
from ib_insync import IB, Stock, MarketOrder, util


ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)             # 7497 = TWS paper; 7496 = IBG paper


def submit_market(symbol: str, qty: int, side: str):
    contract = Stock(symbol, "SMART", "USD")
    order = MarketOrder(side, qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    return trade


def positions() -> dict:
    return {p.contract.symbol: p.position for p in ib.positions()}


# Live data
def stream_ticks(symbol: str):
    contract = Stock(symbol, "SMART", "USD")
    ib.reqMktData(contract, "", False, False)
    while True:
        ib.waitOnUpdate()
        ticker = ib.ticker(contract)
        process(ticker)
```

`ib_insync` is sync-style by default but supports async via `asyncio`. The `ib.sleep` is necessary because the IB API requires you to yield to its event loop periodically.

IBKR's TWS / Gateway must be running on the same machine (or accessible at a configured IP). The free desktop app version uses IBKR's data feeds.

## Universal Broker Interface

For a strategy framework that supports any of the three, abstract behind a Protocol:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Broker(Protocol):
    def submit_market(self, symbol: str, qty: int, side: str) -> str: ...
    def submit_limit(self, symbol: str, qty: int, side: str, limit: float) -> str: ...
    def cancel(self, order_id: str) -> None: ...
    def positions(self) -> dict[str, int]: ...
    def cash(self) -> float: ...


class AlpacaBroker:
    def __init__(self, key, secret, paper=True):
        self.client = TradingClient(key, secret, paper=paper)
        self.name = "alpaca"
    def submit_market(self, symbol, qty, side):
        # ... as above
        ...
    # etc


class IBKRBroker:
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
        self.name = "ibkr"
    # ...
```

This mirrors the `Feed` protocol pattern from Module 0. Your strategy code only knows about `Broker`; the concrete implementation is a deployment decision.

## Authentication and secrets

**Never commit API keys**. Use environment variables or a secrets manager:

```python
import os

key = os.environ["ALPACA_API_KEY"]
secret = os.environ["ALPACA_SECRET_KEY"]
if not key or not secret:
    raise RuntimeError("Missing Alpaca credentials")
```

For production, use HashiCorp Vault, AWS Secrets Manager, or `secret-tool` (Linux). The course's `engine.data` adapters all read from env vars by convention.

## Rate limits and retries

Every broker has rate limits. Alpaca: 200 requests/minute. Tradier: variable; check docs. IBKR: ~50 requests/second.

Always wrap broker calls with retry-and-backoff (Module 5 chapter 2):

```python
import time, random

def with_retry(fn, attempts=5, base=0.5):
    def wrapped(*args, **kwargs):
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except RateLimitError:
                time.sleep(base * (2 ** i) * (1 + random.random()))
        raise
    return wrapped


submit_with_retry = with_retry(submit_market)
```

## Idempotency

If your strategy submits an order and the network drops before the response arrives, did the order go through? Without **idempotency keys**, you might re-submit and get duplicate fills.

The pattern: every order gets a unique idempotency key. The broker recognises duplicates and rejects them.

```python
import uuid

def submit_idempotent(broker, symbol, qty, side, idempotency_key=None):
    if idempotency_key is None:
        idempotency_key = str(uuid.uuid4())
    return broker.submit_market(symbol, qty, side, client_order_id=idempotency_key)
```

Alpaca's `client_order_id` and IBKR's `clientOrderId` are idempotency keys. Use them on every order.

## Pitfalls

!!! warning "Broker downtime"
    Every broker has occasional outages. Your strategy should handle "broker unavailable" gracefully — pause trading, alert, retry. Don't crash.

!!! warning "Symbol differences"
    The same instrument may have different symbols across brokers. SPY at IBKR is `Stock("SPY", "SMART", "USD")`; at Alpaca it's `"SPY"`. Maintain a symbol map.

!!! warning "Time zone confusion"
    Order timestamps from different brokers may be in different timezones. Convert everything to UTC at the boundary.

!!! warning "Order status semantics"
    "Filled" might mean fully filled or partially filled. Read the broker's specific semantics carefully.

## Bottom line

For broker integration:

- **Choose by what you trade**: Alpaca for equity, Tradier for options, IBKR for everything else.
- **Abstract behind a Protocol** so your strategy code is broker-agnostic.
- **Always use idempotency keys** on orders.
- **Wrap broker calls with retry-and-backoff**.
- **Reconcile positions** from broker state on every startup.

Continue to **[Order routing and idempotency](03-routing.md)**.
