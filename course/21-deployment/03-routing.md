# Order routing and idempotency

In a live trading system, network failures, broker timeouts, and process restarts can all cause an order to be submitted and lost-track-of. Without strict idempotency, you can double-trade. This chapter is the discipline that makes order routing safe.

## The fundamental problem

Your code calls `broker.submit_market("SPY", 100, "BUY")`. The TCP connection times out. Did the order go through?

If you retry naively, you may now have 200 shares instead of 100. Worse, if the original order *did* go through and you trade against it, you've sold 100 you didn't actually own.

The fix: **every order has a unique client-side identifier**. The broker recognises duplicates and rejects them.

## Idempotency keys in practice

```python
import uuid
from dataclasses import dataclass


@dataclass
class OrderRequest:
    symbol: str
    qty: int
    side: str
    order_type: str = "MARKET"
    limit: float | None = None
    idempotency_key: str = ""

    def __post_init__(self):
        if not self.idempotency_key:
            self.idempotency_key = str(uuid.uuid4())


def submit_with_idempotency(broker, request: OrderRequest, max_retries: int = 3):
    """Submit; retry on transient errors; the broker dedupes via idempotency_key."""
    for attempt in range(max_retries):
        try:
            return broker.submit(request)
        except (TimeoutError, ConnectionError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

The same `idempotency_key` is sent on every retry. The broker, having seen it, returns the original order ID rather than creating a new one.

Alpaca uses `client_order_id`; IBKR uses `clientOrderId`. Both deduplicate.

## Persisting orders for crash recovery

If your strategy crashes mid-submission, on restart you need to know what orders are outstanding. Persist intentions before sending:

```python
import sqlite3


class OrderLog:
    def __init__(self, path: str = "orders.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                idempotency_key TEXT PRIMARY KEY,
                submitted_at TEXT,
                symbol TEXT,
                qty INTEGER,
                side TEXT,
                broker_order_id TEXT,
                status TEXT
            )
        """)
        self.conn.commit()

    def record_intent(self, req: OrderRequest):
        self.conn.execute("INSERT OR IGNORE INTO orders (idempotency_key, submitted_at, "
                          "symbol, qty, side, status) VALUES (?, datetime('now'), ?, ?, ?, ?)",
                          (req.idempotency_key, req.symbol, req.qty, req.side, "PENDING"))
        self.conn.commit()

    def update_broker_id(self, idempotency_key: str, broker_id: str):
        self.conn.execute("UPDATE orders SET broker_order_id = ?, status = 'SUBMITTED' "
                          "WHERE idempotency_key = ?", (broker_id, idempotency_key))
        self.conn.commit()

    def pending(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT idempotency_key FROM orders WHERE status = 'PENDING'").fetchall()]
```

On startup, query `pending()`. For each, ask the broker whether it has an order with that idempotency key. If yes, update local state. If no, decide whether to retry.

## Position reconciliation on startup

Even with order persistence, your local position state may drift from the broker's. On every startup:

```python
def reconcile_positions(broker, local_positions: dict) -> dict:
    """Compare local and broker positions; broker is source of truth."""
    broker_pos = broker.positions()
    discrepancies = {}
    for sym in set(local_positions) | set(broker_pos):
        local = local_positions.get(sym, 0)
        actual = broker_pos.get(sym, 0)
        if local != actual:
            discrepancies[sym] = {"local": local, "broker": actual}
    return discrepancies


# Always
discrepancies = reconcile_positions(broker, load_local_positions())
if discrepancies:
    alert(f"Position discrepancy on startup: {discrepancies}")
    save_local_positions(broker.positions())     # broker wins
```

The broker is always the source of truth. Your local cache is a convenience; reconcile aggressively.

## Order lifecycle state machine

An order goes through states: PENDING → SUBMITTED → ACK → WORKING → (PARTIAL | FILLED | REJECTED | CANCELED). Your code should explicitly track each transition:

```python
from enum import Enum


class OrderState(Enum):
    PENDING = "pending"           # we've decided to send
    SUBMITTED = "submitted"       # we sent; awaiting ack
    ACK = "ack"                   # broker received
    WORKING = "working"           # in the market
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


def update_order_state(idempotency_key: str, new_state: OrderState):
    # Persist; emit metrics; etc.
    ...
```

For each broker event, validate the state transition is allowed. Going from FILLED → WORKING is illegal — something has broken. Alert and pause.

## The "stuck order" pattern

A common failure: an order sits in WORKING for hours, possibly because the broker's order-events stream missed a fill event. Detection:

```python
def stale_orders_check(order_log: OrderLog, broker, threshold_minutes: int = 5):
    """Reconcile orders that have been WORKING for > threshold."""
    stale = order_log.working_for_longer_than(threshold_minutes)
    for key in stale:
        broker_id = order_log.get_broker_id(key)
        broker_state = broker.get_order_state(broker_id)
        if broker_state in ("FILLED", "CANCELED"):
            # Our state was stale; fix it
            order_log.update_state(key, OrderState[broker_state])
        else:
            # Genuinely working at the broker too; nothing to do
            pass
```

Run this every few minutes as a watchdog. Stale orders are usually missed events; the cure is a re-query.

## Pre-trade checks

Before submitting any order, enforce hard limits:

```python
def pre_trade_check(req: OrderRequest, current_positions: dict, kill_switches: dict) -> bool:
    """Hard pre-trade validation. Return False to block the order."""
    if kill_switches["all"]:
        log.warning(f"All trading killed; blocking order {req.symbol}")
        return False
    new_position = current_positions.get(req.symbol, 0) + (req.qty if req.side == "BUY" else -req.qty)
    if abs(new_position) > MAX_POSITION_PER_SYMBOL:
        log.warning(f"Order would exceed max position for {req.symbol}")
        return False
    gross_after = sum(abs(p) for p in current_positions.values()) + req.qty
    if gross_after > MAX_GROSS_EXPOSURE:
        log.warning(f"Order would exceed max gross exposure")
        return False
    return True
```

Pre-trade checks are non-overridable. Even if the strategy is correct, the pre-trade check is the last defence against software bugs.

## Bottom line

For order routing:

- **Idempotency keys on every order**.
- **Persist intentions** before sending.
- **Reconcile positions** from broker state on startup.
- **Track order lifecycle** as an explicit state machine.
- **Pre-trade checks** that no strategy can override.

These are not optional. Every fund that has had a "fat finger" or "double-fill" incident has implemented them after.

Continue to **[Kill switches](04-kill-switches.md)**.
