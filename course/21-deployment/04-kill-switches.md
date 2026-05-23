# Kill switches

A kill switch is a mechanism that halts trading regardless of what the strategy is doing. They're the difference between a bad day and a Knight-Capital-style catastrophe. Every production trading system has them; the only question is at how many layers.

## The layers

In order of how quickly they trigger:

1. **Broker-side** — most brokers offer max-loss, max-trade-count, max-notional caps that the broker enforces. Use them.
2. **Process-level** — your strategy code checks limits before every order; refuses to send if breached.
3. **Operating-system-level** — a watchdog process kills the strategy if it crosses a hard threshold.
4. **Manual** — a "panic button" web page that you (or any team member) can hit to stop everything.

Implement all four. They have different failure modes.

## Broker-side limits

Almost every broker supports:

- **Max daily loss** — broker stops accepting orders once you've lost X.
- **Max position size** — refuses to grow a position past Y.
- **Order-rate limit** — refuses more than Z orders per minute.

These are the most reliable because they don't depend on your code working. Set them at sensible levels for your strategy.

For Alpaca: configure in the web UI under Account → Risk Controls. For IBKR: TWS → File → Global Configuration → Presets. Tradier offers similar.

## Process-level kill checks

```python
from dataclasses import dataclass


@dataclass
class KillSwitches:
    """Process-level kill switches. Check before every order."""
    daily_max_loss_dollars: float
    max_gross_exposure_dollars: float
    max_orders_per_minute: int
    max_position_per_symbol_shares: int
    halted: bool = False

    def check_can_trade(self, current_pnl: float, current_exposure: float,
                         recent_order_count: int, requested_position: int,
                         symbol: str) -> tuple[bool, str]:
        if self.halted:
            return False, "halted manually"
        if current_pnl < -self.daily_max_loss_dollars:
            return False, f"daily max loss breached: {current_pnl}"
        if current_exposure > self.max_gross_exposure_dollars:
            return False, f"gross exposure breached: {current_exposure}"
        if recent_order_count > self.max_orders_per_minute:
            return False, f"order rate breached: {recent_order_count}/min"
        if abs(requested_position) > self.max_position_per_symbol_shares:
            return False, f"max position breached for {symbol}: {requested_position}"
        return True, "ok"
```

Wire into your order submission:

```python
def submit_order_with_checks(broker, kill_switches, req: OrderRequest,
                               current_pnl, current_exposure, recent_orders):
    ok, reason = kill_switches.check_can_trade(
        current_pnl, current_exposure, recent_orders,
        current_position(req.symbol) + (req.qty if req.side == "BUY" else -req.qty),
        req.symbol,
    )
    if not ok:
        alert(f"Kill switch fired: {reason}")
        return None
    return broker.submit(req)
```

The kill switch is checked **before every order**. If it fires, the order is dropped and an alert is raised.

## Watchdog process

A separate process that monitors the trading process and kills it on disaster signals:

```python
# watchdog.py — runs in a separate process
import psutil
import time
import requests


def fetch_pnl_from_broker():
    return get_alpaca_account_value() - INITIAL_VALUE


def watchdog_loop():
    trading_pid = find_trading_process()
    initial_value = fetch_pnl_from_broker()
    while True:
        time.sleep(30)
        try:
            current_value = fetch_pnl_from_broker()
            drawdown = (current_value - initial_value) / initial_value
            if drawdown < -0.10:                # 10% drawdown — hard stop
                log.error(f"Drawdown breach: {drawdown:.2%}; killing trading process")
                psutil.Process(trading_pid).terminate()
                send_pagerduty_alert(f"Trading killed: {drawdown:.2%} drawdown")
                # Submit a market-close order via broker REST API as well
                close_all_positions()
                break
        except Exception as e:
            send_pagerduty_alert(f"Watchdog error: {e}")


if __name__ == "__main__":
    watchdog_loop()
```

The watchdog runs as a separate `systemd` service or container, independent of the trading process. If the trading process is buggy, the watchdog still works.

## Manual panic button

A simple web page on a fixed IP/port that anyone with credentials can hit:

```python
from flask import Flask, request


app = Flask(__name__)


@app.route("/halt", methods=["POST"])
def halt():
    auth_token = request.headers.get("Authorization")
    if auth_token != EXPECTED_TOKEN:
        return "unauthorized", 401
    set_kill_switch("halted", True)
    close_all_positions_immediately()
    send_pagerduty_alert(f"MANUAL HALT triggered by {request.remote_addr}")
    return "halted", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
```

In a real incident, the panic button is sometimes the only thing that works fast enough. Set it up; test it monthly; document the URL where your team can find it at 3am.

## When to fire the kill switch

The list of disaster signals:

- **Drawdown exceeds X%** in a single day. Hard cap; uses watchdog.
- **Realised vol exceeds N× the design vol** in a window. Strategy is operating outside its regime.
- **Order rejection rate exceeds threshold**. Broker is rejecting most orders → assume something is wrong with our state.
- **Position discrepancy** between local state and broker exceeds N shares. Reconciliation has failed; pause.
- **Feed stops updating** for > N seconds during market hours. Either feed is down or we're disconnected; don't trade on stale data.
- **External alert** (Black Monday-style move on the index). Pause and re-evaluate.

Each gets a kill switch. Don't try to make the strategy "robust to all of these" — it can't be. Pause, investigate, restart.

## Drills

Once a quarter, **deliberately trigger a kill switch in production**. Verify:

- The kill switch actually stops trading.
- All open orders are cancelled.
- Positions are closed (if your kill behaviour is "go flat").
- Alerts are received by the right people.

The first time you fire a kill switch in real anger should not be the first time you've fired it.

## Pitfalls

!!! warning "Kill switches that depend on your code working"
    If your code is buggy, your code-level kill switch is buggy too. Always have a broker-side or OS-level kill switch as a backup.

!!! warning "Kill switch but no auto-close"
    Halting orders isn't enough — you still have open positions. The kill switch should optionally close-to-flat (especially for intraday strategies).

!!! warning "Kill thresholds too tight"
    A 1% daily drawdown threshold will fire constantly. Pick thresholds that distinguish "bad day" from "catastrophe."

!!! warning "Not testing the kill switch"
    A kill switch that's never been tested is not a kill switch.

## Bottom line

For kill switches:

- **Multiple layers**: broker, process, watchdog, manual.
- **Pre-trade checks** at every order.
- **Watchdog process** independent of trading process.
- **Manual panic button** at known URL.
- **Quarterly drills** to verify everything works.

Continue to **[Observability](05-observability.md)**.
