# Module 21 — Deployment

A strategy isn't real until it's running in production with real money. This module is the final-mile checklist: paper trading, broker integration, order routing, idempotency, kill switches, observability, postmortems. The discipline here is what separates "we ran a backtest" from "we manage a fund."

Pages:

1. **[Paper trading first](01-paper-trading.md)** — why, how, and the discipline.
2. **[Brokerage integration](02-brokerages.md)** — Alpaca, Tradier, IBKR.
3. **[Order routing and idempotency](03-routing.md)** — never double-trade.
4. **[Kill switches](04-kill-switches.md)** — hard limits that no algorithm can override.
5. **[Observability](05-observability.md)** — Prometheus / Grafana for trading systems.
6. **[Postmortems and not over-tuning](06-postmortems.md)** — what to do when something breaks.
