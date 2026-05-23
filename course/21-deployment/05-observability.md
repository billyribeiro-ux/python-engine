# Observability — Prometheus, Grafana, structured logs

A trading system that nobody can see into is one bug away from disaster. Observability is the discipline of making the system's state visible — through metrics, structured logs, and traces — so you catch problems early and diagnose them quickly.

This chapter is the working setup.

## The three pillars

1. **Metrics** — numerical time-series of system state (P&L, position counts, order rates, latencies). Stored in Prometheus or similar. Visualised in Grafana.
2. **Logs** — structured event records (one order submitted; one fill received; one strategy decision made). Stored in Loki / Elasticsearch / CloudWatch.
3. **Traces** — distributed request flow (a single strategy decision → several broker calls → fills). Stored in Jaeger or Tempo.

For a single-process trading system, metrics + logs are usually enough. Traces matter when you have multiple services.

## Metrics with Prometheus

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# Strategy-level metrics
orders_submitted = Counter("orders_submitted_total", "Total orders submitted",
                            labelnames=["symbol", "side"])
order_submission_latency = Histogram("order_submission_latency_seconds",
                                      "Time to submit an order")
position_value = Gauge("position_value_dollars", "Mark-to-market value",
                        labelnames=["symbol"])
realised_pnl = Gauge("realised_pnl_dollars", "Realised P&L today")
unrealised_pnl = Gauge("unrealised_pnl_dollars", "Unrealised P&L")
kill_switch_state = Gauge("kill_switch_active", "1 if halted, 0 otherwise")


# In strategy code
def on_order_submitted(symbol: str, side: str):
    orders_submitted.labels(symbol=symbol, side=side).inc()


def on_position_update(symbol: str, value: float):
    position_value.labels(symbol=symbol).set(value)


# Start metrics endpoint
start_http_server(8000)            # Prometheus scrapes here
```

Prometheus scrapes the `/metrics` endpoint every 15 seconds. Grafana queries Prometheus and builds dashboards.

## Standard dashboards

For a trading strategy, the must-have panels:

- **P&L** — realised + unrealised, intraday and historical.
- **Drawdown** — current and historical max.
- **Position count and sizes** per symbol.
- **Order rates** — submitted, filled, rejected per minute.
- **Latency** — order submission, fill receipt, broker response.
- **Kill switch state** — should always be "OK"; alert on anything else.
- **Strategy-specific metrics** — signal strength, conformal interval widths, regime indicators.

A typical setup is one Grafana dashboard per strategy + one overarching "Trading Health" dashboard for the whole system.

## Structured logging

JSON-formatted logs are queryable, parseable, machine-readable:

```python
import structlog
from datetime import datetime, timezone


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger()


def on_order_submitted(req: OrderRequest, broker_response):
    log.info("order_submitted",
              symbol=req.symbol, qty=req.qty, side=req.side,
              order_type=req.order_type, limit=req.limit,
              broker_order_id=broker_response.id,
              idempotency_key=req.idempotency_key)
```

Each log entry is a JSON object with key-value pairs. Searchable by any field; aggregatable; perfect for post-mortem analysis.

Output to stdout; have your log shipper (Vector, Fluentd) forward to Loki or whatever centralised store you use.

## Alerts

Set up alerting on the **disaster signals** from chapter 4:

```yaml
# Prometheus alerting config
groups:
  - name: trading
    rules:
      - alert: HighDailyDrawdown
        expr: realised_pnl_dollars + unrealised_pnl_dollars < -10000
        for: 1m
        labels: { severity: critical }
        annotations:
          summary: "Daily P&L below -$10k"

      - alert: OrderRejectRate
        expr: rate(orders_rejected_total[5m]) > 0.5
        for: 2m
        labels: { severity: warning }
        annotations:
          summary: "Order rejection rate > 0.5/sec"

      - alert: KillSwitchActive
        expr: kill_switch_active == 1
        for: 0m
        labels: { severity: critical }
        annotations:
          summary: "Kill switch is ACTIVE"

      - alert: FeedStale
        expr: time() - last_feed_update_timestamp > 60
        for: 30s
        labels: { severity: critical }
        annotations:
          summary: "Market data feed stale > 60s"
```

Route critical alerts to PagerDuty / Opsgenie; warnings to Slack. For a small team, even just Slack is fine.

## Traces (for multi-service systems)

If your trading system has multiple services (data ingestion, model server, execution engine), distributed tracing helps:

```python
from opentelemetry import trace


tracer = trace.get_tracer(__name__)


def make_decision(symbol: str):
    with tracer.start_as_current_span("make_decision", attributes={"symbol": symbol}):
        with tracer.start_as_current_span("fetch_features"):
            features = fetch_features(symbol)
        with tracer.start_as_current_span("model_predict"):
            prediction = model.predict(features)
        with tracer.start_as_current_span("submit_order"):
            order = submit_order(symbol, prediction)
    return order
```

Jaeger or Tempo shows you the full call chain with timing — useful when "the system is slow today" and you need to find where.

For a single-process Python strategy, traces are overkill. Logs + metrics suffice.

## Daily reports

In addition to live dashboards, generate a daily PDF / HTML email report:

- Yesterday's P&L breakdown (per strategy, per symbol).
- Yesterday's trades (count, notional, fills vs intent).
- Drawdown updates.
- Anomalies (orders that got unusual slippage, vol regime indicators that fired).
- A short narrative ("Strategy A made $X driven by Y").

Even with great real-time dashboards, the daily summary keeps you (and your stakeholders) informed without having to log in.

## Pitfalls

!!! warning "Metric cardinality explosion"
    A counter with `labelnames=["symbol", "side", "venue", "size_bucket"]` quickly becomes millions of unique time series. Prometheus chokes. Keep cardinality reasonable.

!!! warning "Logs that don't shop"
    Per-tick logs are useful for debugging but flood production storage. Use sampling (`if random.random() < 0.001: log.info(...)`).

!!! warning "Alert fatigue"
    Too many alerts → people ignore them. Tune thresholds so alerts mean something. Each alert should require a human response.

!!! warning "Time series with wrong timestamps"
    Logs / metrics with the local time of the producing machine create chaos when machines are in different timezones. Always use UTC.

## Bottom line

For trading observability:

- **Prometheus + Grafana** for metrics dashboards.
- **Structured JSON logs** routed to centralised storage.
- **Alerts on disaster signals** routed to PagerDuty / Slack.
- **Daily summary reports** even with good dashboards.
- **Traces** only if you have multiple services.

Continue to **[Postmortems and not over-tuning](06-postmortems.md)**.
