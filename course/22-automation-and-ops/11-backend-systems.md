# Backend systems with FastAPI

You need a REST endpoint. A webhook receiver. A small internal admin UI. A model-serving API. FastAPI is the Python web framework for all of these — async-native, type-driven, fast, ergonomic. This chapter is the production-grade FastAPI you'll actually deploy.

## The minimum viable service

```python
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Python Engine Ops API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=app.version)
```

Run: `uvicorn main:app --reload`. OpenAPI docs at `/docs` (Swagger UI) and `/redoc` (ReDoc). Type-hint-driven; everything is auto-documented.

## Endpoints with typed request and response

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import date


class BackfillRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=16, pattern=r"^[A-Z0-9.]+$")
    start_date: date
    end_date: date | None = None
    interval: str = Field(default="1d")


class BackfillResponse(BaseModel):
    symbol: str
    bars_loaded: int
    duration_seconds: float


@app.post("/backfill", response_model=BackfillResponse, status_code=202)
async def backfill(req: BackfillRequest):
    if req.end_date and req.end_date < req.start_date:
        raise HTTPException(400, "end_date before start_date")
    t0 = time.perf_counter()
    n = await run_backfill(req.symbol, req.start_date, req.end_date, req.interval)
    return BackfillResponse(symbol=req.symbol, bars_loaded=n,
                             duration_seconds=time.perf_counter() - t0)
```

Pydantic validates the request body and rejects bad inputs before they reach your handler. The OpenAPI docs show the exact schema.

## Async first

FastAPI shines with async I/O:

```python
import httpx


@app.get("/quote/{symbol}")
async def quote(symbol: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://vendor.example/quote/{symbol}", timeout=5.0)
        r.raise_for_status()
    return r.json()
```

For a handler that mostly does I/O (HTTP calls, DB queries), async lets one process handle hundreds of concurrent requests. For CPU-heavy work (ML inference), use `def` instead and FastAPI will run it on a thread pool.

## Dependency injection

For shared resources (database connections, configs), use `Depends`:

```python
from fastapi import Depends
from sqlalchemy.orm import Session


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/orders/{order_id}")
def get_order(order_id: int, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(404, "not found")
    return order
```

Each request gets its own database session; cleanup is automatic. Same pattern for auth, rate-limiting, request-tracing.

## Authentication

For an internal API, a shared bearer token is enough:

```python
from fastapi import Header, HTTPException
import os


def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {os.environ['API_TOKEN']}":
        raise HTTPException(401, "unauthorised")


@app.post("/trade", dependencies=[Depends(verify_token)])
def submit_trade(req: TradeRequest):
    ...
```

For real production with multiple users, use OAuth2 / OIDC — FastAPI has built-in helpers (`fastapi.security.OAuth2PasswordBearer`) and integrates with Auth0, Okta, Keycloak.

## Background tasks

For "respond fast, do the heavy work after":

```python
from fastapi import BackgroundTasks


@app.post("/start-job")
def start_job(req: JobRequest, background_tasks: BackgroundTasks):
    job_id = create_job_record(req)
    background_tasks.add_task(do_heavy_work, job_id)
    return {"job_id": job_id, "status": "queued"}
```

The handler returns immediately; the heavy work runs after the response is sent. For real distributed task processing, use `Celery` or `RQ`.

## Webhook receivers

Many vendors push events to your webhook URL. The pattern:

```python
import hmac
import hashlib


@app.post("/webhooks/broker")
async def broker_webhook(request: Request, x_signature: str = Header(...)):
    body = await request.body()
    expected_sig = hmac.new(SHARED_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(x_signature, expected_sig):
        raise HTTPException(401, "invalid signature")

    event = await request.json()
    # Make the handler IDEMPOTENT — webhooks may be retried by the sender
    if await event_already_processed(event["id"]):
        return {"status": "ok", "duplicate": True}
    await process_event(event)
    await mark_event_processed(event["id"])
    return {"status": "ok"}
```

Two non-negotiables:

1. **Verify the signature** so spoofed callers can't trigger your webhook.
2. **Idempotent processing** — most webhook providers retry on a non-2xx; without idempotency, duplicate events corrupt state.

## Streaming responses

For server-sent events or long-running results:

```python
from fastapi.responses import StreamingResponse
import asyncio
import json


@app.get("/stream/scanner-results")
async def stream_results():
    async def gen():
        for batch in iterate_scanner_batches():
            yield f"data: {json.dumps(batch)}\n\n"
            await asyncio.sleep(0)
    return StreamingResponse(gen(), media_type="text/event-stream")
```

Or `WebSocket` for true bi-directional:

```python
from fastapi import WebSocket


@app.websocket("/ws/quotes")
async def quote_stream(ws: WebSocket):
    await ws.accept()
    try:
        async for quote in stream_quotes():
            await ws.send_json(quote)
    except WebSocketDisconnect:
        log.info("client disconnected")
```

## Error handling

Custom exceptions with structured error responses:

```python
from fastapi.responses import JSONResponse
from fastapi.requests import Request


class DomainError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code; self.message = message; self.status = status


@app.exception_handler(DomainError)
async def handle_domain(request: Request, exc: DomainError):
    return JSONResponse(
        status_code=exc.status,
        content={"error_code": exc.code, "message": exc.message},
    )


# In a handler:
@app.post("/cancel-order")
def cancel(order_id: int):
    order = lookup(order_id)
    if order.status == "FILLED":
        raise DomainError(code="ORDER_NOT_CANCELABLE", message="already filled")
    ...
```

Clients get a consistent `{error_code, message}` shape they can parse.

## Production deployment

For a real service:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[core,ml]"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

For higher load, `gunicorn` with `uvicorn` workers:

```
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

For autoscaling: Kubernetes, ECS, Cloud Run — all native FastAPI deployment targets.

## Observability

FastAPI integrates cleanly with Prometheus and OpenTelemetry:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)        # /metrics endpoint
```

Now you have request rate, duration, error rate, broken down by route. Grafana dashboard, alerts (Module 21 chapter 5).

For tracing:

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)
```

Sends spans to whichever OTLP backend (Jaeger, Tempo) you've configured.

## Pitfalls

!!! warning "Sync DB call in an async handler"
    `db.query(...)` (sync SQLAlchemy) inside `async def` blocks the event loop. Either use `async def` with `asyncpg`/`async SQLAlchemy`, or use `def` so FastAPI runs it on the thread pool.

!!! warning "Returning huge JSON"
    A 100 MB JSON response materialises in memory and crashes under load. Stream large responses or paginate.

!!! warning "No request body size limit"
    Without `--limit-request-line` / `--limit-request-field-size`, a malicious client can OOM your service. Use Uvicorn/Gunicorn flags.

!!! warning "Secrets in code"
    Read env vars at startup; never commit `.env` files. Use a real secrets manager for production.

## Bottom line

For production backend services:

- **FastAPI + Pydantic** for typed, documented APIs.
- **Async by default** for I/O-bound; `def` for CPU work.
- **Dependency injection** for shared resources.
- **Idempotent + signature-verified** webhook handlers.
- **`prometheus_fastapi_instrumentator`** for metrics; OTLP for traces.
- **Gunicorn + Uvicorn workers** in production; container deploy.

Continue to **[Enterprise production engineering](12-production-engineering.md)**.
