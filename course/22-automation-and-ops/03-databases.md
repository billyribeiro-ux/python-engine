# Database scripting

Most production data engineering eventually hits a database. SQLite for local stores; PostgreSQL or MySQL for shared transactional systems; DuckDB for analytical scratch space (Module 4 chapter 6); ClickHouse or TimescaleDB for time-series. This chapter is the working Python toolkit across all of them.

## `sqlite3` — the standard library's gift

For local single-user state, sqlite3 is dramatically underrated. Zero setup; transactional; ACID; reliable enough that Boeing puts it in aeroplanes.

```python
import sqlite3
from pathlib import Path

DB = Path("orders.db")

def init_schema():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty INTEGER NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                status TEXT DEFAULT 'PENDING'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")


def insert_order(symbol: str, qty: int, side: str):
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO orders (ts, symbol, qty, side) VALUES (datetime('now'), ?, ?, ?)",
            (symbol, qty, side),
        )
```

Three patterns to internalise:

1. **Parameterise queries.** `?` placeholders. **Never** f-string SQL — that's SQL injection.
2. **`with sqlite3.connect(...)`** auto-commits on exit, rolls back on exception.
3. **Indexes on the columns you query by.** Without indexes, every query is a full table scan.

## SQLAlchemy Core — the right level of abstraction for scripts

SQLAlchemy has two layers: Core (SQL expressions) and ORM (Python classes mapped to tables). For operational scripts, Core is usually right — it gives you the SQL abstraction without the impedance mismatch.

```python
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData, select, insert

engine = create_engine("postgresql://user:pass@host/db", future=True)
metadata = MetaData()

orders = Table(
    "orders", metadata,
    Column("id", Integer, primary_key=True),
    Column("ts", DateTime, nullable=False),
    Column("symbol", String(16), nullable=False, index=True),
    Column("qty", Integer, nullable=False),
    Column("side", String(8), nullable=False),
)

metadata.create_all(engine)

# Insert
with engine.begin() as conn:
    conn.execute(
        insert(orders),
        [
            {"ts": now, "symbol": "SPY", "qty": 100, "side": "BUY"},
            {"ts": now, "symbol": "QQQ", "qty": 50, "side": "SELL"},
        ],
    )

# Query
with engine.connect() as conn:
    rows = conn.execute(select(orders).where(orders.c.symbol == "SPY")).fetchall()
```

`engine.begin()` is a transaction; commits on success, rolls back on exception. Use it for any write. `engine.connect()` is read-only and doesn't commit.

## SQLAlchemy ORM — for richer domain modelling

When the schema represents a domain (orders, fills, positions), ORM gives you Python classes with methods and relationships:

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    qty: Mapped[int]
    side: Mapped[str]
    fills: Mapped[list["Fill"]] = relationship(back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    qty: Mapped[int]
    price: Mapped[float]
    order: Mapped[Order] = relationship(back_populates="fills")
```

Querying:

```python
from sqlalchemy.orm import Session

with Session(engine) as session:
    order = session.scalar(select(Order).where(Order.symbol == "SPY"))
    if order:
        for fill in order.fills:
            print(fill.ts, fill.price)
```

The ORM is great for read-heavy operational scripts where each object's relationships are explored. Avoid the ORM for write-heavy bulk operations — use Core's `insert([...])` or raw SQL for speed.

## `psycopg` (v3) — direct Postgres

For Postgres-specific work, `psycopg` (the v3 successor to `psycopg2`) is excellent:

```python
import psycopg
from psycopg.rows import dict_row

with psycopg.connect("postgresql://user:pass@host/db") as conn:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM orders WHERE symbol = %s AND ts > %s", ("SPY", since))
        for row in cur:
            process(row)
```

Killer features:

- **Native async**: `async with await psycopg.AsyncConnection.connect(...)`
- **COPY for bulk loads**: dramatically faster than INSERTs for thousands of rows.
- **Listen/Notify**: real-time pub/sub via Postgres' notification system.

## Transactions, idempotent upserts

For "insert if new, update if exists" without race conditions, use the database's native upsert:

```python
# Postgres / SQLite (3.24+)
with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO positions (symbol, qty)
            VALUES (:symbol, :qty)
            ON CONFLICT (symbol) DO UPDATE SET qty = positions.qty + :qty
        """),
        {"symbol": "SPY", "qty": 100},
    )
```

`ON CONFLICT ... DO UPDATE` is the idempotent pattern. Multiple processes calling this concurrently produces consistent results.

For MySQL: `ON DUPLICATE KEY UPDATE`. For SQL Server: `MERGE`. The syntax varies; the concept is universal.

## Migrations — Alembic

Schema changes need to be versioned. **Alembic** is SQLAlchemy's migration tool:

```bash
alembic init migrations
# Edit alembic.ini to point at your database
alembic revision --autogenerate -m "add orders.notional column"
# Inspect the generated migration script
alembic upgrade head
```

Each migration is a Python script with `upgrade()` and `downgrade()` functions:

```python
# migrations/versions/abc123_add_orders_notional.py
from alembic import op
import sqlalchemy as sa


def upgrade():
    op.add_column("orders", sa.Column("notional", sa.Float, nullable=True))


def downgrade():
    op.drop_column("orders", "notional")
```

Run `alembic upgrade head` to apply pending migrations; `alembic downgrade -1` to revert.

For production:

- **Never run autogenerate against production blindly.** Generate locally, review, commit the migration file.
- **Run migrations as part of deployment**, before the new app code starts.
- **Test migrations with `downgrade`** in CI so you know they're reversible.

## Connection pooling

For long-running services, **don't create a new connection per request**. Use a pool:

```python
engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,           # check the connection is alive before use
    pool_recycle=3600,            # recycle connections after 1 hour
)
```

`pool_pre_ping=True` adds a small overhead but avoids "stale connection" errors when the database closes idle connections.

## A pattern: idempotent ETL into a database

```python
def upsert_daily_bars(engine, bars_df: pd.DataFrame, table: str = "bars"):
    """Idempotent: re-running with the same data is a no-op."""
    if bars_df.empty:
        return 0
    with engine.begin() as conn:
        rows = bars_df.reset_index().to_dict(orient="records")
        for chunk in _chunks(rows, 1000):
            conn.execute(
                text(f"""
                    INSERT INTO {table} (timestamp, symbol, open, high, low, close, volume)
                    VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)
                    ON CONFLICT (timestamp, symbol) DO UPDATE
                    SET open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
                """),
                chunk,
            )
    return len(rows)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
```

Re-runnable; chunked for memory; transactional; upsert-on-conflict. The shape of every production ETL.

## Pitfalls

!!! warning "SQL injection"
    `f"SELECT * FROM users WHERE name = '{name}'"` is exploitable. Always parameterise.

!!! warning "Long-running transactions"
    A transaction holds locks. A query that takes 2 hours blocks every other writer for 2 hours. Keep transactions short.

!!! warning "N+1 queries via ORM"
    `for order in orders: print(order.fills)` queries the database once per order. Use `selectinload(Order.fills)` to eagerly fetch related data in one query.

!!! warning "No connection limits**
    Without `pool_size` and `max_overflow`, you can exhaust the database server's connection limit. The DB then refuses connections from everyone. Bound your pool.

!!! warning "Forgetting to index foreign keys"
    Most databases don't auto-index FK columns. Joins on un-indexed FKs are full table scans. Always `Column(ForeignKey(...), index=True)`.

## Bottom line

For production database work:

- **sqlite3** for local single-user state.
- **SQLAlchemy Core** for operational scripts.
- **SQLAlchemy ORM** for richer domain modelling.
- **psycopg v3** for direct Postgres (async, COPY, Listen/Notify).
- **Alembic** for schema migrations.
- **`ON CONFLICT DO UPDATE`** for idempotent upserts.
- **Connection pooling** for long-running services.

Continue to **[Web scraping done right](04-web-scraping.md)**.
