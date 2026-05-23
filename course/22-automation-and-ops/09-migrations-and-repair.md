# Migrations and repair scripts

Every production system needs occasional intervention: schema changes, data backfills, fixing rows corrupted by a bug, splitting one table into two. These are **migrations** (planned, versioned, reversible) and **repair scripts** (ad-hoc, often urgent, always risky). This chapter is the discipline for both.

## Schema migrations with Alembic (recap)

Module 22 chapter 3 covered Alembic basics. The production rules:

1. **Every schema change is a versioned migration**, committed to source control.
2. **Auto-generate locally; review the diff; commit; deploy.** Never run `alembic upgrade head --autogenerate` against production blindly.
3. **Migrations are forward-only in production**. Test `downgrade` in CI so you *can* roll back; but in practice, you usually fix forward.
4. **Run migrations as a separate deployment step** before the app code that depends on them.

## Data migrations — the dangerous cousin

A schema migration changes the table structure. A data migration changes the *contents* of existing rows. They're often combined ("add column `notional` then backfill from `qty * price`"), but data migrations have their own failure mode: they're long-running and not always reversible.

Pattern for a safe data migration:

```python
# alembic/versions/XYZ_backfill_notional.py
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text


def upgrade():
    # 1. Add the column nullable (cheap; non-blocking)
    op.add_column("orders", sa.Column("notional", sa.Float, nullable=True))
    # 2. Backfill in batches (avoid long lock)
    conn = op.get_bind()
    batch_size = 10_000
    while True:
        result = conn.execute(text("""
            UPDATE orders
            SET notional = qty * price
            WHERE notional IS NULL
            LIMIT :n
        """), {"n": batch_size})
        if result.rowcount == 0:
            break
    # 3. Now make it NOT NULL (cheap; only after backfill is complete)
    op.alter_column("orders", "notional", nullable=False)


def downgrade():
    op.drop_column("orders", "notional")
```

Three-step pattern: cheap-DDL → batched-backfill → cheap-DDL-constraint.

The batched backfill is critical: a `UPDATE orders SET notional = qty * price` on a 50M-row table takes a long lock; everything else waiting on `orders` stalls. Small batches release the lock between iterations.

## Repair scripts

A repair script fixes inconsistent state caused by a bug. Examples:

- "Orders table has 500 rows where `filled_qty > qty`; bug in calc; need to recompute."
- "Position state in the trading DB doesn't match broker state; need to reconcile."
- "Decimal precision bug: 100 rows have `price = 47.300000000004` instead of `47.30`."

Five principles:

### 1. Always write a dry-run mode first

```python
def repair_overfilled_orders(conn, dry_run: bool = True) -> dict:
    """Re-compute filled_qty from the fills table; alert if mismatched."""
    mismatches = conn.execute(text("""
        SELECT o.id, o.filled_qty, COALESCE(SUM(f.qty), 0) AS actual_filled
        FROM orders o
        LEFT JOIN fills f ON f.order_id = o.id
        GROUP BY o.id, o.filled_qty
        HAVING o.filled_qty != COALESCE(SUM(f.qty), 0)
    """)).fetchall()
    stats = {"mismatches_found": len(mismatches), "fixed": 0}
    for order_id, current, actual in mismatches:
        if dry_run:
            print(f"  would update order {order_id}: {current} → {actual}")
        else:
            conn.execute(text("UPDATE orders SET filled_qty = :a WHERE id = :id"),
                         {"a": actual, "id": order_id})
            stats["fixed"] += 1
    return stats
```

Always call with `dry_run=True` first. Inspect the proposed changes. Only then `dry_run=False`.

### 2. Bound the blast radius

```python
def repair(conn, dry_run: bool = True, max_changes: int = 100):
    # ... compute changes ...
    if len(changes) > max_changes:
        raise RuntimeError(f"would change {len(changes)} rows; refusing (max: {max_changes})")
    # ... apply changes ...
```

If the script proposes to change 50,000 rows when you expected 50, something is wrong. Refuse to proceed; investigate; raise the cap deliberately.

### 3. Audit trail

```python
def repair_with_audit(conn, run_id: str, dry_run: bool = True):
    """Log every change to an audit table before applying."""
    audit_rows = []
    for change in proposed_changes:
        audit_rows.append({
            "run_id": run_id,
            "table_name": "orders",
            "row_id": change.id,
            "column_name": "filled_qty",
            "old_value": change.old,
            "new_value": change.new,
            "applied_at": datetime.now(timezone.utc),
            "dry_run": dry_run,
        })
    conn.execute(insert(audit_table), audit_rows)
    if not dry_run:
        for change in proposed_changes:
            conn.execute(text("UPDATE orders SET filled_qty = :v WHERE id = :id"),
                         {"v": change.new, "id": change.id})
```

After a repair, an auditor (or you, in two weeks) can reconstruct exactly what was changed and why. Without an audit trail, the next bug debugging session is impossible.

### 4. Idempotent

A repair script should be safe to re-run. The dry-run finds 0 mismatches the second time. Achieve this by the script's logic itself — match-then-fix patterns are naturally idempotent.

### 5. Transactional where possible

```python
with conn.begin() as txn:
    rows_to_fix = find_bad_rows(conn)
    for row in rows_to_fix:
        conn.execute(text("UPDATE ... WHERE id = :id"), {"id": row.id})
    # txn commits here on clean exit; rolls back on exception
```

If anything mid-loop goes wrong, the entire repair rolls back. Either everything is fixed, or nothing changed. Never half-fixed.

For very large repairs, batched transactions: commit every N rows so you don't hold a multi-hour transaction.

## A worked example: position reconciliation

You suspect the local "positions" table has drifted from the broker's actual positions. Reconcile, with dry-run + audit + bounded:

```python
import logging
from datetime import datetime, timezone
from sqlalchemy import text, insert

log = logging.getLogger(__name__)


def reconcile_positions(engine, broker, run_id: str, dry_run: bool = True,
                          max_changes: int = 50) -> dict:
    """Fix local positions to match broker reality.
    Logs every proposed change; refuses to proceed past max_changes."""
    broker_positions = broker.positions()      # {symbol: qty}

    with engine.begin() as conn:
        local = dict(conn.execute(text("SELECT symbol, qty FROM positions")).fetchall())

        # Compute changes
        changes = []
        for sym in set(local) | set(broker_positions):
            local_qty = local.get(sym, 0)
            broker_qty = broker_positions.get(sym, 0)
            if local_qty != broker_qty:
                changes.append({"symbol": sym, "old": local_qty, "new": broker_qty})

        log.info("found %d position discrepancies", len(changes))
        for c in changes:
            log.info("  %s: local=%d, broker=%d (delta=%+d)",
                     c["symbol"], c["old"], c["new"], c["new"] - c["old"])

        if len(changes) > max_changes:
            raise RuntimeError(f"refusing: {len(changes)} changes exceeds cap {max_changes}")

        # Always write audit (even in dry-run)
        for c in changes:
            conn.execute(text("""
                INSERT INTO position_audit (run_id, symbol, old_qty, new_qty, applied_at, dry_run)
                VALUES (:run_id, :symbol, :old, :new, :applied_at, :dry_run)
            """), {
                "run_id": run_id, "symbol": c["symbol"], "old": c["old"], "new": c["new"],
                "applied_at": datetime.now(timezone.utc), "dry_run": dry_run,
            })

        if not dry_run:
            for c in changes:
                conn.execute(text("""
                    INSERT INTO positions (symbol, qty) VALUES (:symbol, :qty)
                    ON CONFLICT (symbol) DO UPDATE SET qty = EXCLUDED.qty
                """), {"symbol": c["symbol"], "qty": c["new"]})

    return {"changes": len(changes), "dry_run": dry_run}
```

This is the canonical repair-script shape: dry-run-first, audit, bounded, transactional, idempotent.

## The "fix in place" anti-pattern

Don't do this:

```python
# DON'T
conn.execute(text("UPDATE orders SET filled_qty = 0 WHERE filled_qty > qty"))
```

It's a single statement, no audit, no review. If your `WHERE` is wrong, you've nuked half the table.

The discipline: write the SELECT first, eyeball the rows, then turn it into the UPDATE. With audit + dry-run.

## Pitfalls

!!! warning "Missing or wrong WHERE clause"
    `UPDATE orders SET status = 'CANCELED'` — without WHERE — cancels every order ever. This is how funds lose money in minutes.

!!! warning "No backup before destructive change"
    Always snapshot the table (or relevant rows) before bulk operations. `CREATE TABLE orders_backup_20241115 AS SELECT * FROM orders;`

!!! warning "Long-running migrations on a live DB"
    Lock contention starves application queries. Batch in small chunks; commit between; pause if needed.

!!! warning "Repair scripts kept around forever**
    A repair script is for one specific incident. After it's run, archive it (move to `scripts/done/`) with a timestamp; don't leave it as live code where someone might run it again.

## Bottom line

For migrations and repair:

- **Alembic** for schema migrations; committed, reviewed, separate-step deployed.
- **Three-step data migrations**: cheap DDL → batched backfill → constrain.
- **Dry-run mode** on every repair script.
- **Bound the blast radius** with a `max_changes` cap.
- **Audit trail** for every applied change.
- **Snapshot before bulk operations**.
- **Archive used repair scripts** to prevent accidental re-runs.

Continue to **[AI workflows in production](10-ai-workflows.md)**.
