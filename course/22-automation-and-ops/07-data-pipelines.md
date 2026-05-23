# Data pipelines and ETL

A data pipeline is a series of steps that move and transform data. The "ETL" acronym — Extract, Transform, Load — captures the canonical shape. This chapter is the working patterns for building pipelines that are **idempotent**, **observable**, **resumable**, and **boring**, which is what production needs.

## The taxonomy

| Type | Trigger | Example |
|---|---|---|
| **Batch** | Scheduled (nightly, hourly) | EOD price ingest, daily reports |
| **Incremental** | Watermark-based | Append new rows since last run |
| **Streaming** | Continuous | Live market data, fraud detection |
| **Reactive** | Event-driven | "When a new file appears, process it" |

This chapter focuses on batch + incremental — the bulk of operational ETL.

## The three properties production pipelines need

### 1. Idempotency

Running the pipeline twice should produce the same result as running it once. The pattern: design every step to be re-runnable.

**Anti-pattern**: `INSERT INTO daily_bars (...)`. A re-run creates duplicates.
**Pattern**: `INSERT ... ON CONFLICT (date, symbol) DO UPDATE` (Module 22 ch3).

For file outputs:

```python
# Anti-pattern: append
with open("results.csv", "a") as f:
    f.write(new_data)

# Pattern: write a complete, versioned file
output_dir = Path(f"results/{run_date}/")
output_dir.mkdir(parents=True, exist_ok=True)
atomic_write_text(output_dir / "results.csv", complete_data)
```

A re-run for the same `run_date` produces the same file. Downstream consumers re-read the same content.

### 2. Resumability

If the pipeline crashes at step 5 of 10, restarting shouldn't redo steps 1-4. **Checkpoint** state between steps:

```python
import json
from pathlib import Path


class PipelineCheckpoint:
    def __init__(self, run_id: str, root: Path = Path(".checkpoints")):
        self.path = root / f"{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, step: str) -> bool:
        return self.state.get(step) == "done"

    def mark_done(self, step: str):
        self.state[step] = "done"
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


def run_pipeline(run_date: str):
    cp = PipelineCheckpoint(run_id=run_date)

    if not cp.done("extract"):
        extract(run_date)
        cp.mark_done("extract")

    if not cp.done("transform"):
        transform(run_date)
        cp.mark_done("transform")

    if not cp.done("load"):
        load(run_date)
        cp.mark_done("load")
```

A crash mid-transform → restart resumes at transform (skipping extract).

### 3. Observability

Every step logs its inputs, outputs, and duration:

```python
import logging
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def step(name: str, **context):
    log.info("step_start", extra={"step": name, **context})
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("step_failed", extra={"step": name, **context})
        raise
    finally:
        dt = time.perf_counter() - t0
        log.info("step_complete", extra={"step": name, "duration_s": dt, **context})


# Usage
with step("extract", date=run_date):
    extract(run_date)
```

Structured logs (Module 21 chapter 5) let you query "show all step_failed events for the last 7 days, grouped by step name."

## Watermark-based incremental loading

For "load all new rows since last run", track a watermark:

```python
def incremental_load(table: str, source_query: str):
    """Pull rows from source_query newer than the last watermark."""
    watermark_path = Path(f".watermarks/{table}.txt")
    last_seen = watermark_path.read_text().strip() if watermark_path.exists() else "1970-01-01"
    new_rows = fetch_from_source(source_query, since=last_seen)
    if new_rows.empty:
        log.info("no new rows since %s", last_seen)
        return
    new_watermark = new_rows["updated_at"].max().isoformat()
    upsert_to_db(table, new_rows)
    # Update watermark only AFTER successful load
    watermark_path.parent.mkdir(parents=True, exist_ok=True)
    watermark_path.write_text(new_watermark)
```

Critical: update the watermark only after the load is committed. If it crashes before the watermark write, the next run picks up the same rows again (idempotent upsert handles duplicates).

## Backfills

When you change a transformation, you often need to re-process historical data. Design pipelines so a backfill is just "loop the date range and call the daily pipeline":

```python
def backfill(start_date: str, end_date: str):
    for date in pd.date_range(start_date, end_date, freq="B"):
        log.info("backfilling %s", date.date())
        run_pipeline(date.date().isoformat())
```

Because `run_pipeline` is idempotent, you can interrupt and restart with no harm.

## A worked example: a small batch ETL

```python
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from engine.data import YFinanceFeed, ParquetCache

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, run_date: str, output_root: Path = Path("data/etl")):
        self.run_date = run_date
        self.root = output_root / run_date
        self.checkpoint = PipelineCheckpoint(run_id=run_date)
        self.feed = ParquetCache(YFinanceFeed(), root="data/bars")
        self.engine = create_engine("sqlite:///data/etl.db")

    def extract(self):
        if self.checkpoint.done("extract"):
            return
        with step("extract", date=self.run_date):
            symbols = ["SPY", "QQQ", "IWM"]
            self.root.mkdir(parents=True, exist_ok=True)
            for sym in symbols:
                bars = self.feed.bars(sym, "2024-01-01", self.run_date)
                atomic_write_bytes(self.root / f"{sym}.parquet", bars.to_parquet(compression="zstd"))
        self.checkpoint.mark_done("extract")

    def transform(self):
        if self.checkpoint.done("transform"):
            return
        with step("transform", date=self.run_date):
            frames = []
            for path in self.root.glob("*.parquet"):
                df = pd.read_parquet(path)
                df["symbol"] = path.stem
                df["return"] = df["close"].pct_change()
                df["vol_20"] = df["return"].rolling(20).std() * (252 ** 0.5)
                frames.append(df.reset_index())
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(self.root / "combined.parquet", compression="zstd")
        self.checkpoint.mark_done("transform")

    def load(self):
        if self.checkpoint.done("load"):
            return
        with step("load", date=self.run_date):
            combined = pd.read_parquet(self.root / "combined.parquet")
            combined.to_sql("daily_bars", self.engine, if_exists="append", index=False,
                              chunksize=10_000, method="multi")
        self.checkpoint.mark_done("load")

    def run(self):
        self.extract()
        self.transform()
        self.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = datetime.now(timezone.utc).date().isoformat()
    Pipeline(today).run()
```

That's a complete batch pipeline. Idempotent (the upsert + checkpointing), resumable (each step checks done), observable (`step` context manager). About 50 lines.

## Makefile as a pipeline

For small pipelines, `make` is surprisingly good. It tracks file dependencies, only rebuilds what's stale, and parallelises with `-j`:

```makefile
# data/raw/%.parquet: pulled from vendor
data/raw/%.parquet:
	python scripts/pull_one.py $(basename $(notdir $@)) $@

# data/clean/%.parquet: depends on the raw
data/clean/%.parquet: data/raw/%.parquet
	python scripts/clean.py $< $@

# Final report depends on cleaned versions of every symbol
SYMBOLS := SPY QQQ IWM TLT
CLEAN := $(patsubst %,data/clean/%.parquet,$(SYMBOLS))

report.html: $(CLEAN)
	python scripts/build_report.py $(CLEAN) $@

.PHONY: all clean
all: report.html
clean:
	rm -rf data/clean report.html
```

`make -j8 report.html` rebuilds in parallel. `make` knows what's already up-to-date and skips it.

For small pipelines (10-20 steps), this is genuinely fine. For 100+ steps with non-file dependencies, move to Prefect / Dagster / Airflow.

## Dagster — the modern orchestrator

For a pipeline that lives across many files / processes:

```python
from dagster import asset, AssetExecutionContext


@asset
def raw_bars(context: AssetExecutionContext) -> pd.DataFrame:
    from engine.data import YFinanceFeed
    feed = YFinanceFeed()
    return feed.bars("SPY", "2024-01-01", "2024-12-31")


@asset
def clean_bars(raw_bars: pd.DataFrame) -> pd.DataFrame:
    return raw_bars.dropna().pipe(add_features)


@asset
def report(clean_bars: pd.DataFrame) -> None:
    write_report(clean_bars, "out/report.html")
```

Dagster auto-detects dependencies, schedules, retries, observes, and presents a UI showing your pipeline graph. For non-trivial pipelines, the productivity gain is real.

## Pitfalls

!!! warning "Mutable in-place data transformations"
    "Append the new rows to the existing file" is fragile. A crash mid-append leaves the file in an unknown state. Always write a complete new file (or a new partition) and atomically replace.

!!! warning "Watermarks that race with the source"
    If the source is still writing new rows when you read, your watermark may exclude rows added between your last and current read. Use a watermark with a lag (e.g., "since last run, but never within the past 5 minutes").

!!! warning "Pipeline that's hard to re-run"
    If re-processing a single day requires unpicking other days' work, you've coupled too tightly. Design for date-independent processing per partition.

!!! warning "No DAG visualisation"
    Pipelines with 20+ steps and implicit dependencies are unmaintainable. A tool like Dagster or Airflow (or even just a Makefile dependency graph) is worth it past some complexity.

## Bottom line

For production data pipelines:

- **Idempotent steps** — re-running is a no-op.
- **Checkpoint between steps** — resume after failure.
- **Structured logging** — observable.
- **Watermarks** for incremental loads.
- **Atomic file outputs** — no half-written results.
- **Tool ladder**: cron + scripts → Makefile → Dagster / Prefect / Airflow.

Continue to **[Admin tooling and remote ops](08-admin-tooling.md)**.
