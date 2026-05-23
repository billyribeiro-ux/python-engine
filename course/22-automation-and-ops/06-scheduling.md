# Scheduling — cron, systemd, APScheduler

You wrote a script. Now it has to run every day at 06:30, or every 5 minutes, or after market close. This chapter is the decision tree for *how* to schedule it: cron when the script is small, systemd when you need reliability, APScheduler when scheduling logic lives in your application.

## The decision matrix

| Need | Use |
|---|---|
| Daily / weekly / hourly batch on a single Linux box | **cron** |
| Same, but with monitoring / dependency / retries built in | **systemd timer** |
| Inside a long-running Python process | **APScheduler** |
| Container orchestration (K8s, ECS) | **K8s CronJob** / **ECS Scheduled Tasks** |
| Cloud serverless | **EventBridge Scheduler** (AWS) / **Cloud Scheduler** (GCP) |
| Cross-machine, dependency graphs | **Airflow** / **Dagster** / **Prefect** |

## cron — the basics

```cron
# m h dom mon dow command
30 6 * * 1-5 /usr/bin/python /opt/scripts/morning_backfill.py >> /var/log/cron/morning.log 2>&1
0 17 * * 1-5 /usr/bin/python /opt/scripts/eod_report.py
*/5 9-16 * * 1-5 /usr/bin/python /opt/scripts/intraday_scanner.py
```

The five fields: minute, hour, day-of-month, month, day-of-week.

**Always redirect stdout + stderr to a log file** — cron mails errors by default; on most boxes mail isn't configured, so errors silently disappear.

`crontab -e` to edit per-user cron. `/etc/cron.d/myjob` for system-wide jobs.

### cron pitfalls

1. **PATH is minimal**. `python` may not be found; specify the full path.
2. **Cwd is `$HOME`**. Use absolute paths in scripts; or `cd /opt/myapp && ...`.
3. **Environment is empty**. Source `.env` first if your script depends on env vars.
4. **No overlap protection**. If a 6:30 job takes 90 minutes, the 7:30 fire of the same job starts in parallel. Use `flock`:

```cron
30 * * * * flock -n /var/lock/morning.lock /usr/bin/python /opt/scripts/morning_backfill.py
```

`flock -n` exits silently if the lock is held; `flock -w 60` waits up to 60 seconds.

5. **No retry**. If the script fails, cron doesn't retry. Build retry into the script (Module 5 chapter 2).

## systemd timers — cron with superpowers

systemd timers are the modern replacement for cron on Linux. They have:

- Native logging via `journalctl`.
- Built-in restart policies.
- Resource limits (CPU, memory).
- Dependency graphs.
- Calendar specs more flexible than cron.

```ini
# /etc/systemd/system/morning-backfill.service
[Unit]
Description=Morning bars backfill
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=trading
WorkingDirectory=/opt/trading
EnvironmentFile=/etc/trading/env
ExecStart=/opt/trading/.venv/bin/python /opt/trading/scripts/morning_backfill.py
StandardOutput=journal
StandardError=journal
```

```ini
# /etc/systemd/system/morning-backfill.timer
[Unit]
Description=Run morning bars backfill daily at 06:30

[Timer]
OnCalendar=Mon..Fri 06:30
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

`Persistent=true`: if the box was down when the job should have run, fire it immediately on startup.

`RandomizedDelaySec=60`: spread load. If 100 boxes have the same timer, they jitter their starts across 60 seconds.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now morning-backfill.timer
systemctl list-timers
journalctl -u morning-backfill -f
```

## APScheduler — for in-process scheduling

When your scheduling logic lives in a long-running Python application (a trading service, a Discord bot, a daemon), APScheduler is the idiomatic choice:

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone


sched = BlockingScheduler(timezone="America/New_York")


@sched.scheduled_job("cron", day_of_week="mon-fri", hour=9, minute=30)
def market_open():
    print(f"market open at {datetime.now(timezone.utc)}")


@sched.scheduled_job("interval", minutes=5)
def heartbeat():
    print(f"alive at {datetime.now(timezone.utc)}")


@sched.scheduled_job("cron", day_of_week="mon-fri", hour=16, minute=15)
def eod_report():
    generate_report()


if __name__ == "__main__":
    sched.start()
```

Three trigger types:

- **`cron`** — calendar-like.
- **`interval`** — every N seconds/minutes/hours.
- **`date`** — once at a specific timestamp.

For an `asyncio`-based application, use `AsyncIOScheduler` instead.

## Idempotent jobs

A scheduled job that runs twice should produce the same result as running once. The pattern:

```python
def my_daily_job(date: str):
    """Idempotent: re-running for the same date is a no-op."""
    state_path = Path(f"state/{date}.done")
    if state_path.exists():
        log.info(f"already ran for {date}")
        return
    # ... do the work
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.touch()
```

When a job's retry policy fires the same job twice, you want it to be safe. Built-in idempotency via a state file or database flag is the cleanest approach.

## Catch-up and `Persistent=true`

What happens when a scheduled job's time falls in a window when the machine was off?

- **cron** — silently skips.
- **systemd timer with `Persistent=true`** — fires immediately on next start.
- **APScheduler** — depends on `misfire_grace_time` setting.

For idempotent jobs, "catch up" is what you want. For non-idempotent (e.g., "send an email"), you may want to skip — a stale alert is worse than no alert.

## Leader election for high-availability

If you run your scheduler on multiple machines for HA, you need exactly one to actually fire each job.

For Kubernetes: a `CronJob` is replica-aware; only one pod runs per scheduled time.

For arbitrary machines: distributed lock via Redis, ZooKeeper, or etcd. Each machine tries to acquire the lock; only the winner runs the job.

```python
import redis


def with_redis_lock(redis_conn, lock_name, ttl_sec=300):
    """Context manager: only the lock holder enters the block."""
    # Try to acquire
    acquired = redis_conn.set(lock_name, "locked", nx=True, ex=ttl_sec)
    if not acquired:
        return False
    return True


# In your job:
r = redis.Redis(host="redis.internal")
if with_redis_lock(r, "morning_backfill_lock", ttl_sec=300):
    try:
        do_the_job()
    finally:
        r.delete("morning_backfill_lock")
else:
    log.info("another instance has the lock; skipping")
```

The `nx=True` is "set if not exists"; `ex=300` is "expire after 300s as a safety net." Both essential.

## GitHub Actions / GitLab CI as a scheduler

For lightweight, occasional jobs, hosted CI works as a scheduler:

```yaml
# .github/workflows/daily-report.yml
on:
  schedule:
    - cron: "30 11 * * 1-5"           # 11:30 UTC = 06:30 EST

jobs:
  run-daily-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[core,ml]"
      - run: python scripts/run_scanners.py
        env:
          POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
      - uses: actions/upload-artifact@v4
        with:
          name: daily-report
          path: daily_reports/
```

Limits: hosted runners have a ~5-minute startup, GitHub schedules drift by 5-15 minutes, secrets are usable but rate-limited. Fine for daily summaries; not for high-precision trading triggers.

## Workflow orchestrators

For multi-step pipelines with dependencies — "first fetch yesterday's prices, then run the ETL, then train the model, then generate the report" — a workflow tool is structurally what you need.

**Lightweight**: `make` with proper targets and dependencies (`make daily-report` triggers `backfill` and `train`).
**Mid-weight**: **Prefect** (Python-native, easy local + cloud).
**Heavyweight**: **Airflow** (industry standard, complex), **Dagster** (modern alternative).

For a small shop: cron + idempotent scripts is enough. For a fund with 100+ daily jobs and inter-dependencies: invest in a real orchestrator.

## Pitfalls

!!! warning "Timezone mismatches"
    Cron defaults to system timezone; APScheduler is timezone-aware. Always explicit. The single most common scheduling bug is "the job runs at 6:30 UTC, not 6:30 New York."

!!! warning "Overlapping runs"
    Without `flock`, a long-running job's next firing starts in parallel. Two writers to the same database = chaos. Always lock.

!!! warning "Cron silently swallows errors"
    Always log stdout+stderr to a file. Even better: have the script email/Slack on failure.

!!! warning "Cron doesn't tell you when jobs stop firing"
    Set up monitoring. A scheduled job that hasn't run in 24 hours should alert. **Dead-man's-switch** pattern: each successful run writes a heartbeat; a separate monitor alerts if heartbeats stop.

## Bottom line

For scheduling:

- **cron** for boring, simple periodic jobs.
- **systemd timer** for anything you want logged + restartable.
- **APScheduler** for in-process scheduling.
- **Idempotency** in every scheduled job.
- **`flock`** for cron overlap protection.
- **Dead-man's-switch monitoring** so silent failures don't last days.

Continue to **[Data pipelines and ETL](07-data-pipelines.md)**.
