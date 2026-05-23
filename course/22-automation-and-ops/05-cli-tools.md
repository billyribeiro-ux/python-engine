# CLI tools and shell scripting in Python

Half of operational engineering is writing the script you wish bash could do. Python is dramatically better than bash for anything non-trivial, but only if you write it as a *real* command-line tool — with arguments, exit codes, environment-variable defaults, and behaviour suitable for being chained into other tools.

## `argparse` — stdlib, always works

For a script with a few arguments, `argparse` is enough:

```python
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Backfill bars for a symbol")
    parser.add_argument("symbol", help="ticker symbol")
    parser.add_argument("--start", default="2020-01-01", help="ISO date")
    parser.add_argument("--end", default=None, help="ISO date (default: today)")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "5m", "1m"])
    parser.add_argument("--dry-run", action="store_true", help="don't write to disk")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for INFO, -vv for DEBUG")
    parser.add_argument("--out", type=Path, default=Path("data/bars"), help="output directory")
    args = parser.parse_args()
    return run(args)


def run(args) -> int:
    # ... do the work, return 0 for success, non-zero for failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Key conventions:

- **`sys.exit(main())`** — return an int from `main`; let `sys.exit` propagate it. Non-zero = failure; 0 = success.
- **`--dry-run`** for any script that writes — the standard pattern for "show me what you'd do".
- **`-v` / `-vv` counter** for log verbosity.
- **`type=Path`** for path arguments — avoids string-vs-path confusion later.
- **`choices=[...]`** for discrete values — gives helpful error messages.

## `click` — when you have subcommands

For multi-command tools (`mytool backfill ...`, `mytool inspect ...`), `click` is the standard:

```python
import click


@click.group()
def cli():
    """Python Engine ops CLI."""


@cli.command()
@click.argument("symbol")
@click.option("--start", default="2020-01-01")
@click.option("--end", default=None)
@click.option("--dry-run", is_flag=True)
def backfill(symbol, start, end, dry_run):
    """Backfill bars for SYMBOL."""
    # ...


@cli.command()
@click.argument("symbol")
def inspect(symbol):
    """Inspect cached bars for SYMBOL."""
    # ...


if __name__ == "__main__":
    cli()
```

Usage: `python tool.py backfill SPY --start 2020-01-01`. Help: `python tool.py backfill --help`.

Click handles colour, prompts, progress bars, password input, table output. For a real CLI used by humans, it's much nicer than argparse.

## `typer` — type-hint-driven CLIs

For the cleanest API, `typer` derives argument parsing from your function's type hints:

```python
import typer
from typing import Optional


app = typer.Typer()


@app.command()
def backfill(
    symbol: str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    dry_run: bool = False,
):
    """Backfill bars for SYMBOL."""
    # ...


if __name__ == "__main__":
    app()
```

That's the whole thing. `typer` reads the type hints and generates the same CLI as click. Recommend it for new projects.

## `subprocess` — running other commands

For shelling out:

```python
import subprocess


# Capture output
result = subprocess.run(
    ["git", "log", "--oneline", "-10"],
    capture_output=True,
    text=True,
    check=True,                       # raise on non-zero exit
    timeout=30,
)
print(result.stdout)
```

Patterns to internalise:

- **List of args**, not a shell string. Avoids shell injection. `shell=True` is dangerous; never with untrusted input.
- **`text=True`** for string output (vs bytes).
- **`check=True`** raises `CalledProcessError` on non-zero exit. Don't silently swallow errors.
- **`timeout=`** — never leave a subprocess unbounded.

For long-running subprocesses where you want to stream output:

```python
proc = subprocess.Popen(
    ["python", "long_running.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,                        # line buffered
)
for line in proc.stdout:
    print("[subproc]", line, end="")
proc.wait()
```

## Environment variables and config

For config that varies by environment (dev vs prod), env vars are the standard:

```python
import os

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    print("Missing POLYGON_API_KEY", file=sys.stderr)
    sys.exit(1)

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
```

For richer config, `pydantic-settings` (Module 1 chapter 6) is the production answer.

## Exit codes — convention

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | generic failure |
| 2 | misuse (bad arguments) |
| 64-78 | reserved by `<sysexits.h>` — useful for shell-integration tools |
| 130 | killed by Ctrl-C (SIGINT) |
| 143 | killed by SIGTERM |

Be deliberate. A script that always exits 0 even on failure makes "if this script succeeded, do that" pipelines impossible.

## Signal handling

```python
import signal
import sys


def graceful_shutdown(signum, frame):
    print(f"Received signal {signum}; cleaning up...")
    # ... close DB connections, flush logs, etc.
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
```

For long-running scripts and services, handle SIGTERM (sent by Docker / Kubernetes / systemd for graceful shutdown) and SIGINT (Ctrl-C). Without handlers, the process gets killed mid-write, leaving inconsistent state.

## Logging from scripts

Not `print()` — `logging`:

```python
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

log.info("starting backfill for %s", symbol)
try:
    do_work()
except Exception:
    log.exception("backfill failed")    # captures the traceback
    sys.exit(1)
```

Always `.exception(...)` (not `.error(...)`) inside `except:` — captures the traceback. Save yourself debugging time.

## A pattern: "wrap a long-running batch as a CLI"

```python
import sys
import typer
import logging
from pathlib import Path
from datetime import datetime


app = typer.Typer()
log = logging.getLogger(__name__)


def _setup_logging(verbose: int):
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@app.command()
def backfill(
    symbol: str,
    start: str = "2020-01-01",
    end: str = None,
    interval: str = "1d",
    out: Path = Path("data/bars"),
    dry_run: bool = False,
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
):
    """Backfill bars for SYMBOL into OUT directory."""
    _setup_logging(verbose)
    end = end or datetime.utcnow().date().isoformat()
    log.info("backfill %s %s..%s interval=%s dry_run=%s", symbol, start, end, interval, dry_run)
    try:
        if dry_run:
            log.warning("dry run; no files will be written")
            return
        from engine.data import YFinanceFeed, ParquetCache
        cache = ParquetCache(YFinanceFeed(), root=out)
        bars = cache.bars(symbol, start, end, interval=interval)
        log.info("wrote %d bars", len(bars))
    except KeyboardInterrupt:
        log.warning("interrupted")
        sys.exit(130)
    except Exception:
        log.exception("backfill failed")
        sys.exit(1)


if __name__ == "__main__":
    app()
```

That's a complete, production-quality ops script: argument parsing, logging, dry-run support, exception handling, exit codes, environment-friendly. Maybe 40 lines.

## Pitfalls

!!! warning "`print` for "logs""
    Print sends to stdout; logs go to stderr. Print can't be filtered by level. Always `logging`.

!!! warning "Capture exceptions broadly"
    `except Exception:` is fine at the top of a script. `except:` (no class) catches `KeyboardInterrupt` and `SystemExit` — almost always wrong.

!!! warning "`subprocess` with `shell=True` and untrusted input**
    Classic injection. If you need shell features, sanitise; better, restructure to avoid the shell.

!!! warning "Long Python startup time"
    `python -c 'import pandas'` takes 1+ second. For a script invoked thousands of times in a loop (a git hook, a cron-per-minute), Python's startup is the bottleneck. Either keep the script running in a loop (poll instead of cron) or rewrite in faster startup language.

## Bottom line

For production CLI tools:

- **`typer`** for new code; `argparse` for the standard library; `click` for fine control.
- **`subprocess.run` with `check=True, timeout=, text=True, capture_output=True`**.
- **Exit codes**: 0 for success, non-zero for failure. Be deliberate.
- **Signal handling** for long-running scripts.
- **`logging`** with `.exception()` in `except` blocks.

Continue to **[Scheduling — cron, systemd, APScheduler](06-scheduling.md)**.
