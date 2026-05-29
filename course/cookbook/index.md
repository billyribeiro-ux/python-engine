# Cookbook

100+ short, copy-pasteable recipes for everyday Python tasks. Index below; jump to whichever section you need.

- **[Files & paths](#files-paths)**
- **[Text & strings](#text-strings)**
- **[Dates & times](#dates-times)**
- **[HTTP & APIs](#http-apis)**
- **[JSON, YAML, TOML](#json-yaml-toml)**
- **[CSV & Excel](#csv-excel)**
- **[Databases](#databases)**
- **[Concurrency](#concurrency)**
- **[Subprocesses & shell](#subprocesses-shell)**
- **[Environment & config](#environment-config)**
- **[Logging](#logging)**
- **[Crypto & hashing](#crypto-hashing)**
- **[Numeric & arrays](#numeric-arrays)**
- **[Pandas one-liners](#pandas-one-liners)**
- **[Errors & retries](#errors-retries)**
- **[System & OS](#system-os)**

---

## Files & paths

**List all files under a tree, recursively:**

```python
from pathlib import Path
files = list(Path("data").rglob("*.csv"))
```

**Sort files by modification time:**

```python
files = sorted(Path(".").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
```

**Read text safely with explicit encoding:**

```python
content = Path("file.txt").read_text(encoding="utf-8", errors="replace")
```

**Atomic write:**

```python
import os
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(content, encoding="utf-8")
os.replace(tmp, path)
```

**Read large file line by line:**

```python
with open("huge.log") as f:
    for line in f:
        process(line.rstrip())
```

**Tail the last N lines:**

```python
from collections import deque
with open("log.txt") as f:
    last = deque(f, maxlen=100)
print("".join(last))
```

**Get file size:**

```python
size_bytes = Path("file.txt").stat().st_size
```

**SHA-256 of a file (streaming):**

```python
import hashlib
h = hashlib.sha256()
with open(path, "rb") as f:
    while chunk := f.read(64 * 1024):
        h.update(chunk)
print(h.hexdigest())
```

**Remove a directory tree:**

```python
import shutil
shutil.rmtree("path/to/dir", ignore_errors=True)
```

**Make a temporary file:**

```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tf:
    tf.write(b"data")
    tf.flush()
    process(tf.name)
```

---

## Text & strings

**Strip whitespace from each line:**

```python
clean = "\n".join(line.strip() for line in text.splitlines())
```

**Replace multiple whitespace with single space:**

```python
import re
clean = re.sub(r"\s+", " ", text).strip()
```

**Normalise unicode:**

```python
import unicodedata
clean = unicodedata.normalize("NFKC", text)
```

**Slugify (make URL-friendly):**

```python
import re
def slugify(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)
```

**Match a US ZIP code:**

```python
zip_match = re.fullmatch(r"\d{5}(-\d{4})?", "12345")
```

**Strip ANSI escape codes:**

```python
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
clean = ansi_re.sub("", colored_text)
```

**Wrap long text to N columns:**

```python
import textwrap
wrapped = textwrap.fill(long_string, width=80)
```

**Format a big number with commas:**

```python
f"{1234567:,}"                    # "1,234,567"
f"${1234.5:,.2f}"                 # "$1,234.50"
```

**Pad / centre a string:**

```python
"x".center(10, "-")               # "----x-----"
"x".rjust(5)                       # "    x"
"x".ljust(5, ".")                  # "x...."
```

---

## Dates & times

**Now, UTC, ISO 8601:**

```python
from datetime import datetime, timezone
datetime.now(tz=timezone.utc).isoformat()
```

**Parse ISO 8601:**

```python
datetime.fromisoformat("2024-11-15T14:30:00+00:00")
```

**Convert between timezones:**

```python
from zoneinfo import ZoneInfo
ny = datetime.now(ZoneInfo("America/New_York"))
utc = ny.astimezone(timezone.utc)
```

**Tomorrow at noon in NY:**

```python
from datetime import date, time, datetime
from zoneinfo import ZoneInfo
ny = ZoneInfo("America/New_York")
tomorrow_noon = datetime.combine(date.today(), time(12, 0), tzinfo=ny) + timedelta(days=1)
```

**Add one month (calendar-aware):**

```python
from dateutil.relativedelta import relativedelta
date(2024, 1, 31) + relativedelta(months=1)        # 2024-02-29
```

**Seconds since epoch (Unix timestamp):**

```python
int(datetime.now(timezone.utc).timestamp())
```

**Format human-readable:**

```python
datetime.now().strftime("%Y-%m-%d %H:%M")
```

**Days between two dates:**

```python
delta = (date(2024, 12, 31) - date(2024, 1, 1)).days
```

---

## HTTP & APIs

**GET JSON:**

```python
import httpx
r = httpx.get("https://api.example.com/items", timeout=30)
r.raise_for_status()
items = r.json()
```

**POST JSON:**

```python
r = httpx.post("https://api.example.com/orders", json={"id": 1, "qty": 100})
```

**With auth:**

```python
r = httpx.get(url, headers={"Authorization": f"Bearer {token}"})
```

**Download a big file (streaming):**

```python
with httpx.stream("GET", url) as response:
    with open("file.bin", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
```

**Async parallel fetches:**

```python
import asyncio, httpx
async def fetch_many(urls):
    async with httpx.AsyncClient(timeout=30) as c:
        return await asyncio.gather(*(c.get(u) for u in urls))
results = asyncio.run(fetch_many(["https://...", "https://..."]))
```

**Retry on failure:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def fetch(url): return httpx.get(url).json()
```

**URL building:**

```python
from urllib.parse import urlencode, urljoin
url = urljoin("https://api.example.com/", "items") + "?" + urlencode({"q": "x"})
```

---

## JSON, YAML, TOML

**Read JSON:**

```python
import json
data = json.loads(Path("data.json").read_text())
```

**Write pretty JSON:**

```python
Path("out.json").write_text(json.dumps(data, indent=2, sort_keys=True))
```

**JSON with datetime support:**

```python
def default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError
json.dumps(obj, default=default)
```

**Read YAML:**

```python
import yaml
data = yaml.safe_load(Path("config.yaml").read_text())
```

**Read TOML (Python 3.11+):**

```python
import tomllib
data = tomllib.loads(Path("pyproject.toml").read_text())
```

**JSONL (newline-delimited JSON):**

```python
with open("data.jsonl") as f:
    for line in f:
        record = json.loads(line)
```

---

## CSV & Excel

**Read CSV to list of dicts:**

```python
import csv
with open("data.csv", newline="") as f:
    rows = list(csv.DictReader(f))
```

**Write CSV from list of dicts:**

```python
with open("out.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
```

**Pandas CSV with explicit types:**

```python
import pandas as pd
df = pd.read_csv("data.csv", dtype={"id": "Int64", "value": float}, parse_dates=["date"])
```

**Stream big CSV in chunks:**

```python
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    process(chunk)
```

**Read multi-sheet Excel:**

```python
sheets = pd.read_excel("workbook.xlsx", sheet_name=None)   # dict of {name: df}
```

**Write to Excel with formatting:**

```python
with pd.ExcelWriter("out.xlsx", engine="openpyxl") as xw:
    df.to_excel(xw, sheet_name="Data", index=False)
```

---

## Databases

**SQLite from scratch:**

```python
import sqlite3
with sqlite3.connect("local.db") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER, value TEXT)")
    conn.execute("INSERT INTO items VALUES (?, ?)", (1, "a"))
```

**SQLAlchemy engine:**

```python
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@host/db", pool_pre_ping=True)
```

**Read SQL into DataFrame:**

```python
df = pd.read_sql("SELECT * FROM trades WHERE date >= '2024-01-01'", engine)
```

**Write DataFrame to table:**

```python
df.to_sql("trades", engine, if_exists="append", index=False, method="multi", chunksize=10_000)
```

**Idempotent upsert (Postgres):**

```python
from sqlalchemy import text
with engine.begin() as conn:
    conn.execute(text("""
        INSERT INTO positions (symbol, qty) VALUES (:sym, :q)
        ON CONFLICT (symbol) DO UPDATE SET qty = EXCLUDED.qty
    """), {"sym": "SPY", "q": 100})
```

---

## Concurrency

**Thread pool:**

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(fn, items))
```

**Process pool (CPU-bound):**

```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as pool:
    results = list(pool.map(heavy_fn, items))
```

**Asyncio task group:**

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        for x in items:
            tg.create_task(do_work(x))
```

**Rate-limit concurrent ops:**

```python
sem = asyncio.Semaphore(8)
async def bounded(item):
    async with sem:
        return await work(item)
```

---

## Subprocesses & shell

**Run and capture:**

```python
import subprocess
r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
print(r.stdout.strip())
```

**With timeout:**

```python
subprocess.run(["sleep", "10"], timeout=5)
```

**Stream output:**

```python
proc = subprocess.Popen(["python", "long.py"], stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    print(line, end="")
```

**Check if a binary exists:**

```python
import shutil
exists = shutil.which("git") is not None
```

---

## Environment & config

**Read env var with default + cast:**

```python
import os
port = int(os.environ.get("PORT", "8000"))
debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
```

**Pydantic settings:**

```python
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    api_key: str
    log_level: str = "INFO"
settings = Settings()
```

**Load `.env`:**

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Logging

**Basic structured logging:**

```python
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)
log.info("started")
```

**JSON logs:**

```python
import structlog
structlog.configure(processors=[structlog.processors.TimeStamper("iso"),
                                  structlog.processors.JSONRenderer()])
log = structlog.get_logger()
log.info("event", user_id=42)
```

**Log exception with traceback:**

```python
try:
    risky()
except Exception:
    log.exception("risky failed")
```

---

## Crypto & hashing

**SHA-256:**

```python
import hashlib
hashlib.sha256(b"data").hexdigest()
```

**Random URL-safe token:**

```python
import secrets
secrets.token_urlsafe(32)
```

**HMAC signature (e.g., webhook):**

```python
import hmac, hashlib
hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
```

**Constant-time string compare:**

```python
hmac.compare_digest(a, b)
```

**Generate UUID:**

```python
import uuid
str(uuid.uuid4())
```

---

## Numeric & arrays

**NumPy where (vectorised if-else):**

```python
import numpy as np
np.where(arr > 0, "pos", "neg")
```

**Vectorised mask:**

```python
arr[arr > 0] = 0
```

**Rolling window stats:**

```python
import pandas as pd
s.rolling(20).mean()
s.ewm(span=20).std()
```

**Cumulative max:**

```python
np.maximum.accumulate(arr)
```

**Top-K without full sort:**

```python
top10_idx = np.argpartition(arr, -10)[-10:]
top10 = arr[top10_idx]
```

---

## Pandas one-liners

**Read parquet:**

```python
df = pd.read_parquet("data.parquet")
```

**Reset multi-index to columns:**

```python
df = df.reset_index()
```

**Pivot:**

```python
wide = df.pivot(index="date", columns="symbol", values="close")
```

**Group by + multiple aggs:**

```python
df.groupby("symbol").agg(mean=("ret", "mean"), std=("ret", "std"), n=("ret", "size"))
```

**Cross-sectional rank per day:**

```python
df["rank"] = df.groupby("date")["signal"].rank(pct=True)
```

**Forward-shift (no leakage):**

```python
df["signal_next"] = df["signal"].shift(-1)         # for labels, not for trading
```

**Fill NaN by group:**

```python
df["x"] = df.groupby("symbol")["x"].ffill()
```

---

## Errors & retries

**Retry with backoff:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def fetch():
    return httpx.get(url).raise_for_status()
```

**Suppress specific exceptions:**

```python
import contextlib
with contextlib.suppress(FileNotFoundError):
    Path("maybe.txt").unlink()
```

**Capture and re-raise with context:**

```python
try:
    parse(blob)
except Exception as e:
    raise ValueError(f"failed to parse: {blob[:50]}") from e
```

---

## System & OS

**Get current working directory:**

```python
import os
os.getcwd()
```

**Get hostname:**

```python
import socket
socket.gethostname()
```

**Get cpu count:**

```python
os.cpu_count()
```

**Get memory usage of current process:**

```python
import psutil
psutil.Process().memory_info().rss / 1e6     # MB
```

**Disk free space:**

```python
import shutil
shutil.disk_usage("/").free / 1e9             # GB
```

**Exit with code:**

```python
import sys
sys.exit(2)         # non-zero = failure
```

---

## End

These are the patterns that compose into real systems. If something isn't here that you find yourself googling repeatedly, it probably belongs in a follow-up commit to this cookbook.

Back to the **[course landing](../index.md)**.
