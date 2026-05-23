# Web scraping done right

A lot of "data unavailable through an API" is sitting on a public web page somewhere. Scraping it is a legitimate engineering tool — when done politely, respectfully of the site's resources, and within the law of your jurisdiction. This chapter is the working stack and the ethical baseline.

## The library matrix

| Need | Use |
|---|---|
| Static HTML | `httpx` + `BeautifulSoup` |
| Faster HTML parsing | `httpx` + `lxml` (with CSS selectors) |
| JavaScript-rendered pages | `Playwright` (headless Chromium) |
| Rate-limited / authenticated APIs | `httpx` + retries via `tenacity` |
| Bulk crawl of many pages | `httpx` with `asyncio` |
| Anti-bot defences | accept defeat, find a real data feed |

## `httpx` — the modern HTTP client

`requests` is fine but old; `httpx` is its modern successor with native async support and the same ergonomics:

```python
import httpx

with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
    response = client.get("https://example.com/api/data")
    response.raise_for_status()
    data = response.json()
```

Settings every production caller should use:

- **`timeout=`** with separate connect and total timeouts; never leave default infinite.
- **`raise_for_status()`** before consuming the response; 4xx/5xx should error explicitly.
- **`Client` (re-use connections)** instead of one-shot `httpx.get(...)`. Saves the TLS handshake on every request.

## Async for bulk fetches

```python
import asyncio
import httpx


async def fetch_one(client, url):
    r = await client.get(url)
    r.raise_for_status()
    return r.text


async def fetch_many(urls, concurrency=10):
    sem = asyncio.Semaphore(concurrency)

    async def bounded(url):
        async with sem:
            return await fetch_one(client, url)

    async with httpx.AsyncClient(timeout=30) as client:
        return await asyncio.gather(*(bounded(u) for u in urls))


results = asyncio.run(fetch_many(urls, concurrency=10))
```

For 1,000 URLs at 200 ms each, sequential is 3 minutes; concurrency=10 brings it to 20 seconds.

## BeautifulSoup + lxml for HTML parsing

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "lxml")

# Find by tag + attributes
title = soup.find("h1", class_="page-title").get_text(strip=True)

# CSS selectors
prices = [el.get_text(strip=True) for el in soup.select("table.prices tr > td.amount")]

# Walk children
for li in soup.select("ul.symbols > li"):
    sym = li.find("a")["data-symbol"]
    process(sym)
```

For CSS-selector-heavy parsing, `lxml` direct is faster:

```python
from lxml import html

tree = html.fromstring(html_text)
prices = tree.cssselect("table.prices tr > td.amount")
print([p.text_content().strip() for p in prices])
```

For XPath:

```python
prices = tree.xpath("//table[@class='prices']/tr/td[@class='amount']/text()")
```

## Playwright — when JavaScript is involved

Many modern sites render content client-side. `httpx` gets you the empty shell; `Playwright` runs an actual headless browser and waits for content to load.

```python
from playwright.sync_api import sync_playwright


def scrape_js_site(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        # Wait for a specific element
        page.wait_for_selector("div.stock-data", timeout=10_000)
        html = page.content()
        browser.close()
    return html
```

Playwright also offers async; for many JS-rendered pages, parallel browsers.

The cost: each Playwright session is ~200 MB RAM and ~2-5 seconds. For 10,000 pages, you need queuing.

## Robots.txt and the ethical baseline

```python
from urllib.robotparser import RobotFileParser


rp = RobotFileParser()
rp.set_url("https://example.com/robots.txt")
rp.read()

if not rp.can_fetch("MyScraperBot/1.0", "https://example.com/path"):
    print("blocked by robots.txt")
    raise SystemExit(1)
```

Robots.txt is non-binding (it's an advisory protocol), but respecting it is the minimum bar of polite scraping. Sites that explicitly disallow scraping are signalling, "we don't want bots here." Find the data elsewhere.

## Rate limiting

A polite scraper limits its own request rate. The simplest pattern:

```python
import asyncio
import time


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1 / rps
        self.next_ok = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = max(0, self.next_ok - now)
            self.next_ok = max(now, self.next_ok) + self.min_interval
        if wait > 0:
            await asyncio.sleep(wait)
```

Cap your scraper at the rate the site can tolerate. For most public sites, 1-2 RPS is polite. For known APIs, check their docs.

## Retries with exponential backoff

Transient HTTP failures happen. `tenacity` handles retry elegantly:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def fetch_with_retry(client, url):
    r = client.get(url)
    r.raise_for_status()
    return r
```

5 attempts; exponential backoff 2s → 4s → 8s → 16s → 30s; only retry on network errors, not on 4xx.

For 429 (Too Many Requests), inspect the `Retry-After` header explicitly:

```python
try:
    r = client.get(url)
    r.raise_for_status()
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 429:
        delay = int(exc.response.headers.get("Retry-After", "60"))
        time.sleep(delay)
        # retry
```

## Caching

Re-running a scraper without re-fetching unchanged pages saves time and is polite. `hishel` is the modern HTTP cache for httpx:

```python
import httpx
import hishel


storage = hishel.FileStorage(base_path=".cache/http")
transport = hishel.CacheTransport(transport=httpx.HTTPTransport(), storage=storage)
with httpx.Client(transport=transport) as client:
    r = client.get("https://example.com/data")
```

Respects HTTP `Cache-Control` headers. For sites that don't set them, force-cache for a fixed TTL.

For static content (CSV dumps, image archives), a content-addressed store on disk works:

```python
def get_cached_or_fetch(url: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    cached = cache_dir / key
    if cached.exists():
        return cached.read_bytes()
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    atomic_write_bytes(cached, response.content)
    return response.content
```

(`atomic_write_bytes` from chapter 1.)

## Headers and user-agent

Always identify your bot honestly:

```python
HEADERS = {
    "User-Agent": "PythonEngineCourse/1.0 (https://example.com; contact@example.com)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
}

client = httpx.Client(headers=HEADERS)
```

A real user agent with a contact link lets sysadmins reach you before they ban you.

## A worked example: a polite scraper

```python
import asyncio
import httpx
from pathlib import Path
import hashlib


CACHE_DIR = Path(".cache/scraper")
HEADERS = {"User-Agent": "PythonEngineCourse/1.0 (contact@example.com)"}
RATE_LIMIT = RateLimiter(rps=1.0)


async def fetch_one(client, url):
    await RATE_LIMIT.acquire()
    key = hashlib.sha256(url.encode()).hexdigest()
    cached = CACHE_DIR / key
    if cached.exists():
        return cached.read_bytes()
    r = await client.get(url, headers=HEADERS)
    r.raise_for_status()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(cached)
    return r.content


async def scrape_all(urls):
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(*(fetch_one(client, u) for u in urls))
```

Rate-limited, cached, polite, robust. The shape of a production scraper.

## Pitfalls

!!! warning "Scraping rendered SPAs without Playwright"
    Many modern sites render content with JavaScript; `httpx` will return an empty shell. If you see no data in the parsed HTML, switch to Playwright before assuming your selector is wrong.

!!! warning "Anti-bot defences"
    Cloudflare, DataDome, Akamai Bot Manager. Defeating them often violates terms-of-service. If you hit serious anti-bot infrastructure, find a different source.

!!! warning "Storing scraped data is not always legal"
    Some sites' terms of service prohibit re-use of scraped data. Check before you build a product on it.

!!! warning "Selector fragility"
    A change to the site's HTML breaks your scraper. Use the most stable selectors (semantic class names, data attributes), not the most specific (`div > div > div > span:nth-child(3)`).

!!! warning "Don't hammer logged-in or paid endpoints**
    If you have an account, the rate-limit math is different. Respect it; otherwise the account gets banned.

## Bottom line

For production web scraping:

- **`httpx`** for HTTP; `Client` for connection re-use.
- **`asyncio` + semaphore** for bulk fetches.
- **`BeautifulSoup` / `lxml`** for HTML; **`Playwright`** for JS.
- **`tenacity`** for retries; respect `Retry-After`.
- **Cache aggressively** (`hishel` or content-addressed disk).
- **Respect robots.txt; honest user agent; bounded rate**.

Continue to **[CLI tools and shell scripting in Python](05-cli-tools.md)**.
