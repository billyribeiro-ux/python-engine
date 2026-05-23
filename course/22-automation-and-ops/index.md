# Module 22 — Practical Automation & Production Engineering

Most of the course is about trading. This module is about everything else: the file handling, scripting, scraping, database work, scheduling, pipeline construction, repair scripts, AI-assisted workflows, backend services, and enterprise patterns that a senior engineer ships every week — for trading and beyond.

If you take the rest of the course in isolation, you'll know how to build models and strategies. If you take this module too, you'll know how to **operate** them in production: the data flows that keep them fed, the scripts that fix them when they break, the pipelines that aggregate them, and the systems that monitor them.

The content here is general-purpose Python production engineering. None of it is trading-specific. All of it is what L7+ engineers reach for daily.

Pages:

1. **[File handling at scale](01-file-handling.md)** — `pathlib`, glob patterns, atomic writes, file locks, archives, checksums, directory watching.
2. **[CSV, Excel, and tabular interchange](02-csv-and-excel.md)** — `csv` module proper use, pandas Excel I/O, openpyxl formatting, large-file streaming, dialect handling.
3. **[Database scripting](03-databases.md)** — `sqlite3`, SQLAlchemy core + ORM, `psycopg`, transactions, idempotent upserts, migrations with Alembic.
4. **[Web scraping done right](04-web-scraping.md)** — `httpx`, BeautifulSoup, `lxml`, Playwright for JavaScript, rate limiting, robots.txt, retries, caching.
5. **[CLI tools and shell scripting in Python](05-cli-tools.md)** — `argparse`, `click`, `typer`, `subprocess`, exit codes, env vars, signal handling.
6. **[Scheduling — cron, systemd, APScheduler](06-scheduling.md)** — when to use which, anti-patterns, idempotent jobs, leader election.
7. **[Data pipelines and ETL](07-data-pipelines.md)** — idempotency, retries, batch + incremental, Dagster/Prefect lite, Makefile-as-pipeline.
8. **[Admin tooling and remote ops](08-admin-tooling.md)** — `paramiko`, `fabric`, `pyinfra`, `psutil`, structured output for automation chains.
9. **[Migrations and repair scripts](09-migrations-and-repair.md)** — schema migrations, data backfills, idempotent repairs, dry-run patterns, audit trails.
10. **[AI workflows in production](10-ai-workflows.md)** — Anthropic / OpenAI batch processing, prompt engineering for ops, embeddings + vector stores, RAG basics, cost control.
11. **[Backend systems with FastAPI](11-backend-systems.md)** — internal APIs, async services, webhooks, background tasks, dependency injection.
12. **[Enterprise production engineering](12-production-engineering.md)** — secrets, configuration layering, observability, deployment patterns, on-call hygiene.

Read sequentially if you're new to operational engineering. Skim and dip if you're already a working SRE / platform engineer.

Start with **[File handling at scale](01-file-handling.md)**.
