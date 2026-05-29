# Enterprise production engineering

The last chapter of the course. By now you have the trading machinery, the ML stack, the data pipelines, the scanners, the backend services. This chapter ties them together with the **operational discipline** that turns "code that works on my laptop" into "system that runs the business" — secrets, configuration, observability, deployment patterns, and on-call hygiene.

## Configuration

### Layered configuration

Production code reads config from multiple sources, in order of precedence:

1. **Command-line flags** (highest priority — for one-off overrides).
2. **Environment variables** (per-deployment configuration).
3. **Config file** (`.env`, `config.toml`).
4. **Defaults** (lowest priority — in code).

`pydantic-settings` handles this cleanly:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///local.db"
    api_token: str
    log_level: str = "INFO"
    feed_name: str = "yfinance"
    max_position_dollars: float = 100_000.0
    enable_paper_trading: bool = True


settings = Settings()
```

Reads `APP_DATABASE_URL` (or `database_url` from `.env`) and validates the schema. Missing required keys fail at startup, not in production.

### Per-environment configs

For dev/staging/prod, layer config files:

```yaml
# config.yaml
base:
  feed_name: yfinance
  log_level: INFO

development:
  database_url: sqlite:///dev.db
  enable_paper_trading: true

production:
  database_url: postgresql://prod-db.internal/trading
  log_level: WARNING
  enable_paper_trading: false
```

Load the env-specific subtree based on `$ENVIRONMENT`. The base inherits; overrides win.

## Secrets management

**Never commit secrets.** Three production approaches:

1. **Environment variables**, populated from a secrets manager at runtime.
   - AWS Secrets Manager / Parameter Store
   - GCP Secret Manager
   - HashiCorp Vault
   - Doppler / 1Password CLI

2. **Encrypted file commits** via `sops` (Mozilla) or `git-crypt`. The repo contains encrypted secrets; only authorised devs can decrypt.

3. **Workload identity** — for cloud-deployed services, attach an IAM role / service account that has permission to read secrets; no static credentials anywhere.

The pattern for most modern shops: **Workload identity** + **Secrets manager** + **env vars in the process**. No `.env` file in production; the secrets manager fills env vars at container start.

## Observability — the production stack

Module 21 chapter 5 covered the basics. For enterprise scale:

- **Metrics**: Prometheus + Grafana (or DataDog / New Relic / Honeycomb).
- **Logs**: Centralised aggregation — Loki, Elasticsearch, CloudWatch Logs, Datadog.
- **Traces**: Jaeger or Tempo for distributed tracing across services.
- **Errors**: Sentry for unhandled-exception aggregation with context.

For a real fund, you want all four. For a small shop starting out, metrics + structured logs are enough.

### Three layers of alerts

1. **Page** (wakes someone up): system down, daily loss > limit, kill switch fired.
2. **Notify** (Slack/email; review within hours): degraded performance, single-strategy failure, partial data feed loss.
3. **Inform** (dashboards, no notification): daily P&L, capacity utilisation, slowly-changing metrics.

Each alert needs a runbook — a documented step-by-step response. "Without a runbook, an alert is just a notification." Build the runbook before going on call.

## Deployment patterns

### Blue/green

Two identical environments — "blue" (live) and "green" (new code). Deploy to green, validate, then switch traffic. If something's wrong, switch back instantly.

Costs 2× infrastructure during deploys; gives instant rollback.

### Canary

Deploy new code to 5% of traffic (or 1 server); watch metrics; gradually ramp up. If errors spike, roll back before full deploy.

Standard for high-availability services.

### Rolling

Replace instances one at a time; new code propagates over minutes. Kubernetes default.

For most services this is fine. For trading systems where partial state across versions causes problems (e.g., serialisation format change), blue/green is safer.

### Immutable infrastructure

The deployed unit is a container image, not "patches to a server." Servers are cattle, not pets. Rolling a deploy means launching new container instances and terminating old ones.

For modern shops: Docker + Kubernetes (or ECS, Cloud Run, Nomad) is the default. Don't ssh into a production box to "fix" something — fix the source, redeploy.

## Database in production

Three operational practices that matter:

1. **Backups + recovery drills.** Backups are easy; recovery is rare and high-stakes. Drill it quarterly.
2. **Read replicas** for analytics queries; don't run a `SELECT * FROM trades` on the primary.
3. **Connection pooling** at the application level (PgBouncer for Postgres) so a runaway service can't exhaust DB connections.

For the trading database specifically: hourly point-in-time backups; full daily backups; off-site copies; tested restore process. A trading database that's lost without recovery is a fund-ending event.

## On-call hygiene

A small shop has 1-2 people on call. Some discipline:

- **Rotate explicitly.** "Whoever's around" leads to burnout for the most responsible person.
- **Document the rotation publicly.** PagerDuty / Opsgenie / a shared calendar.
- **Hand off shift state.** End-of-shift summary: what fired, what's pending, what to watch.
- **Postmortem every page.** Page → ticket → fix or runbook update or alert tweak. Pages that fire repeatedly are a process bug.
- **Quiet hours.** Mid-day pages are tolerable; 4am pages destroy productivity. Reserve nighttime pages for true emergencies.

## Code review and deploy gates

Bare minimum:

- **PR review** before merge to main.
- **CI must pass** (Module 22 ch6: tests, lint, build).
- **Manual approval** for production deploys.

For high-stakes systems, add:

- **Deploy windows** — no deploys on Fridays, no deploys mid-market.
- **Canary auto-rollback** on error-rate spike.
- **Two-person sign-off** for changes touching money or production data.

## Capacity planning

You'll grow. Two metrics to track over time:

1. **Database size and growth rate.** Plan for partitioning and archival before tables hit 1B rows.
2. **Compute spend per dollar of AUM.** As AUM grows, the cost should grow sub-linearly (you amortise fixed costs). If it's growing super-linearly, you're scaling the wrong way.

Quarterly: project growth 12-18 months out; identify what's about to break; budget for upgrades.

## Incident response

A simple framework: **DAFR**.

- **Detect** — alerts fire.
- **Assess** — what's the actual scope; who/what is impacted.
- **Fix** — make the bleeding stop. Maybe roll back; maybe shut down a feature; maybe scale up.
- **Review** — postmortem after the dust settles (Module 21 chapter 6).

The temptation in the middle is to skip Assess and jump to Fix. Don't. Wrong fixes prolong the incident; right fixes resolve it in one move. Take five minutes to understand before acting (unless the bleeding is dollars-per-second).

## The mature operational stack — in one picture

```
                      ┌─ pydantic-settings + secrets manager
                      │
                      ▼
        Code (FastAPI, scripts, services)
                      │
                      ▼
        Container image (Docker)
                      │
                      ▼
       Orchestrator (K8s / ECS / Cloud Run)
              │       │       │
              ▼       ▼       ▼
        Postgres   Redis   Object store (S3)
              │       │       │
              ▼       ▼       ▼
        Prometheus + Loki + Sentry + Jaeger
                      │
                      ▼
                  Grafana + PagerDuty
```

That's roughly the shape of a modern small-fund tech stack. Each piece is replaceable; the contract between them is what matters.

## The closing principle

> Production engineering is the discipline of making your system **boring** — predictable, observable, and recoverable. Boring is good. Boring is what lets you sleep through the night and trust that your strategy is doing what you designed.

Every chapter in this module exists in service of boring. File handling that doesn't lose data. Pipelines that resume after failure. Scanners that fire reliably. Backends that don't fall over. Deployments that don't surprise you. Postmortems that prevent recurrence.

Get boring right, and the interesting parts (the strategies, the models, the frontier methods) get to actually run.

## Onward

You've now seen the foundations (Python + NumPy + pandas), the data layer, the quant core, the modern ML/DL/RL stack, the options machinery, the strategy modules, the scanners, execution + microstructure, risk + portfolio, deployment, and the production-engineering discipline that holds it together. One chapter remains in this module — the modern agentic-AI layer that increasingly sits on top of all of it.

Continue to **[Agentic AI, MCP, and extended thinking](13-agentic-ai.md)**.
