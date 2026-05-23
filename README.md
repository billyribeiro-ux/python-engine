# The Python Engine

A distinguished-engineer course on Python, machine learning, and trading — covering both stocks and options, plus the operational engineering that turns research into production.

Two parallel tracks for trading:

- **Production track** — methods that actually ship at quant funds today.
- **Frontier track** — the exotic stuff: neural SDEs, Hawkes processes, topological data analysis, transfer entropy, deep hedging, RL execution, GNNs on correlation graphs.

Plus a dedicated **operational engineering** module (Module 22) covering automation, scripting, file handling, CSV/Excel, database scripting, web scraping, CLI tools, scheduling, ETL pipelines, admin tooling, migrations & repair scripts, AI workflows, FastAPI backends, and enterprise production patterns.

The course is delivered as **interactive HTML** (with one-click copy on every code block, search, dark mode) and as a **paginated PDF book** for offline reading. The source is plain Markdown so it's also fully readable on GitHub.

**Status: complete.** 23 modules + appendix, **~157k words across 167 pages**, 33 passing tests, CI pipeline + pre-commit hooks. See [Phase rollout](#phase-rollout) below for what shipped when.

---

## Quick start

```bash
git clone https://github.com/billyribeiro-ux/python-engine.git
cd python-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[core,docs,dev]"
make serve      # open http://127.0.0.1:8000
```

To build the static site:

```bash
make site       # writes ./site
```

To build the PDF:

```bash
make pdf        # writes ./site/pdf/python-engine-course.pdf
```

Run the tests (skips network tests when offline):

```bash
make test
```

Operational scripts:

```bash
make smoke         # end-to-end integration smoke (synthetic + real-data modes)
make scanners      # daily scanner report against a default ETF universe
make precommit     # install pre-commit hooks (ruff, black, file checks)
```

## CI

`.github/workflows/ci.yml` runs on every push:

- `ruff check` (lint)
- `black --check` (formatting)
- `pytest -m "not network"` (unit + integration)
- `mkdocs build --strict` (docs build, zero warnings)
- `python scripts/end_to_end_smoke.py --no-network` (synthetic end-to-end)
- uploads the built site as an artifact

## What's in the box

```
course/        all teaching content (Markdown)
engine/        importable Python library
  data/        universal market-data adapter
  features/    feature engineering (later phases)
  models/      HMM, Kalman, GBMs, NNs, RL (later phases)
  backtest/    vectorised + event-driven backtester (Module 9)
  options/     Black-Scholes, Heston, SABR, vol surface (Modules 14-15)
  scanners/    scanner framework + concrete scanners (Module 18)
  risk/        CVaR, drawdown, HRP, vol targeting (Module 20)
scripts/       build_assets.py — regenerates course charts
tests/         pytest
notebooks/     optional companion notebooks
```

## The universal data adapter

Every example reads data through one tiny interface so you can swap vendors with a single import:

```python
from engine.data import YFinanceFeed   # free, anonymous, perfect for the course
feed = YFinanceFeed()
spy  = feed.bars("SPY", "2024-01-01", "2024-06-01", interval="1d")
chain = feed.option_chain("SPY")
```

Want to use Polygon, Alpaca, Tradier, or IBKR instead? Same code — just change one line, or set `PYTHON_ENGINE_FEED=polygon` and call `default_feed()`. The stub adapters point at the chapter in the course where each integration is wired up.

## Phase rollout

The course is large enough that we build it in waves. Each wave is independently useful, committed, and pushed before the next begins.

| Phase | Modules | Status |
|------:|---------|--------|
| 1 | Scaffold + Module 0 Orientation, Module 1 Python Foundations, Module 2 Python Hacks, Module 3 NumPy Mastery | **shipped** |
| 2 | Module 4 pandas + polars, Module 5 Data Engineering, Module 6 Stats & Probability, Module 7 Time Series | **shipped** |
| 3 | Module 8 Classical Quant, Module 9 Backtesting, Module 10 ML Foundations | **shipped** |
| 4 | Module 11 Modern ML, Module 12 Deep Learning for Time Series | **shipped** |
| 5 | Module 13 RL, Module 14 Options Foundations, Module 15 Vol Surface | **shipped** |
| 6 | Module 16 Production Strategies, Module 17 Frontier Strategies, Module 18 Scanners | **shipped** |
| 7 | Module 19 Execution & Microstructure, Module 20 Risk & Portfolio, Module 21 Deployment, Appendix | **shipped** |
| 8 | Module 22 Automation & Production Engineering, CI/CD, pre-commit hooks, integration smoke, scanner runner | **shipped** |

## License

MIT.
