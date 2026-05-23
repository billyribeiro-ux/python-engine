# Environment setup

You need three things: a recent Python, a virtualenv, and the package installed in editable mode. Total time: about three minutes.

## Python version

This course targets **Python 3.11 or newer**. We use a handful of features that landed across 3.10–3.12 (structural pattern matching, `Self`, `tomllib`, faster CPython, better tracebacks).

```bash
python3 --version
# Python 3.11.x or 3.12.x or 3.13.x is fine
```

If you're on macOS, the system Python is too old. Install via [pyenv](https://github.com/pyenv/pyenv) or `brew install python@3.12`. On Linux, use your distro's Python or `pyenv`. On Windows, use the official installer or WSL.

## Virtualenv

Always work inside a virtualenv. The course assumes one is active.

```bash
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
python -m pip install -U pip
```

!!! tip "Why a venv, really"
    The dependency graph for ML + quant work is large and version-sensitive. `numpy`, `scipy`, `torch`, `pandas`, `arch`, `statsmodels` all couple to each other. A venv per project is the only way to stay sane.

## Install

The course's package, `engine`, ships with **optional extras** so you only install what you need.

```bash
# Phase 1 (Modules 0-3) needs only:
pip install -e ".[core,docs,dev]"
```

What each extra contains is in `pyproject.toml`, but in short:

| Extra | When you need it |
|---|---|
| `core` | NumPy, pandas, polars, scipy, matplotlib, pyarrow, duckdb |
| `ml`   | scikit-learn, statsmodels, XGBoost, LightGBM, CatBoost, hmmlearn, arch, filterpy, ruptures, MAPIE |
| `dl`   | PyTorch |
| `options` | py_vollib, QuantLib |
| `rl`   | gymnasium, stable-baselines3 |
| `docs` | mkdocs-material, mkdocs-with-pdf, pymdown-extensions |
| `dev`  | pytest, ruff, black, mypy |

You can install multiple at once:

```bash
pip install -e ".[core,ml,dl,options,rl,docs,dev]"
```

That's everything you'll ever need. Total install on a fresh machine: ~2–3 GB on disk (PyTorch is most of it).

## Smoke test

```python
from engine.data import YFinanceFeed
feed = YFinanceFeed()
bars = feed.bars("SPY", "2024-01-02", "2024-01-15", interval="1d")
print(bars)
```

Expected output (your numbers will differ slightly with Yahoo's adjustments):

```text
                                open        high         low       close      volume
timestamp
2024-01-02 00:00:00+00:00  472.160004  473.670013  470.470001  472.649994  ...
2024-01-03 00:00:00+00:00  470.429993  471.190002  468.170013  468.790009  ...
...
```

If you got a DataFrame back, your install works.

## Common install issues

!!! warning "`yfinance` returns an empty DataFrame"
    Yahoo aggressively rate-limits. Wait 30 seconds, try again. If you'll be running many examples in a single session, cache results to Parquet — Module 5 shows you how. For serious work, get a real data feed.

!!! warning "`torch` install is slow / huge"
    If you don't need deep learning yet, skip the `dl` extra. You can add it later with `pip install -e ".[dl]"`. On Apple Silicon, PyTorch ships with MPS (Metal) support by default.

!!! warning "`QuantLib` build errors"
    Use `pip install QuantLib` (the precompiled wheel). The `QuantLib-Python` package is older and tries to build from source.

!!! warning "`mkdocs-with-pdf` fails on first PDF build"
    It depends on a headless Chrome/Chromium under the hood. Install `chromium` via your package manager (`brew install chromium`, `apt install chromium`) and re-run `make pdf`. If it still fails, you can fall back to `pandoc course/**/*.md -o book.pdf` — uglier but reliable.

## Editor setup (optional but worth it)

- **VS Code** — install the Python, Ruff, and Even Better TOML extensions. Set the interpreter to `.venv/bin/python`.
- **PyCharm** — point the project interpreter at the venv. Enable Ruff under Settings → Tools → Ruff.

Continue to **[The universal data adapter](data-adapter.md)**.
