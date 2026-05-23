# Course conventions

A short list of things the course does the same way every time, so you can read fast without re-deriving the rules in each chapter.

## Code style

- **Black** with line length 100. **Ruff** with sensible defaults.
- **Type hints everywhere.** If a function takes a `pd.DataFrame`, it says so. Generic shapes are documented in the docstring.
- **Imports** are at the top of every code block. You can copy any single block standalone and it'll run.
- **No `from x import *`.** Ever.
- **`__all__`** is set on every module that's meant to be imported from.
- **f-strings.** Always.

## Time and timezones

- All timestamps are **tz-aware UTC**. If a vendor returns naive timestamps, the adapter localises them to the exchange tz and then converts to UTC.
- For US equities, `America/New_York` is the canonical exchange tz. We never use `pytz` directly; `zoneinfo` (stdlib, since 3.9) is the way.

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ny = ZoneInfo("America/New_York")
open_ny = datetime(2024, 6, 3, 9, 30, tzinfo=ny)
open_utc = open_ny.astimezone(timezone.utc)
```

## Prices, returns, and log-returns

- **Price** is `close` unless otherwise stated.
- **Return** means simple return: $r_t = P_t / P_{t-1} - 1$.
- **Log return** is $\ell_t = \log(P_t / P_{t-1})$. We use log returns for statistical work (additive over time, closer to Gaussian for short horizons).
- Returns are computed as `prices.pct_change()` or `np.log(prices).diff()` — never by hand in a loop.

## Vectors first, loops never

If you find yourself writing `for t in range(len(df)):` in this course, stop. The exception is **event-driven backtests** (Module 9), where the loop is the whole point. Everywhere else, NumPy or pandas vectorised operations are the rule.

## "Production" and "Frontier" markers

Inside the strategy and scanner modules, sections are tagged:

!!! success "Production"
    Ships at real funds. Use it.

!!! abstract "Frontier"
    Research-grade, occasionally money-making, requires care.

When a chapter has both flavours of the same idea, it shows the production version first.

## "Never trust an unverified backtest"

Every backtest in this course is run with:

- A **train/validation/test** split with a hard time gap (no leakage).
- **Purged k-fold** for hyperparameter selection (Module 9 covers why simple k-fold leaks in time series).
- **Realistic slippage** — at minimum, a linear cost as a function of bar volume.
- **Honest reporting** — Sharpe, max drawdown, Calmar, **deflated Sharpe ratio**, hit rate, turnover, capacity estimate.

A "great" backtest with none of those is a great way to lose money quickly. The course will not show you one without all of them.

## Random seeds and reproducibility

Every randomised example fixes a seed. The seed is always at the top of the snippet:

```python
import numpy as np
rng = np.random.default_rng(42)
```

We **never use `np.random.seed`** at module scope — that's a global side effect. The `default_rng(seed)` pattern is the modern way and is thread-safe.

## A note on the libraries we don't use

- We do not use the `random` standard library module for any numerical work. NumPy's `default_rng` is faster and produces higher-quality streams.
- We do not use `pandas.DataFrame.append` (deprecated). Build a list of frames and `pd.concat` at the end.
- We do not use `inplace=True`. It does not save memory in modern pandas and it confuses static analysis. Return the new frame.
- We do not use `df.iterrows()` for anything performance-sensitive. Vectorise, `apply`, or `itertuples` — in that order of preference.

That's it. Continue to **[Module 1 — Python Foundations](../01-python-foundations/index.md)**.
