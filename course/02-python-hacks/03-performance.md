# Performance and profiling

The first rule of optimisation is: **don't, until you've measured**. The second rule is: **even when you measure, look in the right place**. This chapter shows you how to do both.

## The hierarchy of speed-up

Roughly in order of effort and payoff:

1. **Use the right algorithm.** O(n) beats O(n²) every time. Sorting once and binary-searching beats nested loops.
2. **Use the right data structure.** Dict lookup is O(1); list `.index()` is O(n). Set membership is O(1); list `in` is O(n).
3. **Use the right library.** Pandas/Polars vectorised ops beat Python loops by 50–500×. NumPy beats Python loops by 100–1000×.
4. **Use the GPU.** PyTorch / JAX beats CPU NumPy by another 10–100× on the right kinds of problems.
5. **Use a JIT.** Numba (`@njit`) or Cython for hot numeric loops.
6. **Use C.** Write the inner loop in C (`ctypes`, `cffi`) or Rust (`pyo3`) and call it.

Don't skip steps. Every optimisation past step 3 multiplies the maintenance cost.

## `timeit` for microbenchmarks

```python
import timeit

# Single-line
print(timeit.timeit("sum(range(1000))", number=10000))

# Multi-line
setup = "xs = list(range(10000))"
stmt = "[x*2 for x in xs]"
print(timeit.timeit(stmt, setup=setup, number=1000))
```

In a notebook: `%timeit expr`. `%timeit -n 100 -r 5 expr` for control over runs/repeats.

!!! warning "Microbenchmark traps"
    JIT warmup, branch prediction, CPU frequency scaling, and "cold cache vs warm cache" all distort microbenchmarks. Treat results as order-of-magnitude indicators, not exact ratios. Run the *whole pipeline* to verify a real-world win.

## `cProfile` — function-level profiling

```python
import cProfile, pstats

def run_pipeline():
    ...

cProfile.run("run_pipeline()", "out.prof")

stats = pstats.Stats("out.prof").sort_stats("cumulative")
stats.print_stats(30)
```

You'll see a table of function calls ranked by cumulative time. Two columns matter:

- **`tottime`** — time *inside* this function, excluding child calls.
- **`cumtime`** — time *including* child calls.

The function at the top by `cumtime` is where your latency lives. The function at the top by `tottime` is what's actually slow on its own.

For a nicer UI, install [`snakeviz`](https://jiffyclub.github.io/snakeviz/) and `snakeviz out.prof` — interactive flame graph in the browser.

## `py-spy` — the production profiler

`cProfile` adds overhead. `py-spy` is a sampling profiler that attaches to a running process **with zero source changes**. Great for live systems.

```bash
# Install
pip install py-spy

# Profile a running PID
py-spy top --pid 12345

# Record a flame graph
py-spy record -o flame.svg --pid 12345 --duration 30
```

If a strategy is mysteriously slow in production, `py-spy top` for 10 seconds usually tells you exactly which function is burning CPU.

## `scalene` — CPU *and* memory in one tool

[`scalene`](https://github.com/plasma-umass/scalene) profiles CPU + memory + GPU together, separates native vs Python time, and gives suggestions. Often the right first thing to try for a complete picture.

```bash
pip install scalene
scalene your_script.py
```

The output highlights lines where memory is allocated heavily, which is the secret killer in pandas-heavy code.

## `tracemalloc` — where did all the memory go?

`tracemalloc` is in the standard library and snapshots Python allocations.

```python
import tracemalloc

tracemalloc.start()

# ... do work ...

snap = tracemalloc.take_snapshot()
top = snap.statistics("lineno")
for stat in top[:10]:
    print(stat)
```

Run it twice — once at the start, once at the end — and `snap2.compare_to(snap1, "lineno")` shows you exactly which lines added memory.

```python
snap1 = tracemalloc.take_snapshot()
do_thing()
snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, "lineno")[:10]:
    print(stat)
```

This is how you find the "loop that grows a dict and never frees it" bug that crashes long-running services overnight.

## A real workflow

You suspect a backtest is slow. Walk:

```bash
# 1. Quick overview — where is the time?
python -c "import cProfile; cProfile.run('import backtest_script', 'p.prof')"
snakeviz p.prof

# 2. Suspect a specific function — measure tightly
python -c "import timeit; print(timeit.timeit(...))"

# 3. Confirm a memory hypothesis
scalene backtest_script.py
```

You will almost always find one of these:

- A pandas `groupby.apply` you didn't realise was running Python per group.
- A `.iterrows()` somewhere.
- A `pd.DataFrame.append` in a loop.
- A repeated `df[df.col == something]` that should have been a vectorised mask.
- An accidental Cartesian join.

Each of these has a 10× fix.

## Vectorisation patterns (preview)

The single biggest speedup is replacing Python loops with NumPy/pandas vectorised operations. Module 3 covers this in depth. The teaser:

```python
import numpy as np

# Naive — 200 ms on a million prices
returns = []
for i in range(1, len(prices)):
    returns.append(prices[i] / prices[i-1] - 1)

# Vectorised — 1.2 ms on the same data
returns = prices[1:] / prices[:-1] - 1
```

160× speedup, no third-party libraries, more readable. The rest of the course teaches you to spot opportunities like this everywhere.

## Numba — JIT for the hot inner loops

When you've vectorised everything and still need more speed, [`numba`](https://numba.pydata.org/) is the first thing to reach for. It compiles Python functions decorated with `@njit` to machine code via LLVM:

```python
import numba
import numpy as np

@numba.njit
def realised_vol(returns: np.ndarray, lookback: int) -> np.ndarray:
    n = len(returns)
    out = np.empty(n)
    for i in range(n):
        if i < lookback:
            out[i] = np.nan
        else:
            window = returns[i - lookback + 1: i + 1]
            out[i] = np.sqrt((window * window).sum() / lookback)
    return out
```

This Python-looking function runs at C speed after a one-time compile. Restrictions: numba understands a subset of Python (no pandas, no dicts, limited object support). For numeric loops over arrays, it's the right tool. The course's options module uses it for Heston Monte Carlo.

## Cython — when you need PyObject access too

[Cython](https://cython.org/) compiles a typed Python superset to C. Heavier setup than Numba but supports the full Python object model, classes, etc. The right tool when you want to ship a library with optimised hot paths and don't want the user to install LLVM at runtime.

`pandas` and `scikit-learn` are largely Cython under the hood.

## `mypyc` — turn typed Python into a C extension

A nice middle ground: [`mypyc`](https://github.com/mypyc/mypyc) compiles type-annotated Python to a C extension, often giving 2–4× speedup on real code with zero source changes (other than already-typed code). Worth trying once you have a typed codebase. The `black` formatter ships compiled with mypyc.

## Profiling pandas specifically

Pandas slowness is usually one of three things:

1. **Operations on object-dtype columns** (strings, datetimes-as-objects). The fix is `pd.Categorical`, `pd.to_datetime`, `pd.StringDtype`.
2. **Row-wise apply with a Python function.** The fix is a vectorised op, a numba-jitted apply (`Series.apply` doesn't JIT; pandas has `numba_engine="numba"` on some methods), or a switch to polars.
3. **Repeated indexing in a loop.** Reset the index, use `.iloc` on numpy arrays directly, or vectorise.

A 100k-row DataFrame with a row-wise `.apply` Python callback can take seconds. The same data with a vectorised op takes microseconds.

## When NOT to optimise

If a function runs once per backtest and takes 200 ms, you do not need to make it 10 ms. The total saving is sub-second. Spend that hour on:

- A better backtest split (no leakage).
- A more honest slippage model.
- A cleaner data pipeline.

**Premature optimisation isn't just a Knuth quote; it's the most common way to ship buggy code.**

The right order is: measure, look, fix the worst thing, re-measure, stop when "fast enough" for the use case.

## Continuous performance testing

Once you have a fast path you care about, **lock it in** with a benchmark in CI:

```python
# tests/test_perf.py
import time
import pytest

@pytest.mark.benchmark
def test_returns_compute_fast(big_prices):
    t0 = time.perf_counter()
    returns = compute_returns(big_prices)
    dt = time.perf_counter() - t0
    assert dt < 0.5      # regression if it crosses 500 ms
```

The plugin [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) does this properly with statistical comparisons across commits.

Continue to **[Plugin patterns and registries](04-plugins.md)**.
