# ndarray internals

An `ndarray` is a thin Python object wrapping a contiguous block of memory plus a small bag of metadata. Understanding that mental model is what separates "NumPy user" from "NumPy thinker".

## The five fields that matter

Every `ndarray` has, internally:

1. **`data`** — a pointer to a flat block of bytes.
2. **`dtype`** — what each element is (`int64`, `float64`, `complex128`, ...).
3. **`shape`** — a tuple of dimensions.
4. **`strides`** — a tuple of *byte offsets* per axis: "to move one step along axis `i`, jump `strides[i]` bytes."
5. **`flags`** — booleans about contiguity, write-ability, alignment.

```python
import numpy as np

a = np.arange(12, dtype=np.int64).reshape(3, 4)

print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(a.dtype, a.shape, a.strides, a.flags["C_CONTIGUOUS"])
# int64 (3, 4) (32, 8) True
```

`int64` is 8 bytes. To move one element along the last axis (axis 1), the stride is 8 bytes. To move one element along axis 0, the stride is `4 * 8 = 32` bytes — because each "row" has 4 elements of 8 bytes.

## Views vs copies

This is the most important sentence in the chapter: **most NumPy operations return a view, not a copy**. A view shares memory with the original.

```python
a = np.arange(12).reshape(3, 4)
b = a[:, 1:3]            # slice → view
b[0, 0] = 99
print(a)
# [[ 0 99  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

Modifying `b` modified `a`. That's almost always the right behaviour for performance (NumPy never copies until it has to), but it bites the day you didn't expect it.

How to tell:

- `b.base is a` — yes, `b` is a view of `a`.
- `b.flags["OWNDATA"]` — False if it's a view.
- `np.shares_memory(a, b)` — definitive.

When you need a copy, ask for it: `b = a[:, 1:3].copy()`.

### When you get a view vs a copy

- **Basic slicing** (`a[1:3, ::2]`) → view.
- **Reshaping** that doesn't require re-laying-out memory → view.
- **Transposing** (`a.T`) → view (with reversed strides).
- **Fancy indexing** (`a[[0, 2]]`, boolean mask) → copy.
- **Arithmetic** (`a + b`) → copy (new allocation).

```python
a = np.arange(12).reshape(3, 4)
view = a[:, ::2]
fancy = a[[0, 2]]
print(view.base is a, fancy.base is a)
# True False
```

## C vs Fortran order

By default arrays are **row-major** (C order): rows are contiguous in memory. Some scientific code wants **column-major** (Fortran order) because the equations are linear-algebra-natural that way.

```python
c = np.arange(12).reshape(3, 4)            # C order by default
f = np.asfortranarray(c)                    # same data, F order
print(c.strides, f.strides)
# (32, 8) (8, 24)
```

`f` has stride 8 along axis 0 (moving down a column is one element of 8 bytes) and 24 along axis 1 (moving across to the next column is 3 rows × 8 bytes).

When does it matter?

- **Linear algebra** — `np.linalg`, BLAS calls. F-order can be measurably faster for matrix products in some shapes, because BLAS expects column-major.
- **PyTorch interop** — copies often happen between C-NumPy and F-PyTorch tensors and vice versa.
- **Memory layout-sensitive cache effects** — large iterations along the wrong axis are dramatically slower.

For the vast majority of trading code, leave it as default C order.

## dtype — the slow path is hidden here

```python
# int64 → ~75 ns per addition in pure NumPy
a = np.arange(1_000_000, dtype=np.int64)

# object dtype → ~300x slower because every op is a PyObject call
b = np.arange(1_000_000, dtype=object)
```

If your array's dtype is `object`, you've lost NumPy's main superpower. Common culprits:

- A column of Python `decimal.Decimal`.
- A pandas column of mixed types.
- A column you converted from JSON without coercing types.

The fix is always to convert to a native dtype:

```python
prices_float = np.array(prices, dtype=np.float64)
```

For dates: `np.datetime64` (or `pd.Timestamp`, which is a thin wrapper). For categorical strings: `pd.Categorical` (in pandas) or `pd.StringDtype()`. Anything but `object`.

## A worked example: simulating tick→bar bucketing layout

Suppose you have 10 million ticks and want to bucket them into 1-minute bars. You need timestamps, prices, sizes. If you stored each tick as a tuple, you'd have 10 million Python objects and your program would crawl. The NumPy way is **parallel arrays**:

```python
import numpy as np

n = 10_000_000
rng = np.random.default_rng(0)

ts    = np.sort(rng.integers(0, 86400_000_000_000, size=n))   # nanoseconds within a day
price = 100.0 + np.cumsum(rng.normal(0, 0.001, size=n))
size  = rng.integers(1, 100, size=n)

# Memory: 10M * 8 + 10M * 8 + 10M * 8 = 240 MB. Fits.
# Tuple-of-objects equivalent: 10M * ~200 bytes per tuple ≈ 2 GB and ~50x slower.
```

Bucketing is a single integer-division operation:

```python
bucket = ts // 60_000_000_000          # 1-minute buckets
```

Now `bucket` is a 10M-element int array. To compute per-bucket OHLCV, the classical NumPy way is to sort by bucket, then take per-group reductions. But this is where pandas wins — `pd.DataFrame.groupby` is the right level of abstraction. The point of this example is that the **storage layout** (parallel arrays) is what makes the downstream work fast at all.

## `np.frombuffer` and zero-copy interop

`ndarray` can wrap memory you already have:

```python
buf = some_other_lib.get_buffer()      # bytes-like; e.g. an mmap or a Rust struct
arr = np.frombuffer(buf, dtype=np.float64)
```

Zero copy. Mutate `arr` and you mutate `buf`. This is how Arrow ↔ NumPy ↔ Polars ↔ PyTorch all hand each other data without paying for serialisation. Knowing this exists is the difference between writing "this pipeline is slow" and "this pipeline runs at memory bandwidth".

## `np.lib.stride_tricks` — the door to fast rolling windows

We'll get to this in [chapter 5](05-strided-windows.md). The short version: by manipulating strides, you can construct an `ndarray` view that *looks* like a 2D rolling window over a 1D array but **shares the underlying memory** — no copy, no allocation. The modern, safe API is `np.lib.stride_tricks.sliding_window_view`.

## The mental model

When you write `a[1:3, ::2]`, NumPy does **not** copy. It returns a new `ndarray` object pointing at the same data with adjusted shape and strides. When you write `a + b`, it allocates a new buffer and runs a C loop to fill it. When you write `a[a > 0]`, it allocates a new buffer and runs the mask logic.

If you internalise:

> *Every NumPy operation does either (a) make a view of existing memory, or (b) allocate a new buffer and run a tight C loop.*

— then you can predict the performance and memory of any NumPy code by inspection. That's the foundation everything else in this module stands on.

## A note on NumPy 2.x

NumPy 2.0 (mid-2024) was the first major version bump in 18 years, and 2.x is the current line as of 2026. The mental model above is unchanged, but a few things moved that bite when you run older code or examples:

- **Removed aliases.** `np.float_`, `np.int0`, `np.bool8`, `np.NaN`, `np.infty` are gone. Use `np.float64`, `np.intp`, `np.bool_`, `np.nan`, `np.inf`.
- **Scalar repr changed.** `np.float64(3.0)` now prints as `np.float64(3.0)`, not `3.0`. Cosmetic, but it breaks doctests and string-matched tests.
- **`copy=False` is now strict.** `np.array(x, copy=False)` *raises* if a copy is unavoidable (instead of silently copying). Use `np.asarray(x)` for "view if possible, copy if needed."
- **Renames.** `np.trapz` → `np.trapezoid`; `np.in1d` → `np.isin`; several `np.lib` paths moved.
- **Cleaner namespace.** Many rarely-used functions left the top-level `np.` namespace for submodules.

The course's code targets NumPy 2.x. If you're maintaining a codebase pinned to 1.26, the official `ruff` rule set (`NPY201`) flags every 2.0-incompatible usage automatically — run it before upgrading.

Continue to **[Broadcasting deeply](02-broadcasting.md)**.
