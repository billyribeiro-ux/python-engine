# Module 3 — NumPy Mastery

NumPy is the single most important library in scientific Python. Pandas is built on it. PyTorch's mental model echoes it. JAX is "NumPy on accelerators". A working understanding of NumPy is the gate to everything that follows in this course.

The good news: NumPy's apparent complexity comes down to a handful of ideas that, once internalised, generalise everywhere. This module covers them.

Pages:

1. **[ndarray internals](01-ndarray-internals.md)** — strides, dtype, views vs copies, the C/F layout question.
2. **[Broadcasting deeply](02-broadcasting.md)** — the single rule that explains 90% of NumPy "magic".
3. **[Vectorisation patterns](03-vectorization.md)** — replacing Python loops with array math; the patterns that come up over and over.
4. **[einsum and tensor contractions](04-einsum.md)** — for the operations `@` can't reach.
5. **[Strided rolling windows](05-strided-windows.md)** — `sliding_window_view` and the lower-level `as_strided` for advanced use.
6. **[Numerical stability](06-stability.md)** — log-sum-exp, stable softmax, catastrophic cancellation, condition numbers.

This module is dense. Read it twice if you can.

Start with **[ndarray internals](01-ndarray-internals.md)**.
