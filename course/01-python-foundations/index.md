# Module 1 — Python Foundations

You can write Python for years and never need most of what's in this module. If you're aiming for L7+ work, you need all of it — because the things you do every day (writing fast data pipelines, building tradeable strategies, debugging weird bugs in someone else's library at 2am) are built out of these primitives. Knowing them is the difference between someone who *uses* Python and someone who *understands* it.

This module is split into seven chapters:

1. **[The data model](01-data-model.md)** — what an object actually *is* in Python, dunder methods, the protocol mindset, `__slots__`, MRO.
2. **[Descriptors and dunder methods](02-descriptors.md)** — properties demystified, descriptors from scratch, `__init_subclass__`, `__class_getitem__`, the practical bag of tricks.
3. **[Iteration, generators, coroutines](03-iteration.md)** — the iteration protocol, `yield from`, generator pipelines that beat hand-rolled C, the path from generators to `asyncio`.
4. **[Async, threads, processes, and the GIL](04-concurrency.md)** — when to reach for which, the truth about the GIL (and the free-threaded build), structured concurrency with `asyncio.TaskGroup`.
5. **[Typing for engineers](05-typing.md)** — `Protocol`, `TypeGuard`, generics, `Annotated`, `Self`, `Never`, and why static typing in Python is finally good.
6. **[Dataclasses, attrs, pydantic](06-dataclasses.md)** — three near-identical-looking libraries with very different tradeoffs.
7. **[Modern Python: what changed in 3.12–3.14](07-modern-python.md)** — t-strings, deferred annotations, generic syntax, stdlib zstd, the features that landed in the current releases.

## Why this matters for trading code

Quant systems live and die by three things: **correctness, latency, and observability**. Every advanced Python feature in this module exists to make one of those better.

- **Correctness** comes from types, protocols, immutability, descriptors that enforce invariants.
- **Latency** comes from understanding generators (lazy data flow), asyncio (I/O concurrency), the GIL (CPU concurrency), and `__slots__` (memory + dispatch speed).
- **Observability** comes from understanding the data model deeply enough to write good `__repr__`, custom logging, and debuggable abstractions.

If you skim one module, don't skim this one.

Start with **[The data model](01-data-model.md)**.
