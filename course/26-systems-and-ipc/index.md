# Module 26 — System programming & IPC

Below the application layer: processes, inter-process communication, shared memory, signals, the OS interface. This module covers the primitives that asyncio and multiprocessing sit on top of, and the situations where you reach for them directly.

Pages:

1. **[Multiprocessing in depth](01-multiprocessing.md)** — fork vs spawn, pools, queues, daemons.
2. **[Shared memory](02-shared-memory.md)** — `multiprocessing.shared_memory`, `mmap`, NumPy on shared buffers.
3. **[Pipes, queues, signals](03-pipes-queues-signals.md)** — coordination primitives.
4. **[The `os` module in depth](04-os-module.md)** — file descriptors, environment, working directory, exec.
5. **[Process management and supervision](05-process-management.md)** — supervising long-running services.

Start with **[Multiprocessing in depth](01-multiprocessing.md)**.
