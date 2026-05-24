# Shared memory

For multi-process programs that need to share large arrays (e.g., ML model weights, big NumPy arrays, image buffers), serialising via `Queue` is wasteful — every send pickles the whole thing. **Shared memory** lets multiple processes access the same bytes directly, with zero copy.

This chapter covers the two Python tools: `multiprocessing.shared_memory` (stdlib since 3.8) and `mmap` (older, lower-level, also stdlib).

## `multiprocessing.shared_memory`

Allocate a buffer that survives across processes:

```python
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


# Parent
arr = np.arange(1_000_000, dtype=np.float64)
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
buf[:] = arr[:]
print(f"shm name: {shm.name}")             # this is what workers connect to


# Worker (in another process)
existing = shared_memory.SharedMemory(name=shm_name)
read_buf = np.ndarray((1_000_000,), dtype=np.float64, buffer=existing.buf)
print(read_buf.sum())                       # reads directly from shared memory
existing.close()


# Parent cleanup
shm.close()
shm.unlink()
```

Three concepts:

- **`create=True, size=N`** allocates a new region; remember `name` to pass to workers.
- **`name=...`** in workers attaches to an existing region.
- **`close()`** disconnects this process; **`unlink()`** destroys the region globally.

For NumPy, wrap the buffer with `np.ndarray(shape, dtype, buffer=shm.buf)` — no copy, just a view.

## Worked example: large dataset across worker pool

```python
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def worker(args):
    shm_name, shape, dtype, start, end = args
    existing = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray(shape, dtype=dtype, buffer=existing.buf)
        # Each worker processes a slice
        return float(arr[start:end].mean())
    finally:
        existing.close()


def main():
    big_data = np.random.normal(0, 1, 100_000_000).astype(np.float32)
    shm = shared_memory.SharedMemory(create=True, size=big_data.nbytes)
    shared = np.ndarray(big_data.shape, dtype=big_data.dtype, buffer=shm.buf)
    shared[:] = big_data[:]
    del big_data                          # parent doesn't need its copy anymore

    n_workers = 4
    chunk_size = shared.shape[0] // n_workers
    tasks = [
        (shm.name, shared.shape, shared.dtype, i * chunk_size, (i + 1) * chunk_size)
        for i in range(n_workers)
    ]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker, tasks))
        print(f"slice means: {results}")
        print(f"overall: {np.mean(results)}")
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
```

A 100M-float array (~400 MB) shared across 4 workers without copying. Without shared memory, each worker would receive a 400 MB pickled blob — too slow.

## `mmap` — memory-mapped files

For sharing across processes that aren't related (different launches, services on the same machine), memory-mapped files:

```python
import mmap
import os


def write_mapped():
    with open("shared.bin", "wb") as f:
        f.write(b"\x00" * 1024 * 1024)        # 1 MB
    with open("shared.bin", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm[:11] = b"hello world"
        mm.flush()
        mm.close()


def read_mapped():
    with open("shared.bin", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        print(mm[:11])
        mm.close()
```

The OS treats the file as memory; the file's contents are virtual-memory-mapped. Multiple processes mapping the same file share the same pages.

For NumPy:

```python
import numpy as np


# Writer
arr = np.memmap("data.bin", dtype=np.float64, mode="w+", shape=(1000, 1000))
arr[:] = np.random.random((1000, 1000))
arr.flush()
del arr


# Reader (different process)
read_arr = np.memmap("data.bin", dtype=np.float64, mode="r", shape=(1000, 1000))
print(read_arr[:5, :5])
```

`mode="w+"` creates; `mode="r"` opens read-only; `mode="r+"` opens read-write. Multiple readers share pages; OS handles consistency.

This is how libraries like Apache Arrow / Parquet enable zero-copy data exchange between processes.

## When to use which

| Need | Use |
|---|---|
| Share NumPy array across short-lived workers | `multiprocessing.shared_memory` |
| Share across long-lived independent processes | `mmap` (file-backed) |
| Share across machines | not memory; use TCP / object store |
| Atomic counters / values | `mp.Value` / `mp.Array` |
| Complex shared data structures | `mp.Manager` (slow but flexible) |

## Synchronisation

Shared memory has no built-in mutex; multiple writers race. Coordinate with a lock:

```python
lock = mp.Lock()

def worker(shm_name, lock):
    existing = shared_memory.SharedMemory(name=shm_name)
    with lock:
        arr = np.ndarray((100,), dtype=np.float64, buffer=existing.buf)
        arr[0] += 1
    existing.close()
```

For frequent updates, a global lock kills parallelism. Prefer designs where workers write to *disjoint* regions of the shared buffer (e.g., worker `i` writes only `arr[i*N:(i+1)*N]`).

## Race conditions and atomicity

Even single-word reads/writes are atomic at the CPU level on aligned data, but Python doesn't expose those guarantees uniformly. The safe pattern:

- For accumulation: use `mp.Value("i", 0)` with `.get_lock()`.
- For complex updates: explicit `mp.Lock()`.
- For per-worker chunks: no synchronisation needed.

## Cleanup

Shared memory persists until **unlinked**. Even if all processes close their handles, the region exists in the OS (`/dev/shm` on Linux). Leaks accumulate.

```python
import multiprocessing.shared_memory as shm_module

# Clean up by name
existing = shm_module.SharedMemory(name="orphaned_buffer")
existing.close()
existing.unlink()
```

For production: always `try/finally` around `unlink()`. If your process crashes, the shared memory lingers — periodic cleanup may be needed.

## Pitfalls

!!! warning "Forgetting `close()` and `unlink()`"
    `close()` releases the current process's handle. `unlink()` destroys the region for everyone. Leaving them open exhausts the OS's shared-memory pool (`SystemError: shm_open returned EAGAIN`).

!!! warning "Pickling SharedMemory objects"
    Don't send a `SharedMemory` instance across a process boundary; send its `.name` (a string).

!!! warning "Resource tracker warnings"
    Python's resource tracker monitors shared memory across processes. You'll see "There appear to be N leaked shared_memory objects to clean up at shutdown" if you forget to unlink. Treat as a real warning.

!!! warning "Cross-platform behaviour"
    `multiprocessing.shared_memory` works on Windows + POSIX. `mmap` semantics differ slightly. Test on your target OS.

## Bottom line

For shared memory:

- **`multiprocessing.shared_memory`** for short-lived inter-process arrays.
- **`mmap`** for file-backed sharing across independent processes.
- **No automatic synchronisation** — design for disjoint regions or use locks.
- **Always `close()` + `unlink()`** to avoid leaks.

Continue to **[Pipes, queues, signals](03-pipes-queues-signals.md)**.
