# Multiprocessing in depth

Module 1 covered when to use multiprocessing (CPU-bound work, the GIL story). This chapter goes deeper: spawn vs fork, the right way to share state, common patterns and traps.

## Start methods: spawn, fork, forkserver

| Method | Available on | Behaviour |
|---|---|---|
| **fork** | POSIX | child is a copy of the parent; cheap; shares state at COW |
| **spawn** | all | child is a fresh interpreter; slow but safe |
| **forkserver** | POSIX | a dedicated forking server; middle ground |

```python
import multiprocessing as mp

mp.set_start_method("spawn", force=True)        # set this once at top of program
```

**Default**: `spawn` on macOS (since 3.8) and Windows; `fork` on Linux. **Recommended**: `spawn` everywhere unless you have a specific reason.

`fork` has a famous gotcha on macOS where it can deadlock with certain native libraries (Cocoa, OpenMP). `spawn` is slower to start but reliably correct.

## The `if __name__ == "__main__":` guard

With `spawn`, the child re-imports your script. Without the guard, the child runs your script's top-level code, which often spawns more children → fork bomb:

```python
import multiprocessing as mp


def worker(x):
    return x * 2


def main():
    with mp.Pool(4) as pool:
        result = pool.map(worker, range(10))
    print(result)


if __name__ == "__main__":         # CRITICAL
    main()
```

Skip the guard and your program either crashes or fills your machine with processes. Standard pattern.

## Process Pool — `ProcessPoolExecutor`

```python
from concurrent.futures import ProcessPoolExecutor


def heavy_calc(x):
    # CPU-bound work
    return sum(i ** 2 for i in range(x))


with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(heavy_calc, [100_000] * 100))
```

`ProcessPoolExecutor` is the high-level API. Underneath, it uses `multiprocessing`. Cleaner than `mp.Pool` for most uses.

For tasks of vastly different durations, `as_completed`:

```python
from concurrent.futures import as_completed


with ProcessPoolExecutor() as pool:
    futures = {pool.submit(work, x): x for x in inputs}
    for fut in as_completed(futures):
        x = futures[fut]
        try:
            print(x, fut.result())
        except Exception as exc:
            print(f"task {x} failed: {exc!r}")
```

Each result arrives as it completes, with proper exception propagation.

## Pickling — what crosses process boundaries

`spawn` and `forkserver` send tasks to workers via pickle. So:

- **Lambdas don't pickle.** Use `def` functions or `functools.partial`.
- **Locally-defined classes don't pickle.** Define at module scope.
- **Large objects are slow.** Each task sends its arguments to a worker; if those are 100 MB, pickling them is slow.

For large shared data, use shared memory (next chapter) or memory-mapped files.

## `multiprocessing.Pool` — older but useful

```python
import multiprocessing as mp


def init_worker():
    """Run once per worker process at startup."""
    global expensive_resource
    expensive_resource = load_big_model()


def task(x):
    return expensive_resource.predict(x)


def main():
    with mp.Pool(4, initializer=init_worker) as pool:
        results = pool.map(task, batch_of_inputs)
```

`initializer` runs once per worker — useful for loading a model or opening a database connection that each worker reuses across many tasks.

## Long-running worker processes

Sometimes you want N persistent processes consuming a queue rather than a one-shot pool:

```python
import multiprocessing as mp
import queue


def worker(in_q, out_q):
    while True:
        try:
            task = in_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if task is None:                # sentinel
            break
        result = process(task)
        out_q.put(result)


def main():
    in_q = mp.Queue()
    out_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(in_q, out_q)) for _ in range(4)]
    for p in procs: p.start()

    for x in inputs:
        in_q.put(x)

    # Signal shutdown
    for _ in procs: in_q.put(None)
    for p in procs: p.join()
```

`Queue` here is process-safe; producers and consumers interact without explicit locking.

## Daemon processes

A daemon process is killed automatically when the parent exits:

```python
p = mp.Process(target=background_task, daemon=True)
p.start()
# When the main program exits, p is terminated
```

Useful for background watchdogs that should die with the application. Don't use for tasks that need clean shutdown (they get SIGKILLed without finalisers).

## Sharing values

For simple shared state:

```python
import multiprocessing as mp


def increment(counter):
    with counter.get_lock():
        counter.value += 1


if __name__ == "__main__":
    counter = mp.Value("i", 0)        # 'i' = signed int
    procs = [mp.Process(target=increment, args=(counter,)) for _ in range(100)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(counter.value)              # 100
```

`mp.Value(type_code, init)` for a single shared variable; `mp.Array(type_code, [...])` for an array. Backed by shared memory; lock for atomic operations.

For complex shared state (dicts, lists), `Manager`:

```python
with mp.Manager() as manager:
    shared_dict = manager.dict()
    shared_list = manager.list()
    # Multiple processes can read/write these safely
```

Slow (every access is a serialised RPC to the manager process) but works for any picklable object. Use sparingly.

## The fork + matplotlib / OpenMP trap

```python
# WRONG on macOS — may deadlock
import matplotlib.pyplot as plt
import multiprocessing as mp


def worker(x):
    fig, ax = plt.subplots()
    # ... do stuff ...


if __name__ == "__main__":
    with mp.Pool(4) as p:                     # default is "fork" on Linux
        p.map(worker, range(10))
```

Matplotlib (and OpenMP-using libraries) can deadlock when fork() is called between import-time and library use. The fix: `mp.set_start_method("spawn")`.

## A worked example: parallel image processing

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2


def process_one(args):
    src, dst, size = args
    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w >= h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    resized = cv2.resize(img, (new_w, new_h))
    cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return str(dst)


def main():
    src_dir = Path("photos")
    dst_dir = Path("thumbnails")
    dst_dir.mkdir(exist_ok=True)
    tasks = [(p, dst_dir / p.name, 1024) for p in src_dir.glob("*.jpg")]
    with ProcessPoolExecutor() as pool:
        for result in pool.map(process_one, tasks, chunksize=10):
            if result:
                print(f"done: {result}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

`chunksize=10` batches 10 tasks per send — reduces per-task overhead on many small jobs.

## Pitfalls

!!! warning "Workers crashing silently"
    A worker that raises is collected via the future; if you never check `fut.result()`, you never see the exception. Always `as_completed` or `fut.result()`.

!!! warning "Process pools and file handles"
    Each worker has its own file descriptors. Opening a file in main and using it in workers either errors or works incorrectly. Open files inside the worker.

!!! warning "Memory growth"
    Worker processes may grow over time (memory fragmentation, leaked objects). For very long pools, restart workers periodically (`maxtasksperchild=` in `mp.Pool`).

!!! warning "Signals don't propagate to workers automatically"
    Ctrl-C in the main process doesn't kill workers cleanly on all platforms. Handle SIGINT explicitly in workers.

## Bottom line

For multiprocessing:

- **`ProcessPoolExecutor`** for one-shot batch work.
- **`mp.Process` + `Queue`** for long-running workers.
- **`spawn`** start method by default.
- **`if __name__ == "__main__":`** always.
- **Shared memory or memory-mapped files** for large shared data.

Continue to **[Shared memory](02-shared-memory.md)**.
