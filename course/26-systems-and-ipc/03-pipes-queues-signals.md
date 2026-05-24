# Pipes, queues, signals

The lower-level coordination primitives for multi-process programs. Pipes are one-to-one byte streams; queues are M-to-N message queues; signals are asynchronous notifications between processes (or from the OS).

## `multiprocessing.Pipe`

```python
import multiprocessing as mp


def worker(conn):
    msg = conn.recv()
    conn.send(f"got: {msg}")
    conn.close()


parent_conn, child_conn = mp.Pipe()
p = mp.Process(target=worker, args=(child_conn,))
p.start()
parent_conn.send("hello")
print(parent_conn.recv())            # got: hello
p.join()
```

`Pipe()` returns two endpoints. Send / recv at either end; messages are picklable Python objects. Bidirectional by default; `Pipe(duplex=False)` for one-way.

For high-throughput pipes, raw `os.pipe()` (returns file descriptors) is faster but requires manual serialisation.

## `multiprocessing.Queue`

For M-to-N messaging:

```python
import multiprocessing as mp


def worker(in_q, out_q):
    while True:
        task = in_q.get()
        if task is None:
            break
        result = task * 2
        out_q.put(result)


def main():
    in_q = mp.Queue()
    out_q = mp.Queue()
    workers = [mp.Process(target=worker, args=(in_q, out_q)) for _ in range(4)]
    for w in workers: w.start()
    for i in range(100): in_q.put(i)
    for _ in workers: in_q.put(None)
    for w in workers: w.join()
    while not out_q.empty():
        print(out_q.get())


if __name__ == "__main__":
    main()
```

`Queue` is thread-and-process safe. Underneath, it uses a pipe + a separate feeder thread for buffering.

For unbounded queues, `Queue()`; for bounded with backpressure, `Queue(maxsize=N)` — `put()` blocks when full.

## `JoinableQueue`

`Queue` doesn't natively know when its tasks are done. `JoinableQueue` adds `task_done()` / `join()`:

```python
q = mp.JoinableQueue()

def worker(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        process(item)
        q.task_done()


def main():
    procs = [mp.Process(target=worker, args=(q,)) for _ in range(4)]
    for p in procs: p.start()
    for item in inputs:
        q.put(item)
    q.join()                  # blocks until every item has task_done()'d
    for _ in procs: q.put(None)
    q.join()
    for p in procs: p.join()
```

For "wait until all submitted work is processed."

## `os.pipe()` and shell pipelines

Lower-level pipes (for talking to subprocesses):

```python
import os
import subprocess


r, w = os.pipe()
# Reader side
reader = subprocess.Popen(["grep", "ERROR"], stdin=r, stdout=subprocess.PIPE)
os.close(r)
# Writer
with os.fdopen(w, "w") as f:
    f.write("INFO: ok\n")
    f.write("ERROR: bad\n")
print(reader.stdout.read())          # ERROR: bad
reader.wait()
```

Useful for building shell-pipeline-style Python tools.

## Signals — OS-level coordination

Signals are asynchronous notifications:

```python
import signal
import sys
import time


def graceful_exit(signum, frame):
    print(f"got signal {signum}, cleaning up...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

while True:
    time.sleep(1)
```

Common signals:

| Signal | Number | Use |
|---|---|---|
| **SIGINT** | 2 | Ctrl-C in terminal |
| **SIGTERM** | 15 | "please exit cleanly" (Docker, systemd) |
| **SIGKILL** | 9 | unblockable — instant kill |
| **SIGHUP** | 1 | hangup; often "reload config" |
| **SIGUSR1** | 10 | application-defined |
| **SIGUSR2** | 12 | application-defined |
| **SIGCHLD** | 17 | a child process exited |

`SIGKILL` and `SIGSTOP` cannot be caught. Everything else can be.

## Signal handlers — what's safe

Inside a signal handler, very little Python is safe. The handler interrupts the main code at an arbitrary point — it may be mid-write to a structure. The pattern is **set a flag, return**:

```python
import signal


_stop = False


def handler(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, handler)

while not _stop:
    do_work()
print("clean exit")
```

The main loop checks the flag periodically. Side-effect-free, deterministic.

For asyncio:

```python
import asyncio
import signal


async def main():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    await run_until(stop)


asyncio.run(main())
```

`add_signal_handler` (asyncio-aware) is much cleaner than the raw `signal.signal`.

## Sending signals to other processes

```python
import os
import signal


# To a specific PID
os.kill(pid, signal.SIGTERM)

# To a process group
os.killpg(os.getpgid(pid), signal.SIGTERM)

# To yourself
os.kill(os.getpid(), signal.SIGTERM)
```

For coordinating supervisor → workers, sending SIGTERM gives workers a chance to clean up. If they don't respond in a few seconds, escalate to SIGKILL.

## A worked example: a worker pool with graceful shutdown

```python
import multiprocessing as mp
import signal
import time


def worker(in_q, out_q):
    # Children inherit signal disposition; reset to default and check the queue
    signal.signal(signal.SIGINT, signal.SIG_IGN)        # parent will signal via queue
    while True:
        task = in_q.get()
        if task is None:
            break
        out_q.put(process(task))


def process(task):
    time.sleep(0.1)
    return task * 2


def main():
    in_q = mp.Queue()
    out_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(in_q, out_q)) for _ in range(4)]
    for p in procs: p.start()

    stop = False
    def graceful(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, graceful)
    signal.signal(signal.SIGTERM, graceful)

    try:
        for i in range(1000):
            if stop:
                break
            in_q.put(i)
        for _ in procs: in_q.put(None)
        for p in procs: p.join()
    except Exception:
        for p in procs: p.terminate()
        raise


if __name__ == "__main__":
    main()
```

Workers ignore SIGINT (parent handles it); parent flags stop on signal; clean shutdown via sentinels in queue. Robust to Ctrl-C.

## Common patterns

- **Sentinel value** (`None`) on the queue to signal workers to exit.
- **Timeout on `get()`** if you want workers that don't block forever.
- **`JoinableQueue.task_done()` + `join()`** when you need "all work processed" detection.

## Pitfalls

!!! warning "Queue size and pickling"
    `Queue.put()` serialises with pickle. Big payloads are slow; very big payloads can deadlock if the feeder thread can't keep up.

!!! warning "Don't `terminate()` casually"
    Terminating a worker mid-write to shared resources leaves them inconsistent. Always prefer graceful shutdown via sentinels.

!!! warning "Signal handlers in libraries"
    A library that installs its own SIGINT handler can interfere with yours. Save the existing handler, chain to it if appropriate.

!!! warning "Zombie children"
    A child that exits without being `wait()`ed becomes a zombie (entry in process table). `Process.join()` cleans up. For long-running supervisors, ensure you reap.

## Bottom line

For IPC:

- **Pipe** for one-to-one byte streams.
- **Queue** for M-to-N messaging; **JoinableQueue** when you need completion tracking.
- **Signals** for asynchronous notification (graceful shutdown, reload config).
- **Set a flag, return** in signal handlers; do the work in the main loop.
- **Sentinel values** for queue-based shutdown.

Continue to **[The `os` module in depth](04-os-module.md)**.
