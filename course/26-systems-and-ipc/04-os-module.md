# The `os` module in depth

`os` is the Python interface to the operating system. Most of what you'd write in shell — file descriptors, environment, working directory, process control, file metadata — is here. Knowing the highlights saves you from shelling out for everything.

## File descriptors

```python
import os

fd = os.open("/tmp/log.txt", os.O_RDONLY)
data = os.read(fd, 4096)
os.close(fd)
```

Lower-level than `open()`. Used when you need flags `open()` doesn't expose, or when interfacing with C libraries that return file descriptors.

Common flags (OR them together):

- `os.O_RDONLY`, `os.O_WRONLY`, `os.O_RDWR`
- `os.O_CREAT` (create if needed)
- `os.O_APPEND`
- `os.O_TRUNC`
- `os.O_EXCL` (fail if exists; for atomic creation)
- `os.O_NONBLOCK` (non-blocking)
- `os.O_CLOEXEC` (close on exec — security best practice)

## File metadata — `os.stat`

```python
import os

st = os.stat("/tmp/log.txt")
print(f"size={st.st_size}, mtime={st.st_mtime}, mode={oct(st.st_mode)}")
print(f"uid={st.st_uid}, gid={st.st_gid}")
```

For Python-friendly paths, `pathlib.Path.stat()` returns the same thing.

For symbolic link target metadata vs the symlink itself: `os.stat()` follows; `os.lstat()` does not.

## Environment variables

```python
import os

# Read
api_key = os.environ.get("API_KEY")
debug = os.environ.get("DEBUG", "0") == "1"

# Set (only for this process and its children)
os.environ["LOG_LEVEL"] = "DEBUG"

# All env vars
for k, v in os.environ.items():
    print(f"{k}={v}")
```

Changes to `os.environ` after process start affect only the current process and any future children. Doesn't propagate to the shell that launched you.

## Working directory

```python
print(os.getcwd())
os.chdir("/tmp")
print(os.getcwd())            # /tmp
```

For scripts, `os.chdir` is generally a bad idea — it surprises code that doesn't expect cwd to change. Pass absolute paths instead.

For context-managed cwd changes:

```python
import contextlib
import os


@contextlib.contextmanager
def working_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


with working_dir("/tmp"):
    do_stuff_in_tmp()
# cwd restored
```

## Process info

```python
print(os.getpid())              # current process ID
print(os.getppid())             # parent's PID
print(os.getuid())              # user ID (POSIX)
print(os.getgid())              # group ID
print(os.uname())               # uname info
print(os.cpu_count())           # logical CPU count
```

`os.cpu_count()` is the standard "how many cores?" call. Useful for choosing worker counts.

## `os.path` (legacy) vs `pathlib`

The course defaults to `pathlib` (Module 22 chapter 1). `os.path` predates it; you'll see it in older code:

| `os.path` | `pathlib` equivalent |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.path.basename(p)` | `Path(p).name` |
| `os.path.dirname(p)` | `Path(p).parent` |
| `os.path.splitext(p)` | `Path(p).stem`, `Path(p).suffix` |
| `os.path.abspath(p)` | `Path(p).resolve()` |
| `os.path.expanduser(p)` | `Path(p).expanduser()` |

New code: `pathlib`. Legacy code: `os.path`.

## `os.walk` — tree traversal

```python
for dirpath, dirnames, filenames in os.walk("/var/log"):
    for name in filenames:
        print(os.path.join(dirpath, name))
```

`pathlib` equivalent: `Path("/var/log").rglob("*")` returns all paths; you can filter.

`os.walk` is procedural and a bit faster; `rglob` is more Pythonic.

## Process exec

```python
import os


# Replace current process with another
os.execvp("ls", ["ls", "-la", "/tmp"])
# Nothing after this line runs — current process is gone
```

`execvp` (the `vp` variants find the binary on PATH) is rarely used directly in Python; `subprocess.run` is the modern interface. But `exec` is the OS primitive underneath.

## Files: rename, remove, link

```python
os.rename("a.txt", "b.txt")               # rename within filesystem
os.replace("a.txt", "b.txt")              # rename, atomic, overwrites
os.remove("file.txt")                     # delete
os.unlink("file.txt")                     # same as remove (POSIX terminology)
os.link("a.txt", "hardlink.txt")          # hard link
os.symlink("/abs/path", "shortcut")       # symbolic link
```

`os.replace` is the atomic rename used in Module 22 chapter 1.

## Permissions

```python
os.chmod("/tmp/file.txt", 0o600)          # rw for owner only
os.chown("/tmp/file.txt", uid=1000, gid=1000)
```

For sensitive files (secrets, keys), `0o600`. For shared but read-only, `0o644`. The octal `0o` prefix is essential.

## Disk usage and free space

```python
import shutil

stats = shutil.disk_usage("/")
print(f"total={stats.total / 1e9:.1f} GB, "
      f"used={stats.used / 1e9:.1f} GB, "
      f"free={stats.free / 1e9:.1f} GB")
```

For monitoring scripts, this is the standard "is the disk full?" call.

## Spawning subprocesses (briefly)

For running other programs, `subprocess` (Module 22 chapter 5) is preferred. The lower-level `os.fork` + `os.exec` family exists but is rarely needed in Python:

```python
import os


pid = os.fork()                # POSIX only — creates a child process
if pid == 0:
    # Child
    os.execvp("ls", ["ls"])
    # Never reached
else:
    # Parent
    os.waitpid(pid, 0)
```

This is what `subprocess.Popen` does underneath. Use `subprocess` in your own code.

## `os.urandom` for randomness

For cryptographic randomness:

```python
random_bytes = os.urandom(32)           # 32 random bytes
```

Don't use `random.random()` for anything security-related — it's a Mersenne twister, not cryptographically secure. `os.urandom` reads from the OS's CSPRNG.

For higher-level cryptographic primitives, see Module 27 chapter 5.

## A pattern: locked file write

```python
import os
import fcntl                      # POSIX only


def append_with_lock(path: str, text: str):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, text.encode())
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
```

Two processes appending to the same file without lock can interleave bytes. With `fcntl.flock` (POSIX) or `msvcrt.locking` (Windows), they serialise.

For cross-platform, `filelock` (Module 22 chapter 1) handles both.

## Pitfalls

!!! warning "Forgetting `os.close(fd)`"
    File descriptor leaks. With low ulimit, your process eventually fails on "too many open files". Always `try/finally` or use `with` blocks where possible.

!!! warning "Race conditions in `os.path.exists` + open"
    "Check then act" is a classic TOCTOU bug. Prefer `os.O_EXCL` for "create if doesn't exist" or try/except `FileExistsError`.

!!! warning "Cross-platform `os.path.join` with absolute components"
    `os.path.join("/a", "/b")` returns `/b` — the second absolute path overrides. Surprising; check inputs.

!!! warning "`os.environ` not thread-safe"
    Concurrent reads + writes to env vars can race. For multi-threaded code that mutates env, lock explicitly.

## Bottom line

For the `os` module:

- **`os.environ`** for env vars; **`os.urandom`** for crypto-random.
- **`pathlib` over `os.path`** in new code.
- **`os.replace`** for atomic rename.
- **`fcntl.flock`** (or `filelock`) for cross-process file locks.
- **`subprocess`**, not `os.fork` + `os.exec`, for spawning processes.

Continue to **[Process management and supervision](05-process-management.md)**.
