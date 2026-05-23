# File handling at scale

Half of operational engineering is moving bytes around — log files, data dumps, model artifacts, backups, batch inputs and outputs. Python's standard library has excellent primitives for all of it; this chapter is the working patterns that hold up under real-world conditions.

## `pathlib` is the only API you should use

`os.path` works but is procedural. `pathlib.Path` is object-oriented and dramatically more readable:

```python
from pathlib import Path

root = Path("/var/data/strategies")
for p in root.rglob("*.parquet"):
    print(p, p.stat().st_size, p.stat().st_mtime)
```

Useful methods you'll reach for constantly:

| Method | Does |
|---|---|
| `Path("a/b/c.txt").parent` | `Path("a/b")` |
| `Path("a/b/c.txt").stem` | `"c"` |
| `Path("a/b/c.txt").suffix` | `".txt"` |
| `Path("a/b/c.txt").with_suffix(".gz")` | `Path("a/b/c.gz")` |
| `p.exists()`, `p.is_file()`, `p.is_dir()` | predicates |
| `p.mkdir(parents=True, exist_ok=True)` | create directory tree |
| `p / "subdir" / "file.txt"` | join (overloaded `/`) |
| `p.read_text(encoding="utf-8")` | read a small text file |
| `p.read_bytes()` | read binary |
| `p.write_text(...)` / `p.write_bytes(...)` | write (overwrites) |
| `p.rglob("*.csv")` | recursive glob |
| `p.iterdir()` | non-recursive listing |
| `p.unlink(missing_ok=True)` | delete file |
| `p.rename(new_path)` | atomic rename on same filesystem |

`os.path` should not appear in new code.

## Atomic writes — the single most underused pattern

Naive write:

```python
Path("data.csv").write_text(big_string)
```

If the process crashes mid-write, you have a half-written `data.csv`. Worse, anything reading from it might pick up the partial state.

The atomic pattern: write to a temp file in the same directory, then **rename**:

```python
import os
from pathlib import Path

def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)        # atomic on POSIX + Windows
```

`os.replace` is atomic on the same filesystem. A reader sees either the old file or the new file — never a half-write.

For binary:

```python
def atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(content)
    os.replace(tmp, path)
```

Wrap every "produce an output file" operation in this pattern. The cost is one extra disk write; the benefit is that your downstream readers never see a corrupt file.

## File locks — when multiple processes touch the same file

Two cron jobs both want to update `state.json`. Without coordination, you get a race condition. `fcntl` (POSIX) or `msvcrt` (Windows) gives you advisory locking; the cross-platform library is [`filelock`](https://github.com/tox-dev/py-filelock):

```python
from filelock import FileLock

lock = FileLock("state.json.lock", timeout=10)
with lock:
    state = json.loads(Path("state.json").read_text())
    state["last_run"] = time.time()
    Path("state.json").write_text(json.dumps(state))
```

The lock is **advisory** — only processes that explicitly acquire the lock cooperate. But that's almost always what you want for coordinating your own scripts.

For pure within-process coordination, use `threading.Lock`. For cross-process on the same machine, `FileLock`. For across-machine, you need a real distributed lock service (Redis, ZooKeeper, etcd).

## Glob patterns — the precise rules

```python
from pathlib import Path

root = Path("/data")
list(root.glob("*.csv"))               # one level deep, .csv files
list(root.glob("**/*.csv"))            # any depth (** = anything)
list(root.rglob("*.csv"))              # same — rglob is sugar for **
list(root.glob("2024-??-??.csv"))      # ? = single char; useful for dates
list(root.glob("[12]024-*.csv"))       # character class
```

Watch out: `glob` returns results in **filesystem order**, which isn't sorted. If order matters, `sorted(root.glob(...))`.

## Stream large files; don't load them

For a 10 GB file, `path.read_text()` will OOM. Iterate line by line:

```python
with open("huge.log") as f:
    for line in f:                    # iterates one line at a time
        process(line)
```

For binary, read in fixed-size chunks:

```python
with open("huge.bin", "rb") as f:
    while chunk := f.read(64 * 1024):
        process(chunk)
```

The `:=` walrus is built for exactly this pattern.

For CSV at scale, use `pandas.read_csv(..., chunksize=10000)` to get an iterator of frames; or switch to `polars.scan_csv` for proper lazy reading (Module 4 chapter 5).

## Archives — tar, zip, gzip

```python
import tarfile
import zipfile
import gzip
from pathlib import Path

# Compress a single file with gzip
with open("input.csv", "rb") as src, gzip.open("input.csv.gz", "wb") as dst:
    dst.writelines(src)

# Bundle a directory into a tarball
with tarfile.open("backup.tar.gz", "w:gz") as tar:
    tar.add(Path("data/"), arcname="data")

# Read a specific file out of a zip without extracting all
with zipfile.ZipFile("archive.zip") as zf:
    with zf.open("nested/file.txt") as f:
        content = f.read().decode("utf-8")
```

Streaming a member out of a zip without writing it to disk is sometimes the cleanest way to consume a vendor data dump.

## Checksums — verifying integrity

For any file you receive or transmit, compute its checksum:

```python
import hashlib

def sha256_of(path: Path, chunk_size: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
```

For data pipelines that pull from S3 or a vendor: download → checksum → compare against the vendor's expected sha256 → proceed only if match. Catches transmission errors and tampering.

## Directory watching

For "react when a new file appears", `watchdog` is the standard:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        process_new_file(Path(event.src_path))


obs = Observer()
obs.schedule(NewFileHandler(), path="data/incoming", recursive=False)
obs.start()
try:
    while True:
        time.sleep(60)
finally:
    obs.stop()
    obs.join()
```

For network filesystems (NFS, SMB), inotify-style watching is unreliable. Fall back to **polling** with an mtime check.

## Temporary files

For ephemeral intermediates, `tempfile` cleans up after itself:

```python
import tempfile
from pathlib import Path

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
    tmp_path = Path(tf.name)
    tf.write(b"some,data\n1,2\n")

# tmp_path is now usable; will NOT be deleted automatically because delete=False
# Caller is responsible for cleanup

with tempfile.TemporaryDirectory() as td:
    work_dir = Path(td)
    # work_dir is removed when the with-block exits
```

`tempfile.TemporaryDirectory()` is great for "I need a workspace; clean up when I'm done."

## A worked example: a robust file-mirror script

```python
import shutil
from pathlib import Path

def mirror_directory(src: Path, dst: Path, dry_run: bool = False) -> dict:
    """One-way sync: ensure dst contains the same files as src (by name).
    Atomic per-file replacement; no orphan cleanup."""
    src = src.resolve()
    dst = dst.resolve()
    stats = {"copied": 0, "skipped_identical": 0, "errors": []}
    for src_path in src.rglob("*"):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and dst_path.stat().st_size == src_path.stat().st_size:
            if sha256_of(dst_path) == sha256_of(src_path):
                stats["skipped_identical"] += 1
                continue
        try:
            if not dry_run:
                tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
                shutil.copy2(src_path, tmp)
                tmp.replace(dst_path)
            stats["copied"] += 1
        except OSError as exc:
            stats["errors"].append((str(rel), str(exc)))
    return stats
```

Atomic per file, idempotent (checksum-skip), dry-run flag, error collection rather than fail-fast. That's the production shape.

## Pitfalls

!!! warning "Renaming across filesystems isn't atomic"
    `os.replace` is atomic only within the same filesystem. Crossing devices (e.g. `/tmp` on tmpfs to `/data` on disk) silently degrades to copy+delete with the same crash window as a naive write. Always write the temp file in the same directory as the final target.

!!! warning "File handles leak"
    `open()` without a `with` block (or explicit `close()`) leaks. On long-running processes, this can exhaust the OS file-descriptor limit. Always use `with`.

!!! warning "Globs match `.git` and `__pycache__`"
    `**/*.py` matches files inside hidden directories too. If you want to exclude them, post-filter or use `pathspec` (gitignore-style patterns).

!!! warning "`Path.is_dir()` follows symlinks"
    By default. For raw filesystem state, `os.path.islink(p)` first.

## Bottom line

For production file handling:

- **`pathlib` everywhere**; never `os.path` in new code.
- **Atomic writes** via tmp-file + `os.replace` for any persisted state.
- **FileLock** for cross-process coordination on the same machine.
- **Stream large files**; don't load them.
- **Checksum** files at trust boundaries.
- **`watchdog`** for event-driven flows; **polling** on network filesystems.

Continue to **[CSV, Excel, and tabular interchange](02-csv-and-excel.md)**.
