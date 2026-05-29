# Modern Python: what changed in 3.12–3.14

The language moves. This chapter consolidates the features that landed across Python 3.12, 3.13, and 3.14 (the current release as of 2026) that change how you write everyday code. If you learned Python on 3.8–3.10, this is the diff that matters.

## t-strings — template strings (3.14, PEP 750)

The headline language feature of 3.14. A t-string looks like an f-string but produces a `Template` object instead of a `str` — letting you intercept the interpolated values *before* they're joined into the final string. That's the foundation for **safe** interpolation: HTML escaping, SQL parameterisation, shell-arg quoting, without the injection risk of f-strings.

```python
from string.templatelib import Template


name = "'; DROP TABLE users; --"
template: Template = t"SELECT * FROM users WHERE name = {name}"

# template is NOT a string yet. It exposes the static parts and the
# interpolated values separately:
for item in template:
    match item:
        case str() as static:
            print("static:", static)
        case _:                          # Interpolation object
            print("value:", item.value, "expr:", item.expression)
```

An f-string would have already concatenated the dangerous value. A t-string hands you the parts so a processing function can escape each interpolation correctly:

```python
def safe_sql(template: Template) -> tuple[str, list]:
    """Turn a t-string into a parameterised query + bound values."""
    query_parts, params = [], []
    for item in template:
        if isinstance(item, str):
            query_parts.append(item)
        else:
            query_parts.append("?")           # placeholder
            params.append(item.value)
    return "".join(query_parts), params


query, params = safe_sql(t"SELECT * FROM users WHERE name = {name}")
# query  = "SELECT * FROM users WHERE name = ?"
# params = ["'; DROP TABLE users; --"]        # safely bound, not interpolated
```

The win: libraries can ship `html(t"...")`, `sql(t"...")`, `sh(t"...")` helpers that are injection-safe by construction. Expect web and DB libraries to adopt t-string APIs through 2026.

Use a plain f-string when you just want a string. Use a t-string when the interpolated values cross a trust or escaping boundary.

## Deferred annotation evaluation by default (3.14, PEP 649/749)

For years the pattern was `from __future__ import annotations` to make annotations lazy strings (so forward references and expensive types didn't cost anything at import). In 3.14 **deferred evaluation is the default** — you no longer need the `__future__` import.

```python
# Python 3.14: this just works; no __future__ import needed
class Node:
    def __init__(self, parent: Node | None, children: list[Node]):
        self.parent = parent
        self.children = children
```

`Node` referencing itself in its own method signature used to require quotes or the future import. Now it's free.

To introspect annotations at runtime (what pydantic / dataclasses / FastAPI do internally), use the new `annotationlib`:

```python
import annotationlib

ann = annotationlib.get_annotations(Node.__init__, format=annotationlib.Format.VALUE)
# Resolves forward references to actual objects; FORWARDREF and STRING formats also available
```

For your own code: drop `from __future__ import annotations` on 3.14+ (it's harmless but unnecessary). If you support older versions too, keep it — it's still valid.

## Generic syntax recap (3.12, PEP 695)

Covered in the typing chapter, but worth consolidating here as "modern syntax." Since 3.12 you write generics without `TypeVar` boilerplate:

```python
# Modern (3.12+)
class Stack[T]:
    def push(self, x: T) -> None: ...
    def pop(self) -> T: ...

def first[T](xs: list[T]) -> T:
    return xs[0]

type Vector = list[float]                    # the `type` statement, also 3.12+
```

No more `T = TypeVar("T")` at module scope. For libraries supporting 3.11, the old form is still required; for application code on 3.12+, use the new syntax.

## except without parentheses (3.14, PEP 758)

```python
# 3.14: parentheses optional when there's no `as`
try:
    risky()
except ValueError, TypeError:
    handle()

# Pre-3.14 (still works everywhere):
try:
    risky()
except (ValueError, TypeError):
    handle()
```

Minor, but removes a papercut. Note the parens are still required if you bind with `as e`.

## Zstandard in the standard library (3.14, PEP 784)

The whole course uses `zstd` compression for Parquet (via pyarrow). As of 3.14 it's also in the stdlib directly:

```python
from compression import zstd

compressed = zstd.compress(b"large payload" * 1000, level=10)
original = zstd.decompress(compressed)
```

`gzip`, `bz2`, `lzma`, and now `zstd` all live under the new `compression` package (the old top-level names still work as aliases). For new code compressing logs, backups, or archives, `zstd` is the best speed/ratio trade-off and now needs no third-party dependency.

## Improved error messages (3.12 → 3.14, ongoing)

Each release sharpens tracebacks. By 3.14 you get:

- **Coloured tracebacks** in the REPL and on error.
- **"Did you mean...?"** suggestions for typos in names, attributes, and keyword arguments.
- **Precise carets** pointing at the exact sub-expression that failed (e.g., which attribute access in `a.b.c.d` raised).

Nothing to do but enjoy them. If you script around tracebacks, note the format changed; parse with `traceback` module APIs rather than regex on the text.

## `pathlib` gains (3.12+)

```python
from pathlib import Path

# walk() — pathlib's answer to os.walk (3.12+)
for dirpath, dirnames, filenames in Path("project").walk():
    ...

# case-sensitivity control, relative_to with walk_up (3.12+)
Path("/a/b/c").relative_to("/a/x", walk_up=True)     # ../b/c
```

`Path.walk()` means you rarely need `os.walk` anymore.

## What this means for the course's code

Every code block in this course runs on **3.11+** (the stated floor). The features above are *additive*:

- t-strings, stdlib zstd, subinterpreters, `except` without parens → **3.14+ only**.
- generic syntax, `type` statement, `Path.walk` → **3.12+**.
- `TypeIs` → **3.13+**.

Where a feature needs a newer version than 3.11, the course flags it inline. If you're on the current interpreter (3.14 as of 2026), all of it is available.

## Version-targeting guidance for your own projects

| You support | Write |
|---|---|
| 3.11+ (libraries, max reach) | classic `TypeVar`, `(Exc1, Exc2)`, keep `from __future__ import annotations` |
| 3.12+ (most applications) | PEP 695 generics, `type` statement, `Path.walk` |
| 3.14+ (greenfield, controlled deploys) | t-strings, stdlib `compression.zstd`, drop the `__future__` import, subinterpreters |

For a brand-new internal project in 2026, targeting **3.13 or 3.14** is reasonable — you get the modern syntax and the maturing free-threading story. For a published library, **3.11** is still the pragmatic floor (it's widely deployed and EOL is October 2027).

## Bottom line

The deltas worth adopting now:

- **t-strings** for any string crossing an escaping boundary (3.14).
- **Drop `from __future__ import annotations`** on 3.14+.
- **PEP 695 generic syntax** on 3.12+.
- **stdlib `compression.zstd`** instead of a third-party dependency (3.14).
- **`Path.walk`** instead of `os.walk` (3.12+).

Continue to **[Module 2 — Python Hacks](../02-python-hacks/index.md)**.
