"""Verify the course's Python code blocks.

Two levels of checking:

1. **Syntax check (default):** extract every ```python fenced block from
   ``course/**/*.md`` and ``compile()`` it. Catches typos and broken syntax in
   teaching code without executing anything (no network, no side effects).

   Blocks that are intentionally not standalone-parseable (REPL transcripts,
   deliberate fragments) opt out with a first-line marker:

       ```python
       # verify-docs: skip
       ...
       ```

2. **Execution check (`--run`):** additionally executes blocks marked

       ```python
       # verify-docs: run
       ...
       ```

   in a fresh namespace. Use only for self-contained, side-effect-free,
   network-free snippets.

Exit code is non-zero if any checked block fails — suitable for CI
(`make verify-docs`).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

COURSE = Path(__file__).resolve().parent.parent / "course"
FENCE = "```"


@dataclass
class Block:
    path: Path
    start_line: int
    code: str
    marker: str | None  # "skip", "run", or None


def extract_python_blocks(md_path: Path) -> list[Block]:
    blocks: list[Block] = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # Opening fence for a python block (allow ```python, ```py, with attrs)
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            info = stripped[len(FENCE) :].strip().lower()
            lang = info.split()[0] if info else ""
            if lang in ("python", "py"):
                start = i + 1
                body: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith(FENCE):
                    body.append(lines[i])
                    i += 1
                # Code fenced inside an admonition (!!! ...) is indented;
                # dedent so it parses as a standalone program.
                code = textwrap.dedent("\n".join(body))
                marker = None
                for b in body[:2]:
                    s = b.strip()
                    if s == "# verify-docs: skip":
                        marker = "skip"
                    elif s == "# verify-docs: run":
                        marker = "run"
                blocks.append(Block(md_path, start, code, marker))
        i += 1
    return blocks


_REPL = re.compile(r"^\s*>>> ", re.MULTILINE)
_MAGIC = re.compile(r"^\s*[%!]\w", re.MULTILINE)

# Syntax newer than some running interpreters. A block using these is valid
# Python but won't parse on an older runtime, so we skip-with-note rather than
# fail (unless --strict-version is passed, e.g. when running on 3.14+).
_NEWER_SYNTAX = [
    re.compile(r"^\s*type\s+\w+\s*=", re.MULTILINE),  # PEP 695 type statement (3.12+)
    re.compile(r"\b(?:class|def)\s+\w+\["),  # PEP 695 generics (3.12+)
    re.compile(r"(?<![\w'\"])t[\"']"),  # t-strings (3.14+)
    re.compile(r"^\s*except\s+\w[\w.]*\s*,\s*\w", re.MULTILINE),  # except A, B (3.14+)
]


def is_non_program(code: str) -> bool:
    """True for blocks that aren't standalone programs: REPL transcripts
    (>>> prompts with interleaved output) and IPython/shell magics (%timeit,
    !pip). These are auto-skipped — they can't and shouldn't be parsed as a
    module."""
    return bool(_REPL.search(code) or _MAGIC.search(code))


def uses_newer_syntax(code: str) -> bool:
    return any(p.search(code) for p in _NEWER_SYNTAX)


def check_syntax(block: Block) -> str | None:
    """Return an error string if the block fails to parse, else None."""
    try:
        ast.parse(block.code)
        return None
    except SyntaxError as exc:
        return f"{block.path.relative_to(COURSE.parent)}:{block.start_line}: SyntaxError: {exc.msg}"


def run_block(block: Block) -> str | None:
    ns: dict = {}
    try:
        exec(compile(block.code, str(block.path), "exec"), ns)
        return None
    except Exception as exc:
        return f"{block.path.relative_to(COURSE.parent)}:{block.start_line}: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="also execute blocks marked '# verify-docs: run'"
    )
    parser.add_argument(
        "--strict-version",
        action="store_true",
        help="treat newer-than-runtime syntax as an error (run on Python 3.14+)",
    )
    args = parser.parse_args()

    md_files = sorted(COURSE.rglob("*.md"))
    all_blocks: list[Block] = []
    for md in md_files:
        all_blocks.extend(extract_python_blocks(md))

    syntax_errors: list[str] = []
    skipped = 0
    version_gated = 0
    checked = 0
    for block in all_blocks:
        if block.marker == "skip" or is_non_program(block.code):
            skipped += 1
            continue
        checked += 1
        err = check_syntax(block)
        if err:
            # A parse failure that's explained by newer-than-runtime syntax is
            # skipped (unless --strict-version), not treated as a defect.
            if not args.strict_version and uses_newer_syntax(block.code):
                version_gated += 1
                continue
            syntax_errors.append(err)

    print(
        f"[verify-docs] {len(all_blocks)} python blocks across {len(md_files)} files; "
        f"syntax-checked {checked}, skipped {skipped}, version-gated {version_gated}"
    )

    run_errors: list[str] = []
    if args.run:
        to_run = [b for b in all_blocks if b.marker == "run"]
        print(f"[verify-docs] executing {len(to_run)} blocks marked 'run'")
        for block in to_run:
            err = run_block(block)
            if err:
                run_errors.append(err)

    for e in syntax_errors:
        print("  SYNTAX  " + e)
    for e in run_errors:
        print("  RUNTIME " + e)

    if syntax_errors or run_errors:
        print(f"[verify-docs] FAILED: {len(syntax_errors)} syntax, {len(run_errors)} runtime")
        return 1
    print("[verify-docs] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
