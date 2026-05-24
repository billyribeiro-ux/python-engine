# Dependency management — uv, pip-tools, poetry

Specifying `pandas>=2.0` in `pyproject.toml` says "we need pandas." It doesn't say *which version* you actually ran. **Lockfiles** pin every dependency (including transitive ones) to exact versions so deployments are reproducible.

This chapter covers the three tools you'll encounter: `uv` (the modern fast one), `pip-tools` (the simple old one), `poetry` (the opinionated workflow one).

## `uv` — the new standard

[`uv`](https://github.com/astral-sh/uv) is Astral's Rust-backed package manager. ~10-100× faster than pip; understands `pyproject.toml`; can replace pip + virtualenv + pip-tools + pyenv. As of 2026 it's the recommended default for new projects.

### Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
# or: pip install uv
```

### Daily workflow

```bash
# Create a virtualenv and lockfile
uv venv .venv
source .venv/bin/activate

# Install everything from pyproject.toml + lockfile
uv pip install -e ".[dev]"

# Add a dependency
uv add httpx
uv add --dev pytest

# Sync after pulling someone else's changes
uv sync

# Run a tool without installing (like pipx)
uv tool run black .
uv tool install ruff
```

`uv.lock` is the lockfile — commit it. `uv sync` makes the environment match it exactly.

### Why uv wins

- 10-100× faster resolution than pip. Visible on big dependency graphs.
- Single binary; no Python runtime required for the bootstrap.
- Drop-in compatible with pip CLI (`uv pip install ...`).
- Built-in lockfile + virtualenv + workspace support.

## `pip-tools` — the simple old one

For projects that want a lockfile without changing their pip workflow:

```bash
pip install pip-tools
```

`requirements.in` (your specs):

```
numpy>=1.26
pandas>=2.2
httpx>=0.27
```

Compile to `requirements.txt` (the lockfile):

```bash
pip-compile requirements.in
```

Install:

```bash
pip-sync requirements.txt
```

`pip-compile --upgrade` to bump pinned versions intentionally; `pip-compile` after editing `.in` to keep the lockfile aligned.

For dev deps:

```
# dev-requirements.in
-r requirements.in
pytest
ruff
```

`pip-compile dev-requirements.in` produces `dev-requirements.txt`.

This is the lowest-friction option for legacy projects. No new tooling beyond pip.

## `poetry` — the opinionated workflow

[Poetry](https://python-poetry.org/) is a more opinionated package + venv + lockfile manager:

```bash
# Initialise a new project
poetry init

# Add dependencies
poetry add httpx
poetry add --group dev pytest

# Install (creates venv automatically)
poetry install

# Run something in the venv
poetry run pytest
poetry shell                # activate the venv interactively
```

`poetry.lock` is the lockfile; `pyproject.toml` is the spec.

Poetry has a different `[tool.poetry]` config section that's NOT standards-compliant — it predates PEP 621. Newer Poetry versions support PEP 621 with `[project]`, but older repos use `[tool.poetry.dependencies]`. Mind the difference.

## Choosing

| Need | Use |
|---|---|
| Brand new project, speed matters | **uv** |
| Existing pip-based project, minimum disruption | **pip-tools** |
| Existing Poetry project | **Poetry** (stay) |
| Building a library to publish | any of the three |
| Team coming from JS / Rust | **uv** (`cargo`-like ergonomics) |

Most modern Python teams in 2026 are moving to uv.

## The lockfile discipline

Whatever tool you use, the rules are universal:

1. **Commit the lockfile.** It's part of the repo.
2. **CI installs from the lockfile**, not the spec.
3. **`uv sync` / `pip-sync` / `poetry install` matches the environment exactly.** No drift.
4. **Periodic upgrades**: weekly or monthly, run the equivalent of `--upgrade`; review changes; commit.
5. **Security updates are an exception** — when a CVE drops, upgrade that one package immediately, don't wait for the scheduled bump.

## Constraints vs requirements

A constraint file says "if you install this package, here's the version" without requiring the package itself:

```
# constraints.txt
urllib3==2.0.7         # we don't depend on urllib3, but if a transitive dep does, use this version
```

```bash
pip install -r requirements.txt -c constraints.txt
```

Useful for enterprises with internal security policies pinning specific transitive versions.

## Dev dependencies

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "black>=24.4", "mypy>=1.10"]
```

In `uv`:

```bash
uv add --dev pytest
```

In Poetry:

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
```

For deployment, only install runtime deps (`pip install .` without the extras). For development, `pip install -e .[dev]`.

## Workspaces / monorepos

For a repo containing multiple packages:

```
monorepo/
├── pyproject.toml          # workspace config
├── packages/
│   ├── core/
│   │   ├── pyproject.toml
│   │   └── src/core/
│   ├── trading/
│   │   ├── pyproject.toml
│   │   └── src/trading/
│   └── reports/
│       ├── pyproject.toml
│       └── src/reports/
└── apps/
    └── runner/
        ├── pyproject.toml
        └── src/runner/
```

uv has first-class workspace support:

```toml
# top-level pyproject.toml
[tool.uv.workspace]
members = ["packages/*", "apps/*"]
```

`uv sync` installs everything in one go; cross-package dependencies are resolved as local paths.

Poetry workspaces are less mature; for monorepos in 2026, uv is the better fit.

## Pitfalls

!!! warning "No lockfile in production"
    `pip install -e .` installs from `pyproject.toml`, which has loose specs. Two deployments three weeks apart get different versions. Always lockfile + sync.

!!! warning "`pip install --upgrade` without review"
    Auto-upgrading floods you with semver-breaking changes. Lock; upgrade deliberately.

!!! warning "Multiple Python environments"
    `pip install -e .` into the wrong venv installs into your system Python. Always confirm `which python` first.

!!! warning "Caching upgrade headaches"
    `~/.cache/pip` and `~/.cache/uv` can occasionally serve stale wheels for editable installs. `pip cache purge` / `uv cache clean` if you see weirdness.

## Bottom line

For dependency management:

- **uv for new projects** — fast, modern, well-supported.
- **pip-tools for legacy migration** — minimal change.
- **Commit lockfiles** always.
- **CI installs from lockfile**, not spec.
- **Weekly/monthly upgrade cadence** with review.
- **Workspaces** for monorepos (uv has best support).

Continue to **[Publishing to PyPI and private indexes](08-publishing.md)**.
