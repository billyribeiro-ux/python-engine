# pyproject.toml and modern build systems

`setup.py` is dead. `pyproject.toml` (PEP 518 / 517 / 621) is the modern way to configure a Python project. This chapter is the working `pyproject.toml` you can copy, plus how the build backends differ.

## The minimum

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"


[project]
name = "mypackage"
version = "0.1.0"
description = "Short description"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["keyword1", "keyword2"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]


dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
]


[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "black>=24.4"]
docs = ["mkdocs-material>=9.5"]


[project.urls]
Homepage = "https://github.com/you/mypackage"
Documentation = "https://docs.example.com"
Issues = "https://github.com/you/mypackage/issues"
```

Three sections matter:

1. **`[build-system]`** — what builds the package. `hatchling`, `setuptools`, `flit_core`, `poetry-core` are all valid; `hatchling` is the modern default.
2. **`[project]`** — metadata exposed to PyPI, pip, and discovery tools.
3. **Optional extras** — `pip install mypackage[dev]`.

## Choosing a build backend

| Backend | Best for | Quirks |
|---|---|---|
| **hatchling** | new projects; clean, fast, well-documented | the default recommendation |
| **setuptools** | legacy projects with custom build steps | wider plugin ecosystem; more config |
| **flit_core** | tiny pure-Python packages | minimal; no extension modules |
| **poetry-core** | projects using Poetry as workflow tool | tied to Poetry CLI |
| **maturin** | Rust extensions via PyO3 | the standard for Rust+Python |

For new pure-Python work: **hatchling**. For Cython / C extensions: **setuptools** (or scikit-build-core for CMake-based builds). For Rust: **maturin**.

## Where the code lives — `src/` layout

```
mypackage/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
└── docs/
```

The `src/` layout has one big advantage: you **cannot** import the source from the project root by accident. `pip install -e .` is the only way to use it. That means your tests exercise the *installed* package, not the source layout. Catches packaging bugs early.

For `hatchling`, `src/` is detected automatically. For `setuptools`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

## Dynamic version from `__version__` or git

Three patterns:

### From a `__version__` in code

```toml
[project]
name = "mypackage"
dynamic = ["version"]

[tool.hatch.version]
path = "src/mypackage/__init__.py"
```

```python
# src/mypackage/__init__.py
__version__ = "0.1.0"
```

### From git tags via `hatch-vcs`

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]


[project]
dynamic = ["version"]


[tool.hatch.version]
source = "vcs"
```

Now `0.1.0` comes from your `v0.1.0` git tag. CI tags → release on PyPI; no manual version bumping.

## Entry points

For installing a CLI tool:

```toml
[project.scripts]
mytool = "mypackage.cli:main"
```

After `pip install`, running `mytool` invokes `mypackage.cli.main()`. The standard pattern for command-line tools.

For plugin systems:

```toml
[project.entry-points."mypackage.plugins"]
my_plugin = "myplugin:hook"
```

`myplugin` registers itself as `my_plugin` under the `mypackage.plugins` group. Your package discovers it via `importlib.metadata.entry_points`.

## Local tool config in `pyproject.toml`

Most modern Python tools read their config from `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]


[tool.black]
line-length = 100
target-version = ["py311"]


[tool.mypy]
strict = true
ignore_missing_imports = false


[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"


[tool.coverage.run]
branch = true
source = ["src/mypackage"]
```

One file; all tool configs. No more `.coveragerc`, `pytest.ini`, `mypy.ini`, `.flake8` scattered around.

## Building and installing

```bash
# Build (creates dist/*.whl and dist/*.tar.gz)
python -m build

# Install locally for development (editable)
pip install -e .[dev]

# Install from local wheel
pip install dist/mypackage-0.1.0-py3-none-any.whl

# Install from git
pip install git+https://github.com/you/mypackage.git@v0.1.0
```

`pip install -e .` makes your source directly importable; changes to files take effect immediately, no reinstall.

## Including non-code files

By default, only `.py` files are included. For data files (templates, schema files, JSON):

```toml
[tool.hatch.build]
include = [
    "src/mypackage/**/*.py",
    "src/mypackage/templates/*.html",
    "src/mypackage/schemas/*.json",
]
```

For setuptools:

```toml
[tool.setuptools.package-data]
"mypackage" = ["templates/*.html", "schemas/*.json"]
```

Both work; both achieve the same thing.

## A complete real example

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"


[project]
name = "trading-engine"
description = "Internal trading engine"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Proprietary" }
authors = [{ name = "Quant Team", email = "quant@example.com" }]
dynamic = ["version"]


dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "sqlalchemy>=2.0",
    "httpx>=0.27",
]


[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "hypothesis>=6.0",
    "ruff>=0.5",
    "black>=24.4",
    "mypy>=1.10",
]
options = ["py_vollib>=1.0"]


[project.scripts]
trade = "trading_engine.cli:main"
backfill = "trading_engine.scripts.backfill:main"


[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.sdist]
include = ["src/trading_engine", "tests"]


[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]


[tool.black]
line-length = 100


[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"


[tool.coverage.run]
branch = true
source = ["src/trading_engine"]


[tool.mypy]
strict = true
```

That's a complete, modern Python project file. Copy, modify, ship.

## Pitfalls

!!! warning "Forgetting `[build-system]`"
    Without it, pip uses ancient setuptools defaults. Always specify.

!!! warning "Editable installs with namespace packages"
    `pip install -e .` and PEP 420 namespace packages have historically been finicky. For new projects use a regular package (with `__init__.py`).

!!! warning "Version in two places"
    Static version in `pyproject.toml` AND `__init__.py` drifts. Use `dynamic = ["version"]` to read it from one place.

!!! warning "Treating `extras_require` as runtime deps"
    `pip install mypackage` does NOT install `[dev]` extras. If a dep is needed at runtime, it's not optional.

## Bottom line

For modern packaging:

- **`pyproject.toml` only.** No `setup.py`, no `setup.cfg`.
- **`hatchling`** as the build backend for new projects.
- **`src/` layout** for import hygiene.
- **`dynamic = ["version"]`** from `__init__.py` or git tags.
- **`[project.scripts]`** for CLI tools.
- **All tool configs in one file**.

Continue to **[Dependency management](07-dependency-management.md)**.
