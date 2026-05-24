# pytest fundamentals

`pytest` is the de facto standard. `unittest` is in the stdlib but its API is verbose and inheritance-heavy; pytest is what people actually use. This chapter is the working basics.

## The minimum-viable test file

```python
# tests/test_simple.py
def test_addition():
    assert 1 + 1 == 2


def test_string():
    assert "py" + "thon" == "python"
```

Run: `pytest`. No imports, no class wrapper, no `setUp`. The discovery rules: any file named `test_*.py` or `*_test.py`; any function named `test_*`.

## Plain assert, with rich diffs

pytest rewrites `assert` statements so failures show meaningful diffs:

```python
def test_dict():
    expected = {"name": "alice", "age": 30, "city": "NYC"}
    actual = {"name": "alice", "age": 31, "city": "NYC"}
    assert actual == expected
# AssertionError: assert {'age': 31, ...} == {'age': 30, ...}
#   Differing items:
#   {'age': 31} != {'age': 30}
```

Use plain `assert`. Don't use `self.assertEqual(...)` from unittest.

## Project layout

```
your-project/
├── pyproject.toml
├── src/
│   └── mypackage/
│       └── __init__.py
└── tests/
    ├── __init__.py
    └── test_things.py
```

`src/` layout is the modern best practice — prevents importing the source from the project root (which masks packaging bugs). In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
```

## Parametrising

For testing the same logic across many inputs:

```python
import pytest


@pytest.mark.parametrize("inp,expected", [
    ("hello", 5),
    ("", 0),
    ("hi", 2),
    ("Привет", 6),
])
def test_length(inp, expected):
    assert len(inp) == expected
```

Each row runs as a separate test (`test_length[hello-5]`, etc.). Failures point to the exact row.

For Cartesian products:

```python
@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", ["x", "y"])
def test_pair(a, b):
    assert (a, b) in product([1,2,3], ["x","y"])
```

6 tests from 3×2.

## Markers

Tag tests for selective execution:

```python
import pytest


@pytest.mark.slow
def test_long_running():
    ...


@pytest.mark.network
def test_pulls_from_api():
    ...
```

Run only fast tests: `pytest -m "not slow"`. Run network tests: `pytest -m network`.

Register markers in `pyproject.toml` to avoid warnings:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: tests that take > 1 second",
    "network: tests that require internet",
]
```

`--strict-markers` (recommended) fails on unregistered marker names. Catches typos.

## Expected exceptions

```python
import pytest


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_error_message():
    with pytest.raises(ValueError, match="negative"):
        sqrt(-1)
```

`match=` is a regex against the exception message. More precise than just catching the type.

## Floating point

```python
from pytest import approx


def test_almost_equal():
    assert 0.1 + 0.2 == approx(0.3)
    assert 0.1 + 0.2 == approx(0.3, rel=1e-9)
    assert [0.1+0.2, 1+1] == approx([0.3, 2.0])
```

`approx` works on scalars, lists, dicts, NumPy arrays. Always use it for float comparisons.

## Capturing stdout/stderr

```python
def test_prints_hello(capsys):
    print("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
```

`capsys` is a built-in fixture. Similar: `capfd` (file descriptor level — catches subprocess output too), `caplog` (captures logging).

## Tmp dirs

```python
def test_writes_file(tmp_path):
    target = tmp_path / "out.txt"
    write_something(target)
    assert target.read_text() == "expected"
```

`tmp_path` is a `pathlib.Path` to a unique temp dir per test. Cleaned up automatically.

## A working `conftest.py`

Project-wide fixtures and config live in `conftest.py`:

```python
# tests/conftest.py
import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sample_dataframe():
    import pandas as pd
    return pd.DataFrame({"x": range(100), "y": [i * 2 for i in range(100)]})
```

Any test in any file can use `def test_thing(rng, sample_dataframe):` — fixture resolution is automatic.

## Useful CLI flags

```bash
pytest -x                 # stop at first failure
pytest --lf               # only re-run last failed
pytest --ff               # failed first, then the rest
pytest -k "scanner and not slow"   # filter by name
pytest -m "not network"   # filter by marker
pytest -v                 # verbose: show each test name
pytest -s                 # don't capture stdout (see prints)
pytest --tb=short         # short tracebacks
pytest -p no:cacheprovider   # disable .pytest_cache
pytest -n 8               # parallel (needs pytest-xdist)
```

`-x --lf` is what you want during debugging. `-q -n auto` is what CI wants.

## Test organisation rules

1. **One concept per test.** A test that checks five things makes the failure ambiguous.
2. **Arrange / Act / Assert.** Set up state; call the code; check the result. In that order.
3. **No conditional logic** inside tests (no `if`/`else`). If you need branches, parametrise.
4. **Tests run in any order.** No shared state between tests. Don't rely on `test_a` running before `test_b`.
5. **Tests run in isolation.** No filesystem state between tests (use `tmp_path`). No DB state between tests (use a fresh fixture per test).

## What NOT to do

!!! danger "Asserting against implementation details"
    `assert mock_db.calls[3].arg == "x"` couples the test to refactoring. Test behaviour, not implementation.

!!! danger "Tests that depend on the network without marking"
    A test that hits Yahoo will silently fail offline. Mark `@pytest.mark.network`; skip in default runs.

!!! danger "Massive setUp / fixture chains"
    A 200-line fixture means the test is testing something else entirely. Split the production code instead.

## Bottom line

For real pytest:

- **Plain `assert`** + rich diffs.
- **`src/` layout** for import hygiene.
- **`@pytest.mark.parametrize`** for the same logic over many inputs.
- **Markers + `--strict-markers`** for selective runs.
- **`pytest.raises` and `approx`** for exceptions and floats.
- **`conftest.py`** for project-wide fixtures.

Continue to **[Fixtures and dependency injection](02-fixtures.md)**.
