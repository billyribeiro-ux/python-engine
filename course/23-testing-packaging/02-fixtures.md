# Fixtures and dependency injection

pytest's fixture system is its killer feature. A fixture is a reusable piece of test setup, *injected* into tests that ask for it by parameter name. Once you internalise the model, your tests become smaller, more isolated, and easier to refactor.

## Defining and using a fixture

```python
import pytest


@pytest.fixture
def temp_database():
    """A fresh in-memory SQLite DB per test."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
    yield conn
    conn.close()


def test_insert(temp_database):
    temp_database.execute("INSERT INTO t VALUES (1, 'a')")
    rows = list(temp_database.execute("SELECT * FROM t"))
    assert rows == [(1, "a")]
```

Before `yield` is setup; after `yield` is teardown. The test gets exactly what's yielded.

## Scopes

By default, each test gets a fresh fixture instance. For expensive setup, broaden the scope:

```python
@pytest.fixture(scope="function")   # default — once per test
def per_test():
    return create()


@pytest.fixture(scope="class")      # once per test class
def per_class():
    return create()


@pytest.fixture(scope="module")     # once per test file
def per_module():
    return create()


@pytest.fixture(scope="session")    # once per pytest run
def per_session():
    return create()
```

Rule of thumb: **`function` is safe; everything else risks state leaking between tests.** Use broader scopes only for genuinely-read-only or genuinely-expensive resources (a started database server, a downloaded model).

## Parametrised fixtures

Run the same test against many fixture values:

```python
@pytest.fixture(params=["sqlite", "postgres"])
def db_backend(request):
    if request.param == "sqlite":
        return create_sqlite()
    return create_postgres()


def test_insert(db_backend):
    # Runs twice: once with each backend
    db_backend.insert(...)
```

Each `params` value produces one test invocation. Useful for testing the same logic against multiple implementations.

## Fixture factories

A fixture that *returns a function* lets you generate parameterised values inside the test:

```python
@pytest.fixture
def make_user():
    def _make(name="alice", age=30):
        return User(name=name, age=age)
    return _make


def test_admin(make_user):
    u = make_user(name="root", age=99)
    assert u.name == "root"
```

The pattern is "fixture as builder." Reduces boilerplate when each test needs slight variations.

## Built-in fixtures

The big ones:

| Fixture | Purpose |
|---|---|
| `tmp_path` | unique temp dir per test (Path) |
| `tmp_path_factory` | session-scoped temp dir factory |
| `monkeypatch` | safe attribute / env-var mutation, auto-restored |
| `capsys`, `capfd` | capture stdout/stderr |
| `caplog` | capture logging |
| `request` | introspection on the current test |
| `pytestconfig` | access the pytest config object |

### `monkeypatch` in detail

For temporarily changing env vars, attributes, or even objects:

```python
def test_with_fake_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "fake")
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.setattr("mymodule.client", FakeClient())
    monkeypatch.setattr(SomeClass, "method", lambda self: "stubbed")
    # All changes auto-revert at end of test
```

This is the right tool for "I need to make X look like Y for this one test." Don't reach for global mutation.

## conftest.py — fixture sharing

A `conftest.py` file in `tests/` (or any subdirectory) defines fixtures available to all tests in that directory and below:

```python
# tests/conftest.py
import pytest


@pytest.fixture
def feed_with_fixture_data():
    """A Feed stub with deterministic data."""
    return StubFeed(fixture_path="tests/fixtures/spy_2024.parquet")
```

```python
# tests/strategies/conftest.py
@pytest.fixture
def momentum_signal(feed_with_fixture_data):
    """Pre-computed momentum on the fixture data."""
    return compute_momentum(feed_with_fixture_data.bars("SPY"))
```

Fixture resolution walks up: each test sees fixtures from its own file, then conftests up the directory tree.

## Indirect parametrisation

Pass parameters to fixtures via `indirect`:

```python
@pytest.fixture
def loaded_data(request):
    path = request.param
    return load(path)


@pytest.mark.parametrize("loaded_data", ["small.csv", "medium.csv", "huge.csv"], indirect=True)
def test_process(loaded_data):
    assert loaded_data.is_valid()
```

The string values get passed to the fixture (not directly to the test). The fixture decides what to do with them. Powerful for "same test, multiple data files."

## Fixtures that yield resources

For anything needing cleanup (connections, sockets, processes):

```python
@pytest.fixture
def running_server():
    server = subprocess.Popen(["myserver", "--port", "12345"])
    time.sleep(0.5)
    yield "http://localhost:12345"
    server.terminate()
    server.wait(timeout=5)
```

`yield` is the cleanup boundary. Always close / terminate / cleanup *after* the yield, even if the test fails.

## Anti-patterns

!!! danger "Mutable session-scoped fixtures"
    A session-scoped fixture that returns a list — and tests append to it — silently breaks test isolation. Either freeze the fixture (return a tuple) or use function scope.

!!! danger "Fixture chains 5 deep"
    A fixture that depends on another that depends on another that depends on yet another. Debugging is painful. Flatten when possible.

!!! danger "Side effects in fixture functions"
    A fixture that "logs in" via a real API and stores the token in env vars affects other tests. Keep fixtures pure where possible.

!!! danger "Using `autouse=True` casually"
    `@pytest.fixture(autouse=True)` makes the fixture run for every test in scope without being requested. Reserve for genuine global setup (clearing caches, resetting random seeds).

## A realistic example

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def fixed_random():
    """A deterministic RNG, session-scoped because it's immutable once created."""
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_bars(fixed_random):
    """A 100-day synthetic price series. Fresh per test."""
    n = 100
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(
        100 * np.exp(np.cumsum(fixed_random.normal(0, 0.01, n))),
        index=idx,
        name="close",
    )


# tests/test_returns.py
def test_returns_shape(synthetic_bars):
    rets = synthetic_bars.pct_change()
    assert len(rets) == 100
    assert rets.iloc[0] != rets.iloc[0]   # NaN at start
```

Two fixtures, one test using both. Each test gets a fresh `synthetic_bars` (different per test because `pct_change` is benign but a stateful test would mutate it) backed by the same RNG.

## Bottom line

For pytest fixtures:

- **Function-scoped by default**; broaden only for genuinely expensive or read-only resources.
- **`yield` for cleanup**; runs even on test failure.
- **`monkeypatch`** for safe temporary mutation.
- **`conftest.py`** for project-wide sharing.
- **Parametrise + factory fixtures** for the same logic across many inputs.

Continue to **[Mocking and patching](03-mocking.md)**.
