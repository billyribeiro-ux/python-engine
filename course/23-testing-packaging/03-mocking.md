# Mocking and patching

A unit test isolates the code under test from its dependencies. When the dependency is an external service, a database, or anything with side effects, you replace it with a stand-in: a **mock**. This chapter is the working `unittest.mock` toolkit and the discipline that keeps mocks from making tests useless.

## `Mock` and `MagicMock`

```python
from unittest.mock import Mock, MagicMock

m = Mock()
m.foo                          # auto-creates an attribute (also a Mock)
m.foo("bar")                   # callable; records the call
m.foo.assert_called_with("bar")
m.foo.call_args                # ((args,), {kwargs})
m.foo.call_count               # 1
m.foo.return_value = 42        # next call returns 42
print(m.foo("x"))              # 42
```

`MagicMock` is `Mock` with magic methods (`__iter__`, `__getitem__`, etc.) auto-configured. Default to `MagicMock` unless you have a reason not to.

## Side effects

```python
m = Mock()
m.fetch.side_effect = [1, 2, 3]      # successive calls return these
m.fetch()  # 1
m.fetch()  # 2

m.crash.side_effect = ValueError("bad input")
m.crash()  # raises ValueError

# Or a callable
m.compute.side_effect = lambda x: x * 2
m.compute(5)  # 10
```

`side_effect` overrides `return_value`. The right tool for "different result each call" or "raise an exception."

## `patch` — temporarily replace something

The most common pattern: replace `requests.get` (or any module-level callable) with a mock for the duration of the test.

```python
from unittest.mock import patch


def real_function():
    import requests
    return requests.get("https://api.example.com/data").json()


@patch("requests.get")
def test_real_function(mock_get):
    mock_get.return_value.json.return_value = {"status": "ok"}
    result = real_function()
    assert result == {"status": "ok"}
    mock_get.assert_called_once_with("https://api.example.com/data")
```

Three patterns:

```python
# As a decorator
@patch("mymodule.dependency")
def test_x(mock_dep): ...

# As a context manager
def test_x():
    with patch("mymodule.dependency") as mock_dep:
        ...

# Programmatic, with cleanup via fixture
def test_x(monkeypatch):
    monkeypatch.setattr("mymodule.dependency", Mock())
```

The third is what pytest users actually use most of the time — `monkeypatch` from chapter 2 is a thin wrapper.

## Patch the *call site*, not the source

The single biggest `patch` gotcha:

```python
# mymodule.py
from external import fetch_data

def process():
    return fetch_data() * 2
```

Wrong:
```python
@patch("external.fetch_data")     # NO — patches the source
def test_process(mock_fetch):
    ...
```

Right:
```python
@patch("mymodule.fetch_data")     # YES — patches where it's used
def test_process(mock_fetch):
    ...
```

`process()` looked up `fetch_data` in its own module's namespace, which already has a bound reference. Patching the source doesn't change that reference. Always patch the place that imports.

## `autospec` — patch with the real signature

```python
@patch("mymodule.send_email", autospec=True)
def test_send_email(mock_send):
    send_email("to@x.com", "subj", "body")
    mock_send.assert_called_with("to@x.com", "subj", "body")

    # If your code calls with the wrong signature, this catches it:
    send_email("only_one_arg")
    # TypeError: missing required arguments
```

Without `autospec`, a `Mock` accepts any signature. With `autospec=True`, it mirrors the real function's signature — refactor-safe.

**Always use `autospec=True` for module-level patches** unless you have a specific reason not to. The cost is one keyword argument; the benefit is catching signature drift in tests.

## `pytest-mock`

Wraps `unittest.mock` in a pytest fixture for cleaner syntax:

```python
def test_thing(mocker):
    mock_fetch = mocker.patch("mymodule.fetch_data", autospec=True)
    mock_fetch.return_value = {"x": 1}
    assert process() == {"x": 1, "y": 2}
```

`mocker` is automatically scoped to the test (no decorator, no `with`). For most pytest codebases this is the cleanest form.

## Mocking classes

```python
@patch("mymodule.DatabaseClient", autospec=True)
def test_uses_db(MockDB):
    instance = MockDB.return_value           # the instance the class returns
    instance.query.return_value = [{"id": 1}]
    result = my_function_under_test()
    assert result == [{"id": 1}]
    instance.query.assert_called_with("SELECT ...")
```

When you patch a *class*, `MockDB.return_value` is the mock instance. `MockDB()` in the code under test returns it.

## When NOT to mock

The temptation is to mock anything that's not a pure function. The cost: tests become coupled to *how* the code calls dependencies, not what it actually does. Refactor the implementation, all tests break.

Guidelines:

1. **Don't mock the standard library.** `os.path.join`, `datetime.now`, `random` — usually let them run. If you need a specific value (current date), use `freezegun` instead.
2. **Don't mock the code you're testing.** If a function's behaviour depends on what it returns, you're really testing the mock.
3. **Mock at boundaries.** Network calls, database connections, filesystem state, external services — these are the right things to mock.
4. **Prefer fakes over mocks** for complex dependencies. A `FakeDatabase` class that implements the same interface but in-memory often beats `MagicMock` for clarity.

## A worked example: testing a webhook handler

```python
# myapp/webhooks.py
import requests


def process_webhook(payload):
    if payload.get("type") == "order":
        requests.post("https://internal/order", json=payload)
        return "queued"
    return "ignored"
```

```python
# tests/test_webhooks.py
import pytest
from unittest.mock import call


def test_order_webhook_posts(mocker):
    mock_post = mocker.patch("myapp.webhooks.requests.post", autospec=True)
    result = process_webhook({"type": "order", "id": 42})
    assert result == "queued"
    mock_post.assert_called_once_with("https://internal/order", json={"type": "order", "id": 42})


def test_non_order_webhook_ignored(mocker):
    mock_post = mocker.patch("myapp.webhooks.requests.post", autospec=True)
    result = process_webhook({"type": "other"})
    assert result == "ignored"
    mock_post.assert_not_called()
```

Two clear tests, mocking only the external dependency, asserting the visible behaviour.

## `freezegun` for time

```python
from freezegun import freeze_time
from datetime import datetime


@freeze_time("2024-11-15 14:30:00")
def test_with_frozen_time():
    assert datetime.now() == datetime(2024, 11, 15, 14, 30)
```

Better than mocking `datetime.now()` directly — works across the entire stdlib.

## `responses` and `respx` — for HTTP

For testing code that uses `requests`:

```python
import responses


@responses.activate
def test_api_call():
    responses.add(responses.GET, "https://api.example.com/data",
                  json={"status": "ok"}, status=200)
    result = my_function_that_calls_api()
    assert result == {"status": "ok"}
```

For `httpx`, `respx`:

```python
import respx


@respx.mock
def test_httpx_call():
    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    result = my_function()
```

Cleaner than hand-mocking `httpx.AsyncClient.get` — `respx` understands HTTP semantics.

## Pitfalls

!!! danger "Asserting on `call_count` instead of `assert_called_*`"
    `m.call_count == 1` is fine but `m.assert_called_once_with(...)` is better — it tells you not just the count but the expected args.

!!! danger "Mock.method that doesn't exist on the real object**
    Without `autospec`, `mock.nonexistent_method()` silently succeeds. With `autospec=True`, it raises `AttributeError`.

!!! danger "Patching across processes**
    `multiprocessing` worker processes don't see your patches. If the code spawns subprocesses, you can't mock them with `unittest.mock` — you need to inject the dependency.

## Bottom line

For mocking:

- **`pytest-mock`** (the `mocker` fixture) for the cleanest pytest experience.
- **`autospec=True`** always.
- **Patch the call site**, not the source.
- **Mock at boundaries**; not internal pure functions.
- **`freezegun`** for time; **`responses` / `respx`** for HTTP.

Continue to **[Property-based testing with Hypothesis](04-hypothesis.md)**.
