# Coverage and mutation testing

"100% test coverage" is a common goal. It is also a worse metric than people think. **Line coverage** tells you which lines ran during tests; it tells you nothing about whether the assertions actually check anything meaningful. **Mutation testing** is the harder, more honest metric: it modifies your code in small ways and checks that *some test fails*.

## `coverage.py`

```bash
pip install coverage pytest-cov
pytest --cov=mypackage --cov-report=term-missing
```

Output:

```
Name              Stmts   Miss  Cover   Missing
-------------------------------------------------
mypackage/__init__.py   0      0   100%
mypackage/core.py      85      4    95%   42, 58-60
mypackage/utils.py     32      8    75%   12-19
-------------------------------------------------
TOTAL                 117     12    90%
```

`Missing` is the line numbers not exercised by any test. Open those files and write tests for them.

For HTML reports (much easier to navigate):

```bash
pytest --cov=mypackage --cov-report=html
open htmlcov/index.html
```

## Branch coverage

Line coverage misses something subtle: a line with `if x:` is "covered" if the `if` branch ran, even though the `else` didn't. Branch coverage:

```toml
# pyproject.toml
[tool.coverage.run]
branch = true
source = ["mypackage"]


[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

Now `if x:` is only fully covered when both branches have been exercised.

## What coverage cannot tell you

```python
def divide(a, b):
    return a / b


def test_divide():
    divide(10, 2)         # runs the line; 100% coverage; no assertion
```

100% coverage, zero useful tests. Coverage measures *execution*, not *verification*.

The mitigation: review your own tests. A test without `assert` (or equivalent) is decorative. CI lint rules can enforce "no test function lacks an assert," but the discipline is humans.

## Mutation testing

[`mutmut`](https://github.com/boxed/mutmut) (Python's leading mutation tool) makes small modifications to your code:

- `x > 0` → `x >= 0`
- `x + y` → `x - y`
- `return value` → `return None`

For each "mutant," it runs your tests. If a test fails, the mutant was *killed* (good — your tests caught the change). If all tests pass, the mutant *survived* (bad — your tests don't catch that change, which means they don't really verify the original code).

```bash
pip install mutmut
mutmut run
mutmut results
```

Typical run output:

```
Killed mutants: 145
Survived mutants: 32
Mutation score: 145 / 177 = 82%
```

The 32 surviving mutants are the cases where your code can be modified without any test failing — i.e., where your tests don't actually check what you think they check.

## Interpreting mutation testing

For each surviving mutant:

- **`mutmut show <id>`** — display the modification.
- Decide: is this a real gap in tests, or is the mutation semantically equivalent (e.g., `+ 0` → `* 1`)?

The pragmatic rules:

- **80%+ mutation score on critical paths** (pricing, order routing, risk checks) is the right target.
- **50-70% on everything else** is fine — covering the last 20% takes diminishing-return effort.
- **100% mutation score is suspicious** — either trivial code or you're testing implementation details.

## Coverage in CI

A common pattern:

```toml
[tool.coverage.report]
fail_under = 80      # CI fails if coverage drops below 80%
```

Or in CI:

```yaml
- name: Test with coverage
  run: pytest --cov=mypackage --cov-fail-under=80
```

Set the floor; never let it slip. *Don't* require 100% — it forces tests for trivial getters and config code, where the test cost exceeds the bug-prevention benefit.

## What to test

The 80/20 of test value:

1. **The complicated logic.** Branch-heavy functions. Loops with multiple exit conditions. State machines.
2. **Boundary conditions.** Empty input. Single-element input. Max-size input. Off-by-one boundaries.
3. **Failure modes.** What happens when the network fails? When the DB is locked? When the file doesn't exist?
4. **Critical-path business logic.** Pricing, sizing, order construction.

Don't test:

- Trivial getters / setters.
- Third-party library behaviour (the library has its own tests).
- Implementation details of your own internal helpers.

## Tests as documentation

Well-named tests serve as living documentation:

```python
def test_cancel_returns_unfilled_quantity_to_available_inventory():
    ...


def test_partial_fill_during_cancel_leaves_correct_realised_pnl():
    ...
```

Read the test names; understand the system. Better than half the docstrings.

## Pitfalls

!!! warning "Chasing 100% coverage"
    The last 5% of coverage often comes from trivial branches, type-check stubs, and defensive code that's basically untestable. Spending two hours to cover three lines makes the test suite slower without making the code safer.

!!! warning "Tests that mock everything"
    A 100%-covered codebase where every dependency is mocked tells you nothing about how the whole system works. Mix unit tests with end-to-end / integration tests.

!!! warning "Mutation testing is slow"
    `mutmut` runs your full test suite per mutant. A 500-test suite × 1000 mutants × 0.1 s = 14 hours. Run it on a schedule, not on every push.

!!! warning "Snapshot tests overused"
    Snapshot/golden-file tests are easy to update without thinking. Make sure updates are deliberate, not "the test says my output changed; I'll just update the snapshot."

## Bottom line

For coverage and mutation:

- **Branch coverage** in CI, with a sensible floor (75-85%).
- **Mutation testing periodically** (weekly?) on critical-path code.
- **Test naming as documentation**.
- **Skip coverage of trivial code** — the test cost isn't worth it.

Continue to **[pyproject.toml and modern build systems](06-pyproject.md)**.
