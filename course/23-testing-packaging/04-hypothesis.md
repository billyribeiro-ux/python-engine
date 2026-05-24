# Property-based testing with Hypothesis

Example-based tests check that your function returns the right answer for specific inputs. **Property-based tests** check that a property holds for *every* input — by generating thousands of random inputs and checking the property on each.

[Hypothesis](https://hypothesis.readthedocs.io/) is the Python library for this. It finds bugs traditional tests miss because it generates inputs you wouldn't think of: empty strings, Unicode edge cases, very large numbers, NaN, etc.

## The shape

```python
from hypothesis import given, strategies as st


@given(st.integers())
def test_double_is_even(x):
    assert (x * 2) % 2 == 0


@given(st.lists(st.integers()))
def test_sort_is_idempotent(xs):
    assert sorted(sorted(xs)) == sorted(xs)


@given(st.text())
def test_encode_decode_roundtrip(s):
    assert s.encode("utf-8").decode("utf-8") == s
```

Each test runs for ~100 generated inputs by default. Hypothesis finds the minimal failing input ("shrinking") and reports it.

## Strategies

Strategies are recipes for generating values:

| Strategy | Generates |
|---|---|
| `st.integers()` | any int |
| `st.integers(min_value=0, max_value=100)` | bounded ints |
| `st.floats(allow_nan=False, allow_infinity=False)` | finite floats |
| `st.text()` | any Unicode string |
| `st.text(alphabet="abc", min_size=1, max_size=5)` | restricted strings |
| `st.lists(st.integers(), min_size=1, max_size=10)` | bounded lists |
| `st.tuples(st.integers(), st.text())` | (int, str) tuples |
| `st.dictionaries(st.text(), st.integers())` | dicts |
| `st.sampled_from(["a", "b", "c"])` | one of a fixed set |
| `st.datetimes()` | datetime objects |
| `st.dates()` | date objects |
| `st.booleans()` | True / False |
| `st.none()` | None |
| `st.one_of(a, b, c)` | union — any of these |

## Composing strategies

For complex types, compose:

```python
from hypothesis import strategies as st


# A "valid email-ish" strategy
emails = st.tuples(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
    st.sampled_from(["example.com", "test.org", "x.io"]),
).map(lambda t: f"{t[0]}@{t[1]}")


@given(emails)
def test_email_has_at(email):
    assert "@" in email
```

`.map(f)` transforms a generated value. `.filter(pred)` keeps only values matching a predicate (use sparingly — filtering rejects a lot of generated examples).

## `@composite` for stateful generation

When strategies depend on each other:

```python
from hypothesis import strategies as st


@st.composite
def order_and_partial_fill(draw):
    total_qty = draw(st.integers(min_value=1, max_value=10000))
    filled_qty = draw(st.integers(min_value=0, max_value=total_qty))
    return {"total": total_qty, "filled": filled_qty}


@given(order_and_partial_fill())
def test_fill_le_total(order):
    assert order["filled"] <= order["total"]
```

The `draw` function pulls from a sub-strategy. Lets you express "a value, then a related value derived from it."

## Stateful testing

For testing classes with mutable state:

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant


class OrderBookStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.book = OrderBook()

    @rule(price=st.floats(min_value=1, max_value=1000),
          qty=st.integers(min_value=1, max_value=1000))
    def add_bid(self, price, qty):
        self.book.add_bid(price, qty)

    @rule(price=st.floats(min_value=1, max_value=1000),
          qty=st.integers(min_value=1, max_value=1000))
    def add_ask(self, price, qty):
        self.book.add_ask(price, qty)

    @invariant()
    def best_bid_below_best_ask(self):
        bb, ba = self.book.best_bid, self.book.best_ask
        if bb is not None and ba is not None:
            assert bb[0] < ba[0], f"crossed book: bid={bb} ask={ba}"


TestOrderBook = OrderBookStateMachine.TestCase
```

Hypothesis generates random sequences of rule calls; checks invariants after each step; if any fail, shrinks to the minimal sequence that breaks the invariant. Astonishingly good at finding state machine bugs.

## Settings

For slow tests or expensive properties:

```python
from hypothesis import given, settings


@given(st.integers())
@settings(max_examples=500, deadline=None)
def test_thing(x):
    expensive_check(x)
```

| Setting | Meaning |
|---|---|
| `max_examples=N` | how many examples to try (default 100) |
| `deadline=None` | disable per-example timeout |
| `derandomize=True` | deterministic; same inputs every run |
| `print_blob=True` | print a blob to reproduce failures |

## Shrinking — the killer feature

When Hypothesis finds a failing example, it doesn't stop. It shrinks: tries to find the *minimal* input that still triggers the failure. So instead of "fails on `[27, 14, 88, 33, ..., 901]`," you get "fails on `[0]`."

Shrinking is what makes property-based tests actionable: the reported failure is something you can reason about.

## A worked example: a sorting bug

```python
def buggy_sort(xs):
    if not xs:
        return []
    # Bug: doesn't handle duplicates correctly
    return sorted(set(xs))


@given(st.lists(st.integers()))
def test_sort_preserves_length(xs):
    assert len(buggy_sort(xs)) == len(xs)
```

Run: `pytest test_sort.py`. Hypothesis quickly finds `xs=[0, 0]` — the minimal input that breaks. Without `set`, the test would pass. The bug is found in seconds; the minimal example tells you exactly where to look.

## When property-based testing shines

- **Pure functions** with clear properties (encoders / decoders roundtrip, sort is idempotent, parsers reject malformed input).
- **Stateful systems** where invariants should hold across any sequence of operations.
- **Edge-case discovery** — Hypothesis explores empty/large/zero/negative/Unicode/NaN combinations you'd never write.

## When it's overkill

- Simple CRUD code with no interesting properties.
- Code where "I just need to know this specific real-world input works" — example-based is fine.
- Integration tests against external services — non-determinism breaks Hypothesis's shrinking.

## Pitfalls

!!! warning "Properties that aren't actually properties"
    `assert function(x) > 0` for a function that's defined to return non-negative is wrong. Be precise.

!!! warning "Filters that reject most generated inputs"
    `st.integers().filter(lambda x: x % 7 == 0)` rejects 6 of every 7 examples. Use a `@composite` strategy that generates valid values directly instead.

!!! warning "Stateful test machines that don't reset"
    The state machine class is reset between test sequences automatically. Don't store class-level state.

!!! warning "Hypothesis tests in CI without seed pinning"
    Hypothesis is intentionally non-deterministic. CI runs may find a new failure that doesn't reproduce locally. Always log the seed; use `--hypothesis-seed=N` to reproduce.

## Bottom line

For Hypothesis:

- Use it for **pure functions with clear properties**.
- **Stateful test machines** find concurrency / order-of-operation bugs.
- **`@composite`** for related generated values.
- **Shrinking** gives you minimal failing examples; reproduce with `--hypothesis-seed`.

Continue to **[Coverage and mutation testing](05-coverage-and-mutation.md)**.
