# Regex mastery

`re` is the tool you reach for when text needs to be parsed but doesn't deserve a real parser. Use it judiciously — overuse leads to unmaintainable patterns; underuse means hand-written character-by-character loops. This chapter is the working patterns.

## The basics

```python
import re


# Find single match
m = re.search(r"\d{3}-\d{4}", "Call 555-1234 now")
if m:
    print(m.group())                                  # 555-1234

# Find all
matches = re.findall(r"\b\w+@\w+\.\w+\b", "alice@ex.com, bob@ex.net")
# ['alice@ex.com', 'bob@ex.net']

# Replace
clean = re.sub(r"\s+", " ", "lots   of\twhitespace\n")
# "lots of whitespace "

# Compile once if used many times
pattern = re.compile(r"^\d+$")
for line in lines:
    if pattern.match(line):
        ...
```

`search` finds anywhere; `match` matches from the start; `fullmatch` matches the entire string; `findall` returns all non-overlapping matches; `sub` replaces; `split` splits by pattern.

## Character classes — what you'll use

| | Matches |
|---|---|
| `\d` | digit (0-9) |
| `\D` | not digit |
| `\w` | word character (alphanumeric + underscore) |
| `\W` | not word |
| `\s` | whitespace |
| `\S` | not whitespace |
| `.` | any character except newline (by default) |
| `[abc]` | one of a, b, c |
| `[a-z]` | range |
| `[^abc]` | not a, b, or c |
| `[\d.]` | digit or literal `.` (inside `[]`, most metachars are literal) |

## Quantifiers

| | Means |
|---|---|
| `*` | 0 or more |
| `+` | 1 or more |
| `?` | 0 or 1 (optional) |
| `{n}` | exactly n |
| `{n,}` | n or more |
| `{n,m}` | between n and m |
| `*?` | non-greedy 0 or more |
| `+?` | non-greedy 1 or more |

The non-greedy versions matter:

```python
text = "<b>hi</b> and <i>there</i>"
re.findall(r"<.+>", text)                # ['<b>hi</b> and <i>there</i>']  — greedy
re.findall(r"<.+?>", text)               # ['<b>', '</b>', '<i>', '</i>']  — non-greedy
```

## Anchors

| | Means |
|---|---|
| `^` | start of string (or line, with `re.MULTILINE`) |
| `$` | end of string (or line, with `re.MULTILINE`) |
| `\b` | word boundary |
| `\B` | not a word boundary |

```python
re.findall(r"\bcat\b", "concatenate the cat")        # ['cat']  (not "concatenate")
```

## Groups and capturing

```python
m = re.search(r"(\d{4})-(\d{2})-(\d{2})", "today: 2024-11-15")
if m:
    print(m.group(0))                                 # 2024-11-15
    print(m.group(1), m.group(2), m.group(3))         # 2024 11 15
    print(m.groups())                                 # ('2024', '11', '15')
```

Named groups for readability:

```python
m = re.search(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})", text)
if m:
    print(m.group("year"))                            # 2024
    print(m.groupdict())                              # {'year': '2024', ...}
```

Non-capturing groups for "I want grouping but not a capture":

```python
re.findall(r"(?:Mr|Mrs|Ms)\. (\w+)", "Mr. Smith and Ms. Jones")
# ['Smith', 'Jones']  — title not captured
```

## Lookahead / lookbehind

For matching only when followed/preceded by something:

```python
# Positive lookahead — match "foo" only if followed by "bar"
re.findall(r"foo(?=bar)", "foobar foobaz")           # ['foo']  (one match)

# Negative lookahead
re.findall(r"foo(?!bar)", "foobar foobaz")           # ['foo']  (from foobaz)

# Positive lookbehind
re.findall(r"(?<=\$)\d+", "$100 and 50")             # ['100']

# Negative lookbehind
re.findall(r"(?<!\$)\d+", "$100 and 50")             # ['00', '50'] (yes, both!)
```

Lookarounds don't consume characters. Useful for context-sensitive matching.

## Flags

```python
re.findall(r"hello", "HELLO world", re.IGNORECASE)    # ['HELLO']
re.findall(r"^line", "line one\nline two", re.MULTILINE)
re.findall(r".+", "first\nsecond", re.DOTALL)         # '.' now matches newlines

# Inline flag
re.findall(r"(?i)hello", "HELLO")                     # ['HELLO']

# Verbose mode for readability
phone = re.compile(r"""
    \(?              # optional opening paren
    (\d{3})          # area code
    \)?              # optional closing paren
    [\s-]?           # optional separator
    (\d{3})          # exchange
    [\s-]?           # separator
    (\d{4})          # last 4
""", re.VERBOSE)
```

`re.VERBOSE` is essential for complex patterns. Without it, regex becomes write-only.

## Substitution with backreferences

```python
# Reformat date
re.sub(r"(\d{4})-(\d{2})-(\d{2})", r"\3/\2/\1", "2024-11-15")
# "15/11/2024"

# With named groups
re.sub(r"(?P<first>\w+) (?P<last>\w+)", r"\g<last>, \g<first>", "Alice Smith")
# "Smith, Alice"

# Substitution function
def upper_match(m):
    return m.group(0).upper()

re.sub(r"\b\w+\b", upper_match, "hello world")
# "HELLO WORLD"
```

## Splitting

```python
re.split(r"\s+", "  hello   world  ")                # ['', 'hello', 'world', '']
re.split(r"[,;\s]+", "a, b; c   d")                  # ['a', 'b', 'c', 'd']

# Splitting WITH captures keeps the separators
re.split(r"(\s+)", "a b  c")                          # ['a', ' ', 'b', '  ', 'c']
```

## A pragmatic IBAN validator

```python
import re


IBAN_PATTERN = re.compile(
    r"""^
    [A-Z]{2}        # country code
    \d{2}           # check digits
    [A-Z0-9]{1,30}  # BBAN
    $""",
    re.VERBOSE,
)


def is_valid_iban_format(s: str) -> bool:
    return bool(IBAN_PATTERN.match(s.replace(" ", "")))
```

For format-only validation. Real IBAN check requires the modulo-97 check digit algorithm — regex isn't enough.

## Common patterns

```python
# IPv4 (rough)
IPV4 = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

# URL (rough; for real URLs, use urllib.parse)
URL = r"https?://\S+"

# Email (RFC-aware emails are extremely complex; this is "good enough")
EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

# US phone (xxx) xxx-xxxx, xxx-xxx-xxxx, etc.
PHONE = r"\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})"

# UUID
UUID = r"[a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}"
```

For these specifically: prefer specialised libraries (`email-validator`, `uuid`, `phonenumbers`, `validators`). Regex is good enough for "does this look like a UUID" but not for "is this a valid Slovenian phone number."

## ReDoS — regular expression denial of service

```python
# DANGEROUS — exponential in input length
pattern = re.compile(r"^(\w+)*$")
pattern.match("a" * 30 + "!")        # hangs your process
```

Patterns with nested quantifiers + backtracking + non-matching input can take exponential time. If your regex runs on untrusted input, audit for ReDoS.

Mitigations:

- **Avoid nested quantifiers** (`(a+)+`).
- **Use atomic groups** (`(?>...)`) where available.
- **Set a timeout** on regex calls (Python doesn't natively, but `re2` does — `pip install google-re2`).

For untrusted input, `re2` is a drop-in safer alternative.

## When NOT to use regex

- **Parsing HTML / XML** — use `lxml` / `BeautifulSoup`.
- **Parsing programming languages** — use a real parser (`ast`, `tree-sitter`, `lark`).
- **Validating structured data** — use a schema (`pydantic`, `voluptuous`).
- **Date parsing** — use `dateutil` (Module 27 chapter 1).
- **Anything where the pattern is more than ~50 characters of regex** — refactor into multiple matches or a real parser.

## Pitfalls

!!! warning "Greedy by default"
    `.*` greedily matches as much as possible. For "as little as possible," `.*?`.

!!! warning "`$` at end-of-string vs end-of-line"
    Without `re.MULTILINE`, `$` matches only the end of the entire string. With, it matches each line. Be deliberate.

!!! warning "`re.findall` semantics with groups"
    `re.findall(r"(a)(b)", "ab")` returns `[('a', 'b')]`, not `['ab']`. With one group, returns just the captures. With multiple, tuples. Surprising.

!!! warning "Patterns in user input"
    Never use unescaped user input in a regex. Use `re.escape(user_input)` to escape metacharacters.

## Bottom line

For regex:

- **`re.VERBOSE` + comments** for any pattern > 30 chars.
- **Named groups** for readability.
- **Non-greedy `*?` / `+?`** when you mean it.
- **Specialised libraries** (`email-validator`, `phonenumbers`, `urllib.parse`) over regex for structured formats.
- **`re2`** when running on untrusted input.

Continue to **[Cryptography](04-cryptography.md)**.
