# Dates, times, time zones

Half of operational bugs involve dates. Timezone offsets, daylight savings, leap seconds, ISO formats, business days — this chapter is the working toolkit.

## `datetime` — stdlib

```python
from datetime import datetime, date, time, timedelta, timezone


# Now
now = datetime.now(tz=timezone.utc)         # ALWAYS tz-aware in production
print(now.isoformat())                       # 2024-11-15T14:30:00+00:00

# Components
print(now.year, now.month, now.day, now.hour, now.minute, now.second)

# Arithmetic
tomorrow = now + timedelta(days=1)
two_hours = timedelta(hours=2)
diff = tomorrow - now                        # timedelta

# Parse ISO
parsed = datetime.fromisoformat("2024-11-15T14:30:00+00:00")

# Format
print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
```

The two rules to internalise:

1. **Always use tz-aware datetimes in production.** Naive datetimes (`datetime.now()` without `tz=`) are sources of bugs.
2. **Use UTC for storage and computation.** Convert to local time only for display.

## `zoneinfo` — proper timezone handling

Python 3.9+ has `zoneinfo` in stdlib (replacing `pytz`):

```python
from datetime import datetime
from zoneinfo import ZoneInfo


ny = ZoneInfo("America/New_York")
tokyo = ZoneInfo("Asia/Tokyo")

# Local time
ny_time = datetime(2024, 11, 15, 9, 30, tzinfo=ny)
print(ny_time.isoformat())                   # 2024-11-15T09:30:00-05:00

# Convert
print(ny_time.astimezone(tokyo))             # 2024-11-15T23:30:00+09:00

# Now in a zone
print(datetime.now(ny))
```

`ZoneInfo` is the right way to attach timezones. Avoid `pytz` in new code (different API; bug-prone localize/normalize calls).

Get the list of all zones: `from zoneinfo import available_timezones; print(sorted(available_timezones())[:10])`.

## Daylight Savings Time

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


ny = ZoneInfo("America/New_York")

# Spring forward: 2024-03-10 02:00 doesn't exist
print(datetime(2024, 3, 10, 1, 30, tzinfo=ny).isoformat())   # 01:30 EST
print(datetime(2024, 3, 10, 3, 30, tzinfo=ny).isoformat())   # 03:30 EDT

# "Add 1 hour to 1:30 AM on March 10"
t = datetime(2024, 3, 10, 1, 30, tzinfo=ny)
later = t + timedelta(hours=1)
print(later.isoformat())                                     # 03:30 EDT — skipped 2-3
```

`zoneinfo` handles DST correctly. The naive approach (`datetime.replace(tzinfo=ny)`) doesn't.

For storage / arithmetic, work in **UTC**. Convert to local only for display. This eliminates ~95% of timezone bugs.

## `dateutil` — for parsing and timedeltas

```python
from dateutil import parser
from dateutil.relativedelta import relativedelta


# Parse fuzzy human formats
print(parser.parse("Nov 15, 2024 14:30 EST"))        # works
print(parser.parse("2024-11-15"))                     # works
print(parser.parse("yesterday"))                      # error — dateutil isn't natural-language
print(parser.parse("15/11/2024", dayfirst=True))      # disambiguate


# Calendar arithmetic (months, years)
from datetime import date
print(date(2024, 1, 31) + relativedelta(months=1))    # 2024-02-29 (not Feb 31!)
print(date(2024, 11, 15) + relativedelta(years=1, months=2))
```

`timedelta` knows about days / seconds but not months or years (variable-length). `relativedelta` handles "1 month from now" correctly.

## ISO 8601 — the standard

ISO 8601 is the canonical string format:

```
2024-11-15T14:30:00.123456+00:00
[year]-[mo]-[dy]T[hr]:[mn]:[sc].[us]±[tz_offset]
```

For storage / interchange / logs / APIs: always ISO 8601. Languages and timezones differ on formats; ISO is unambiguous.

```python
from datetime import datetime

dt = datetime.now(tz=timezone.utc)
serialized = dt.isoformat()                          # "2024-11-15T14:30:00.123456+00:00"
parsed = datetime.fromisoformat(serialized)         # roundtrip
```

## Unix timestamps

```python
from datetime import datetime, timezone


t = datetime.now(tz=timezone.utc)
ts = t.timestamp()                                   # 1731685800.123456 (float seconds since epoch)
back = datetime.fromtimestamp(ts, tz=timezone.utc)
```

For external systems (JavaScript: `Date.now() / 1000`, Unix tools, databases): timestamps are common. Always tz-aware on the way back.

Integer milliseconds (the JS / Kafka / many modern APIs convention):

```python
ms = int(t.timestamp() * 1000)
```

## `pendulum` — the friendlier alternative

[Pendulum](https://pendulum.eustace.io/) wraps datetime with a more ergonomic API:

```python
import pendulum


now = pendulum.now("America/New_York")
print(now.to_iso8601_string())
print(now.add(days=3).end_of("month"))
print(pendulum.parse("2024-11-15"))                  # auto-handles many formats
print(now.diff_for_humans())                          # "a few seconds ago"
```

For applications heavy on date arithmetic, pendulum saves a lot of typing. For one-off scripts, stdlib is fine.

## Date arithmetic and pandas

For panel work:

```python
import pandas as pd


idx = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
print(len(idx))                                       # business days in 2024

# Shift
later = idx + pd.Timedelta(days=5)

# Truncate to month
months = idx.to_period("M")

# Business-day arithmetic
from pandas.tseries.offsets import BDay
print(pd.Timestamp("2024-11-15") + BDay(3))
```

Pandas's date / time handling is mature; for any panel-level work, prefer pandas over hand-rolling.

## Common bugs

### The "naive vs aware" mix

```python
# WRONG — mixing
naive = datetime.now()                                # local time, no tz
aware = datetime.now(tz=timezone.utc)
naive < aware                                         # TypeError
```

In production, never use `datetime.now()` without `tz=`. Make it a lint rule.

### Daylight savings double-arithmetic

```python
ny = ZoneInfo("America/New_York")
midnight = datetime(2024, 11, 3, tzinfo=ny)           # 2024-11-03 00:00 EDT
# Add 24 hours
next_day = midnight + timedelta(hours=24)
# 2024-11-04 00:00 EST (DST ended, gained an hour)

# But the actual "next midnight in NY":
actual_next = datetime(2024, 11, 4, tzinfo=ny)
print(midnight + timedelta(hours=24) == actual_next)  # False! (off by an hour)
```

For "midnight tomorrow in this timezone," use `datetime(...)` construction or `relativedelta(days=1)`, not `timedelta(hours=24)`.

### Locale-sensitive formatting

```python
import locale

locale.setlocale(locale.LC_TIME, "fr_FR")
print(datetime.now().strftime("%B"))                  # "novembre" not "November"
```

For machine-readable formats, always use `strftime` with explicit format strings or ISO methods. Locale-sensitive output silently changes between environments.

## A worked example: timezone-aware logging

```python
import logging
from datetime import datetime
from zoneinfo import ZoneInfo


class TimezoneAwareFormatter(logging.Formatter):
    def __init__(self, fmt=None, tz: ZoneInfo = ZoneInfo("UTC")):
        super().__init__(fmt or "%(asctime)s %(levelname)s %(message)s")
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        return dt.isoformat()


handler = logging.StreamHandler()
handler.setFormatter(TimezoneAwareFormatter(tz=ZoneInfo("America/New_York")))
log = logging.getLogger()
log.addHandler(handler)
log.setLevel(logging.INFO)

log.info("event")
# 2024-11-15T09:30:00.123456-05:00 INFO event
```

Logs in the same TZ as the team makes life easier; ISO format makes them sortable / parseable.

## Bottom line

For dates:

- **Tz-aware datetimes always**; UTC for storage / computation.
- **`zoneinfo`** for timezones (stdlib in 3.9+).
- **`dateutil.relativedelta`** for month/year arithmetic.
- **`pendulum`** for ergonomic date code.
- **ISO 8601** for interchange.
- **Never `datetime.now()` without `tz=`** in production.

Continue to **[Business calendars and recurrence](02-calendars.md)**.
