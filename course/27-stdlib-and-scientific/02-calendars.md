# Business calendars and recurrence

"Skip weekends" is easy. "Skip US holidays" requires a calendar. "Every third Thursday from now until December" requires a recurrence rule. This chapter covers the working tools.

## pandas business-day machinery

```python
import pandas as pd
from pandas.tseries.offsets import BDay, CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


# Naive business days (skip weekends only)
print(pd.Timestamp("2024-11-15") + BDay(3))           # 2024-11-20 (skipped weekend)


# US Federal holiday calendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start="2024-01-01", end="2024-12-31")
print(holidays)                                        # ['2024-01-01', '2024-01-15', ...]


# Custom business day skipping holidays
cd = CDay(holidays=holidays)
print(pd.Timestamp("2024-11-26") + cd)                 # skips Thanksgiving
```

`USFederalHolidayCalendar` covers federal holidays. For NYSE specifically (early closes, market-specific holidays), use a specialised calendar.

## `pandas_market_calendars` — exchange calendars

```python
import pandas_market_calendars as mcal


nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date="2024-11-01", end_date="2024-12-31")
print(schedule.head())
# market_open                market_close
# 2024-11-01 13:30:00+00:00  2024-11-01 21:00:00+00:00
# ...
# 2024-11-29 14:30:00+00:00  2024-11-29 18:00:00+00:00      # early close after Thanksgiving
```

Includes:

- **Open and close times**, including early closes.
- **Half-day rules** (day after Thanksgiving, Christmas Eve).
- **Pre-market / after-hours** sessions if you ask.

For any time-of-day-sensitive trading code on US markets, this is the right calendar.

## `workalendar` — global business calendars

```python
from workalendar.usa import UnitedStates
from workalendar.europe import UnitedKingdom, Germany


us = UnitedStates()
uk = UnitedKingdom()
de = Germany()

print(us.is_working_day(date(2024, 11, 28)))           # False (Thanksgiving)
print(uk.is_working_day(date(2024, 12, 26)))           # False (Boxing Day)
print(de.add_working_days(date(2024, 12, 24), 5))      # 5 working days from Christmas Eve in Germany
```

Covers dozens of countries; useful for multi-region scheduling.

## Recurrence rules — `dateutil.rrule`

```python
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY, WEEKLY, MO, TU, WE, TH, FR


# Every 3rd Thursday of every month for the next year
expirations = rrule(
    MONTHLY,
    byweekday=TH(+3),
    count=12,
    dtstart=datetime(2024, 1, 1),
)
for d in expirations:
    print(d)


# Weekly on Tuesdays and Thursdays, until end of year
weekly = rrule(WEEKLY, byweekday=(TU, TH), until=datetime(2024, 12, 31),
                dtstart=datetime(2024, 11, 15))
for d in weekly:
    print(d)
```

`rrule` is the iCalendar recurrence-rule implementation. Powerful for "fire this job every X" calendars.

For options expirations (third Friday of every month):

```python
from dateutil.rrule import FR

monthly_exp = rrule(MONTHLY, byweekday=FR(+3), count=24, dtstart=datetime(2024, 1, 1))
```

## A worked example: scheduling around exchange holidays

```python
import pandas as pd
import pandas_market_calendars as mcal


def is_trading_day(d: pd.Timestamp, exchange: str = "NYSE") -> bool:
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=d, end_date=d)
    return not schedule.empty


def next_trading_day(d: pd.Timestamp, exchange: str = "NYSE") -> pd.Timestamp:
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=d, end_date=d + pd.Timedelta(days=14))
    after = schedule[schedule.index > d]
    return after.index[0] if len(after) else None


def n_trading_days_ahead(d: pd.Timestamp, n: int, exchange: str = "NYSE") -> pd.Timestamp:
    cal = mcal.get_calendar(exchange)
    days = cal.valid_days(start_date=d, end_date=d + pd.Timedelta(days=n * 2 + 30))
    days_after = days[days > d]
    return days_after[n - 1] if len(days_after) >= n else None


print(n_trading_days_ahead(pd.Timestamp("2024-11-26"), 5, "NYSE"))
# skips Thanksgiving + half-day
```

For "fire this strategy 5 trading days from now" — the right calendar matters; calendar-day arithmetic gives the wrong answer.

## Days between exclusive of weekends + holidays

```python
def business_days_between(start: pd.Timestamp, end: pd.Timestamp, calendar="NYSE") -> int:
    cal = mcal.get_calendar(calendar)
    return len(cal.valid_days(start_date=start, end_date=end))
```

Useful for settlement-period calculations (T+2 settlement: "from trade date, add 2 trading days").

## Settlement: T+1 / T+2

US equities are T+1 settlement (since 2024); most other markets are T+2.

```python
def settlement_date(trade_date: pd.Timestamp, calendar: str = "NYSE", lag: int = 1) -> pd.Timestamp:
    cal = mcal.get_calendar(calendar)
    days = cal.valid_days(start_date=trade_date, end_date=trade_date + pd.Timedelta(days=lag * 3 + 7))
    after = days[days > trade_date]
    return after[lag - 1] if len(after) >= lag else None


print(settlement_date(pd.Timestamp("2024-11-27"), "NYSE", lag=1))
# 2024-11-29 (Thanksgiving is the 28th)
```

For futures and options: different settlement schedules per product.

## Cron expressions

For scheduling expressions in human-readable form:

```python
from croniter import croniter
from datetime import datetime


cron = croniter("0 9 * * 1-5", datetime(2024, 11, 15))
for _ in range(5):
    print(cron.get_next(datetime))
# Next 5 weekday 09:00 occurrences
```

Useful when you have user-configurable schedules ("send me a report every weekday at 9 AM" → "0 9 * * 1-5" in the DB).

## Pitfalls

!!! warning "Implicit timezone assumption"
    `pd.Timestamp("2024-11-15")` is naive. Pair with `tz=` or convert before arithmetic.

!!! warning "Holiday calendar staleness"
    A `pandas_market_calendars` version pinned in 2022 may not know about 2025 holidays. Update the package; check for new federal holidays (e.g., Juneteenth added in 2021).

!!! warning "Half-days"
    "Day after Thanksgiving" closes at 13:00 EST. Strategies that assume 16:00 close fire incorrectly. Always read `market_close` from the calendar, don't hardcode.

!!! warning "Multi-region calendars"
    "Business days" means different things in different countries. Don't assume the U.S. calendar globally.

## Bottom line

For business calendars:

- **`pandas_market_calendars`** for exchange schedules.
- **`workalendar`** for international business days.
- **`dateutil.rrule`** for recurrence (third Thursday, every quarter, etc.).
- **`croniter`** for cron-style schedules.
- **Always check `market_close`** rather than assuming a fixed time.

Continue to **[Regex mastery](03-regex.md)**.
