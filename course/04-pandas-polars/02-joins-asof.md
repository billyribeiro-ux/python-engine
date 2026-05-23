# Joins, `merge_asof`, and tick alignment

Joining market data has a thousand ways to go wrong. Look-ahead from a join that quietly picked future rows. Wrong fill behaviour silently changing a backtest. Mixed timezones generating off-by-one bugs every March and November.

This chapter walks the joins you actually need — `merge`, `join`, `merge_ordered`, and the one that wins for tick alignment, `merge_asof`.

## `merge` — relational join, ignoring the index

`pd.merge` is the SQL-style join. By default it joins on the columns that share a name.

```python
import pandas as pd

prices = pd.DataFrame({"symbol": ["SPY", "QQQ"], "px": [475.0, 410.0]})
sectors = pd.DataFrame({"symbol": ["SPY", "QQQ"], "sector": ["Broad", "Tech"]})

pd.merge(prices, sectors)          # inner join on `symbol`
pd.merge(prices, sectors, how="left")  # keep all rows of `prices`
```

Three things to call out:

1. `how=` is `inner` by default. Quietly drops mismatched rows.
2. The index is **not** used unless `left_index=True` / `right_index=True`.
3. Duplicate keys produce a **Cartesian product** for the duplicates — `validate="1:1"` (or `1:m`, `m:1`, `m:m`) catches the surprise version.

```python
pd.merge(a, b, on="symbol", validate="1:1")  # raises if either side has dupes
```

`validate=` is one of the highest-value habits to develop. Use it everywhere you join.

## `join` — index-on-index, convenience wrapper

`df.join(other)` is shorthand for `df.merge(other, left_index=True, right_index=True, how="left")`. Useful when both sides are already indexed and you want a clean "tack the columns on" call.

```python
df.join(macro, how="left")
df.join([macro, fundamentals], how="left")   # join several at once
```

## `concat` — stacking, not joining

`pd.concat([df1, df2])` stacks frames along an axis. Use it when:

- You're building a frame from a list of chunks: `pd.concat(list_of_dfs)`.
- You want to join along axis=1 with index alignment: `pd.concat([a, b], axis=1)`.

The list-comprehension + `concat` pattern is the right replacement for `df.append` in a loop (which is deprecated in pandas 2.x):

```python
# WRONG (slow + deprecated)
out = pd.DataFrame()
for chunk in chunks:
    out = out.append(chunk)

# RIGHT
out = pd.concat([process(chunk) for chunk in chunks], ignore_index=True)
```

`append` was O(n²) — every call copied. The `concat` form is O(n).

## `merge_asof` — the killer for time-series data

This is the join that quant Python lives on. It joins on a sorted key (usually a timestamp) and matches each left row to the nearest right row in the chosen direction.

```python
import pandas as pd

bars = pd.DataFrame({
    "px": [100, 101, 102, 103],
}, index=pd.to_datetime(["09:30", "09:31", "09:32", "09:33"]))

quotes = pd.DataFrame({
    "bid": [99.5, 100.5, 101.5, 102.5, 103.5],
    "ask": [100.5, 101.5, 102.5, 103.5, 104.5],
}, index=pd.to_datetime(["09:30:15", "09:30:45", "09:31:30", "09:32:30", "09:33:30"]))

aligned = pd.merge_asof(
    bars.sort_index(),
    quotes.sort_index(),
    left_index=True, right_index=True,
    direction="backward",     # take the latest quote AT OR BEFORE the bar
)
print(aligned)
#                       px    bid    ask
# 09:30                100   NaN    NaN     # no quote at or before 09:30
# 09:31                101  100.5  101.5    # last quote was 09:30:45
# 09:32                102  101.5  102.5
# 09:33                103  102.5  103.5
```

**Direction matters.**

- `backward` — last value at or before the left timestamp. **Use this for backtests.** No look-ahead.
- `forward` — first value at or after.
- `nearest` — closer of the two. Convenient for analytics; **never** for backtests.

### `tolerance` — bound how stale the match can be

```python
aligned = pd.merge_asof(
    bars, quotes,
    left_index=True, right_index=True,
    direction="backward",
    tolerance=pd.Timedelta("5s"),   # if no quote within 5s, NaN
)
```

This is how you encode "the quote must be reasonably fresh." For tick → bar joins, a 1–10 second tolerance is typical; for daily → minute joins, no tolerance at all.

### `by=` — match within a group

If you have many symbols in one frame, you don't want quotes for AAPL bleeding into SPY's matches. `by=` is the per-group key:

```python
aligned = pd.merge_asof(
    bars.sort_values("ts"),
    quotes.sort_values("ts"),
    on="ts", by="symbol",
    direction="backward",
    tolerance=pd.Timedelta("1s"),
)
```

`merge_asof` requires both sides to be sorted on the asof key. Forgetting to sort first is the #1 reason you'll see a confusing `ValueError: left keys must be sorted`.

## A worked example: aligning a daily signal to intraday bars

You computed a daily momentum signal at the close of each day; now you want each intraday bar of the *next* day labelled with that signal so you can backtest a follow-through strategy.

```python
# daily_signal indexed by date (UTC midnight)
# intraday_bars indexed by minute (UTC)

# the signal becomes effective at the next open — shift forward one day
sig = daily_signal.shift(freq="1D")

aligned = pd.merge_asof(
    intraday_bars.sort_index(),
    sig.sort_index(),
    left_index=True, right_index=True,
    direction="backward",
)
```

Three lines. No look-ahead. The `shift(freq="1D")` is the move that puts the signal *one day in the future* relative to when it was computed — i.e., available at the next session's start, not the same session's. Module 9 will hammer this point: half of backtest leakage comes from index timestamps that don't accurately represent when information was actually known.

## Re-indexing and `reindex`

A different beast from `merge`. `reindex` says: "rearrange this Series/DataFrame to *exactly this* set of labels, filling missing ones."

```python
target = pd.date_range("2024-01-02", "2024-01-31", freq="B", tz="UTC")
aligned_to_business_days = df.reindex(target)
```

Missing labels get NaN by default. `method="ffill"` carries the last value forward; `method="bfill"` carries backward.

`reindex` is the right tool when you want to enforce a calendar (e.g. business days, custom trading sessions). For "find the nearest match in the existing index", that's `merge_asof`.

## A trap: timezones in joins

```python
# left index is tz-aware UTC; right is naive
pd.merge_asof(left, right, left_index=True, right_index=True)
# TypeError: incompatible merge keys ...
```

Always localise *both* sides to the same tz before joining. The course's data adapter (`engine.data.feed`) returns tz-aware UTC frames precisely because tz consistency is enforced at one chokepoint.

## A trap: integer keys vs `Int64` nullable keys

```python
pd.merge(a, b, on="id")
# TypeError: You are trying to merge on int64 and Int64 columns
```

Pandas distinguishes the lowercase numpy types from the uppercase nullable variants. Coerce one side to match the other.

## A trap: merging on string vs categorical

`pd.Categorical` is much faster for repeated string keys. Mixing the two in a merge:

```python
a["sector"] = a["sector"].astype("category")
pd.merge(a, b, on="sector")
# Works, but no longer categorical — pandas materialises one side
```

For repeated-key joins at scale, **both sides categorical with the same categories** is the right setup. Or move to polars / DuckDB.

## Bottom line

For market data:

- **Tick/bar alignment** → `merge_asof` with `direction="backward"`, a sensible `tolerance`, and `by="symbol"` when there are multiple symbols.
- **Tacking on reference data** → `df.join(reference)`.
- **Building a frame from chunks** → `pd.concat([list_of_chunks])`.
- **Enforcing a calendar** → `df.reindex(calendar)`.

Always sort first. Always pass `validate=`. Always make the timezones match.

Continue to **[Rolling, expanding, and `ewm`](03-rolling.md)**.
