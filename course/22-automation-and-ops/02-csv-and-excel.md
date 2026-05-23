# CSV, Excel, and tabular interchange

CSV and Excel are how non-engineers send you data. Your inbox is full of them; your operations are full of generating them. This chapter is the production handling of both — fast, correct, and robust to the formatting weirdness real-world files carry.

## CSV: the standard library's `csv` module

For anything beyond toy size, `pandas.read_csv` is faster. But `csv` from the stdlib has its place: streaming, low-memory parsing, no extra dependency.

```python
import csv
from pathlib import Path

with open("input.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)              # uses first row as headers
    for row in reader:
        process(row["symbol"], float(row["close"]))


# Writing
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["symbol", "close", "volume"])
    writer.writeheader()
    for record in records:
        writer.writerow(record)
```

Two non-obvious things:

1. **`newline=""`** — must always be passed when opening for CSV. Without it, you get mangled line endings on Windows.
2. **Encoding** — always specify it. The default is platform-dependent. UTF-8 is almost always right; if the vendor sends Latin-1 or Windows-1252, specify exactly that.

## Dialect detection

Vendors send CSVs with all kinds of quirks: semicolon separators, tab separators, single-quote quoting, escaped quotes, etc. `csv.Sniffer` reads a sample and detects:

```python
with open("mystery.csv", newline="", encoding="utf-8") as f:
    sample = f.read(2048)
    dialect = csv.Sniffer().sniff(sample)
    has_header = csv.Sniffer().has_header(sample)
    f.seek(0)
    reader = csv.reader(f, dialect=dialect)
    if has_header:
        next(reader)
    for row in reader:
        process(row)
```

When the file's format is unknown, sniff first.

## pandas for CSV: the production defaults

```python
import pandas as pd

df = pd.read_csv(
    "data.csv",
    dtype={"symbol": "string", "volume": "Int64"},   # nullable int
    parse_dates=["timestamp"],
    date_format="%Y-%m-%d %H:%M:%S%z",
    na_values=["", "N/A", "null", "-"],
)
```

Key flags every production caller should set:

- **`dtype`** — explicit per-column types. Prevents pandas from guessing `int64` and silently failing on a missing value.
- **`parse_dates` + `date_format`** — coerce timestamps at read time. Don't post-process.
- **`na_values`** — every vendor uses different "missing" markers. Specify all of them.
- **`usecols`** — only read the columns you actually need; saves time and memory.
- **`chunksize=N`** — for files too big for memory, iterate chunks of N rows.

For really big files, switch to `polars.scan_csv` (Module 4 chapter 5).

## Writing CSV correctly

```python
df.to_csv(
    "output.csv",
    index=False,                  # almost always: skip the row index
    date_format="%Y-%m-%dT%H:%M:%S%z",
    encoding="utf-8",
)
```

For interchange with non-Python consumers (Excel, R, BI tools), pass `date_format` explicitly so timestamps don't roundtrip through localised formats.

## Excel — `openpyxl` + `pandas`

For simple reads and writes, pandas is enough:

```python
df = pd.read_excel("report.xlsx", sheet_name="Strategies", engine="openpyxl")
df.to_excel("output.xlsx", sheet_name="Results", index=False, engine="openpyxl")
```

For multi-sheet workbooks:

```python
all_sheets = pd.read_excel("report.xlsx", sheet_name=None)        # dict of {name: df}
with pd.ExcelWriter("output.xlsx", engine="openpyxl") as xw:
    df_summary.to_excel(xw, sheet_name="Summary", index=False)
    df_details.to_excel(xw, sheet_name="Details", index=False)
```

## Excel with formatting — direct openpyxl

When you need cell formatting, conditional formatting, formulas, charts — drop down to `openpyxl`:

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import CellIsRule


wb = Workbook()
ws = wb.active
ws.title = "P&L"

# Headers with bold + colored fill
headers = ["Date", "Strategy", "P&L", "% of Capital"]
for col_idx, header in enumerate(headers, start=1):
    cell = ws.cell(row=1, column=col_idx, value=header)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="305496")
    cell.alignment = Alignment(horizontal="center")

# Rows
for row_idx, record in enumerate(records, start=2):
    ws.cell(row=row_idx, column=1, value=record.date)
    ws.cell(row=row_idx, column=2, value=record.strategy)
    ws.cell(row=row_idx, column=3, value=record.pnl).number_format = "$#,##0.00"
    ws.cell(row=row_idx, column=4, value=record.pct).number_format = "0.00%"

# Conditional formatting: highlight negative P&L in red
red_fill = PatternFill("solid", fgColor="FFC7CE")
ws.conditional_formatting.add(
    f"C2:C{len(records) + 1}",
    CellIsRule(operator="lessThan", formula=["0"], fill=red_fill),
)

# Auto-size columns
for col_idx in range(1, len(headers) + 1):
    column_letter = get_column_letter(col_idx)
    max_len = max(len(str(c.value or "")) for c in ws[column_letter])
    ws.column_dimensions[column_letter].width = max_len + 2

# Add a Total row with a formula
total_row = len(records) + 2
ws.cell(row=total_row, column=2, value="Total").font = Font(bold=True)
ws.cell(row=total_row, column=3, value=f"=SUM(C2:C{total_row - 1})").number_format = "$#,##0.00"

wb.save("p_and_l_report.xlsx")
```

This is what a "daily P&L report" looks like when generated for non-technical stakeholders. Bold headers, currency formatting, conditional colour, totals row.

## Streaming Excel — for files that don't fit in memory

`openpyxl` has a read-only mode that streams:

```python
from openpyxl import load_workbook

wb = load_workbook("huge.xlsx", read_only=True, data_only=True)
ws = wb["Sheet1"]
for row in ws.iter_rows(values_only=True):
    process(row)
wb.close()
```

For writing huge workbooks, `openpyxl` has a write-only mode:

```python
from openpyxl import Workbook

wb = Workbook(write_only=True)
ws = wb.create_sheet("Data")
ws.append(["col1", "col2", "col3"])         # header
for record in iter_huge_dataset():
    ws.append([record.a, record.b, record.c])
wb.save("huge.xlsx")
```

Both modes trade convenience for memory; you can produce / consume million-row workbooks without OOMing.

## CSV ↔ Database

A common operation: dump a database table to CSV, or load CSV into a table. Using SQLAlchemy:

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")

# Database → CSV
df = pd.read_sql("SELECT * FROM trades WHERE date = '2024-11-15'", engine)
df.to_csv("trades_2024-11-15.csv", index=False)

# CSV → Database (use COPY for speed on Postgres)
df = pd.read_csv("incoming_fills.csv")
df.to_sql("fills", engine, if_exists="append", index=False, method="multi", chunksize=10_000)
```

For very large loads on Postgres, the database's native `COPY` command is dramatically faster:

```python
import psycopg

with psycopg.connect("postgresql://user:pass@host/db") as conn:
    with conn.cursor() as cur:
        with open("massive.csv") as f:
            with cur.copy("COPY fills FROM STDIN WITH (FORMAT CSV, HEADER true)") as copy:
                copy.write(f.read())
        conn.commit()
```

For 10M+ rows, that's an order of magnitude faster than INSERTs.

## Validation at the boundary

Always validate CSV/Excel input. Schema mismatches are the #1 source of pipeline bugs:

```python
from pydantic import BaseModel, validator


class TradeRow(BaseModel):
    date: str            # ISO format
    symbol: str
    quantity: int
    price: float

    @validator("price")
    def price_positive(cls, v):
        if v <= 0:
            raise ValueError("price must be positive")
        return v


def load_trades(path):
    df = pd.read_csv(path)
    trades = []
    errors = []
    for i, row in df.iterrows():
        try:
            trades.append(TradeRow(**row.to_dict()))
        except ValueError as e:
            errors.append((i, str(e)))
    if errors:
        raise ValueError(f"{len(errors)} validation errors: {errors[:3]}")
    return trades
```

This pattern catches bad input at the file's edge, with row-level error messages. Far better than a silent type coercion that surfaces an obscure error 20 layers deep in your pipeline.

## Pitfalls

!!! warning "Excel auto-corrupts gene names"
    Famous example: `MARCH1` (a gene) is silently parsed as a date by Excel. For any column that might be misinterpreted, pre-quote it (`'MARCH1'`) when generating, or specify `dtype=str` when reading.

!!! warning "Encoding mismatches in CSV"
    A CSV produced on Windows is often UTF-8-with-BOM or Windows-1252. UTF-8 reads will silently emit garbled characters. Detect the encoding (`chardet` or `charset-normalizer`) at first ingest.

!!! warning "Date inference is fragile"
    `pd.read_csv` will parse `"01/02/03"` as either Jan 2 2003 or Feb 1 2003 depending on `dayfirst=`. Specify `date_format` explicitly.

!!! warning "Excel column letter overflow"
    The 26-letter scheme wraps to AA, AB, ... at column 27. Use `openpyxl.utils.get_column_letter(n)` rather than rolling your own.

## Bottom line

For production CSV/Excel:

- **CSV**: `pd.read_csv` with explicit dtypes + date_format + na_values; stream with `chunksize=` for big files.
- **Excel**: `pd.read_excel` / `pd.to_excel` for plain I/O; drop to `openpyxl` for formatting, formulas, conditional fills.
- **Validation at the boundary** with pydantic models.
- **Encoding always specified** explicitly.

Continue to **[Database scripting](03-databases.md)**.
