# Report generation

The end-of-day PDF that summarises P&L. The investor update with charts and tables. The compliance report with audit details. These all share a shape: query → format → render → email / archive. This chapter is the end-to-end pattern.

## The architecture

```
data source ──► query / aggregate
                    │
                    ▼
            structured data (dict / DataFrame)
                    │
                    ▼
            Jinja template (HTML)
                    │
                    ▼
            WeasyPrint → PDF
                    │
                    ▼
            email / S3 archive / Slack
```

Each step is replaceable. The contract: a data-shape object goes in; a PDF comes out.

## A complete daily P&L report

```python
import json
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML


TEMPLATE_DIR = Path(__file__).parent / "templates"


def query_daily_pnl(asof: date) -> pd.DataFrame:
    """Pull P&L from your warehouse / DB / strategy log."""
    # ... actual SQL or API call ...
    return pd.DataFrame({
        "strategy": ["momentum", "pairs", "carry", "calendar"],
        "pnl": [1250.0, -300.0, 850.0, 200.0],
        "notional": [50000, 25000, 30000, 15000],
        "sharpe_ytd": [0.85, 0.45, 0.72, 0.55],
    })


def make_equity_chart(pnl_history: pd.Series, out_path: Path):
    """Render an equity-curve PNG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    equity = (1 + pnl_history).cumprod()
    ax.plot(equity.index, equity.values, color="#305496", linewidth=2)
    ax.set_title("Equity Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_daily_report(asof: date, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    pnl = query_daily_pnl(asof)
    total_pnl = float(pnl["pnl"].sum())
    pnl_history = pd.Series([0.001, 0.002, -0.003, 0.001, 0.002])     # toy

    # 2. Assets
    chart_path = output_dir / f"equity_{asof}.png"
    make_equity_chart(pnl_history, chart_path)

    # 3. Render
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(),
    )
    template = env.get_template("daily.html.j2")
    html = template.render(
        asof=asof.isoformat(),
        rows=pnl.to_dict(orient="records"),
        total_pnl=total_pnl,
        chart_path=chart_path.absolute().as_uri(),
    )

    # 4. PDF
    pdf_path = output_dir / f"daily_{asof}.pdf"
    HTML(string=html, base_url=str(output_dir)).write_pdf(pdf_path)
    return pdf_path
```

Template `templates/daily.html.j2`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Daily P&L — {{ asof }}</title>
  <style>
    body { font-family: -apple-system, sans-serif; padding: 1.5em; color: #2c3e50; }
    h1 { border-bottom: 2px solid #305496; padding-bottom: 0.3em; }
    .total { font-size: 1.3em; font-weight: bold; margin: 1em 0; }
    .total.positive { color: #27ae60; }
    .total.negative { color: #c0392b; }
    table { border-collapse: collapse; width: 100%; margin-top: 1em; }
    th { background: #305496; color: white; text-align: left; padding: 8px; }
    td { padding: 8px; border-bottom: 1px solid #ddd; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .negative { color: #c0392b; }
    img { max-width: 100%; margin: 1em 0; }
    @page { size: A4; margin: 1cm; }
  </style>
</head>
<body>
  <h1>Daily P&L Report — {{ asof }}</h1>

  <p class="total {% if total_pnl >= 0 %}positive{% else %}negative{% endif %}">
    Total: ${{ '{:+,.2f}'.format(total_pnl) }}
  </p>

  <img src="{{ chart_path }}" alt="Equity curve">

  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th class="num">P&L</th>
        <th class="num">Notional</th>
        <th class="num">Sharpe YTD</th>
      </tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td>{{ row.strategy }}</td>
        <td class="num {% if row.pnl < 0 %}negative{% endif %}">${{ '{:+,.2f}'.format(row.pnl) }}</td>
        <td class="num">${{ '{:,.0f}'.format(row.notional) }}</td>
        <td class="num">{{ '{:.2f}'.format(row.sharpe_ytd) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <p style="margin-top:2em; font-size:0.8em; color:#888;">
    Confidential — generated {{ asof }}.
  </p>
</body>
</html>
```

Each piece is small; together they produce a polished PDF every morning.

## Multi-page reports

For long reports, CSS page breaks:

```css
.page-break { page-break-after: always; }
h1 { page-break-before: always; }
```

```html
<h1>Section 1: Summary</h1>
... content ...

<h1>Section 2: Detail</h1>
... content ...

<h1>Section 3: Appendix</h1>
... content ...
```

WeasyPrint respects `page-break-*` CSS. Each `h1` starts a new page.

## Embedding charts

Two approaches:

**File-based** (the cleanest for reports):

```python
make_equity_chart(data, output_dir / "chart.png")
# Reference in template: <img src="chart.png">
```

**Inline base64** (single self-contained file):

```python
import base64
from io import BytesIO


def chart_to_data_uri(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
```

Then `<img src="{{ chart_uri }}">` in the template.

## Emailing the report

```python
import smtplib
from email.message import EmailMessage


def email_report(pdf_path: Path, recipients: list[str], subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "reports@example.com"
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf",
                           filename=pdf_path.name)

    with smtplib.SMTP("smtp.example.com", 587) as s:
        s.starttls()
        s.login("reports@example.com", os.environ["SMTP_PASSWORD"])
        s.send_message(msg)
```

For Gmail / Microsoft 365, use OAuth or app passwords (not the account password).

For Slack:

```python
import requests


def post_to_slack(webhook_url: str, message: str, pdf_path: Path):
    # Upload to S3 first, then post the link
    s3_url = upload_to_s3(pdf_path)
    requests.post(webhook_url, json={
        "text": f"{message} — <{s3_url}|Download PDF>",
    })
```

Slack's file upload API is rate-limited; for daily reports, hosting on S3 and linking is more robust.

## Archiving

Save reports to a date-partitioned directory:

```
reports/
  2024-11-15/
    daily_2024-11-15.pdf
    chart.png
    raw_data.csv
  2024-11-16/
    daily_2024-11-16.pdf
    ...
```

For long retention, copy to S3 / Glacier with lifecycle policies. SHA-256 checksum on creation; verify on retrieval.

## Scheduling

Module 22 chapter 6 covered this. For daily reports:

```cron
30 6 * * 1-5 /usr/bin/python /opt/reports/daily.py >> /var/log/daily-report.log 2>&1
```

Or `systemd timer`. Or GitHub Actions schedule. Whatever fits.

## Pitfalls

!!! warning "Template injection"
    User input rendered in templates without auto-escape can inject HTML / CSS. Always `select_autoescape()`.

!!! warning "Fonts that don't render in WeasyPrint"
    System fonts work on your laptop, fail in a Docker container. Bundle fonts (e.g., Inter) in `static/fonts/` and reference via CSS `@font-face`.

!!! warning "Per-page header / footer drift"
    `@page` rules in WeasyPrint are stable; in ReportLab they need explicit `onLaterPages` handlers. Test multi-page output.

!!! warning "Floating point in numeric formatting"
    `${{ '{:.2f}'.format(0.1 + 0.2) }}` shows "0.30" but `0.1 + 0.2 != 0.3`. Round explicitly before display: `'{:.2f}'.format(round(value, 2))`.

## Bottom line

For report generation:

- **Jinja2 + WeasyPrint** for HTML-driven reports.
- **Templates in version control**; styles in CSS, not Python.
- **Charts as files** (or base64 data URIs for portability).
- **Date-partitioned archive** + S3 / Glacier for long retention.
- **Scheduled via cron / systemd / GitHub Actions** — pick one per the operational chapter.

## End of Module 24

You now have the document + media toolkit. The next module covers network programming — sockets, gRPC, WebSocket clients beyond the trading-feed flavour, MQTT, ZeroMQ.

Continue to **[Module 25 — Network programming](../25-network-programming/index.md)**.
