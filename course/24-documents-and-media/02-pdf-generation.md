# Generating PDFs

Three good approaches, each with a different sweet spot:

1. **WeasyPrint** — HTML/CSS to PDF. Best when you can think in web layout.
2. **ReportLab** — programmatic PDF creation. Best for fine control of every element.
3. **PyMuPDF (`fitz`)** — modify existing PDFs or build simple new ones.

## WeasyPrint — HTML → PDF

The "I want a nicely formatted report" approach:

```python
from weasyprint import HTML, CSS


html_content = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Daily Report</title>
</head>
<body>
  <h1>Daily P&L Report — 2024-11-15</h1>
  <table>
    <thead><tr><th>Symbol</th><th>P&L</th></tr></thead>
    <tbody>
      <tr><td>SPY</td><td>+$1,250</td></tr>
      <tr><td>QQQ</td><td>-$300</td></tr>
    </tbody>
  </table>
</body>
</html>
"""

css = CSS(string="""
body { font-family: -apple-system, sans-serif; }
h1 { color: #2c3e50; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 8px; }
th { background: #305496; color: white; }
""")

HTML(string=html_content).write_pdf("report.pdf", stylesheets=[css])
```

If you can render it in a browser, WeasyPrint can render it to PDF. Page breaks via CSS (`page-break-before: always;`). Charts via embedded `<img>` (render to PNG with matplotlib, base64-embed, drop in).

For Jinja templates:

```python
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("report.html.j2")
html = template.render(date="2024-11-15", trades=trades, pnl_summary=pnl)
HTML(string=html, base_url=".").write_pdf("out.pdf")
```

`base_url="."` lets the template reference local images / CSS. Standard pattern for production reports.

## ReportLab — programmatic

For precise control:

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch


doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                         leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                         topMargin=0.5 * inch, bottomMargin=0.5 * inch)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("Daily P&L Report — 2024-11-15", styles["Title"]))
story.append(Spacer(1, 0.2 * inch))

# Table
data = [
    ["Symbol", "P&L", "Notional"],
    ["SPY", "+$1,250", "$50,000"],
    ["QQQ", "-$300", "$25,000"],
    ["IWM", "+$120", "$15,000"],
]
table = Table(data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#305496")),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
    ("BACKGROUND", (0, 1), (-1, -1), HexColor("#F5F5F5")),
]))
story.append(table)

doc.build(story)
```

ReportLab is verbose but precise. You position every element. For complex multi-page reports with charts, ReportLab is the industrial standard.

## Charts in PDFs

Render with matplotlib to PNG, embed:

```python
import matplotlib.pyplot as plt
from io import BytesIO


fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([1, 2, 3, 4], [10, 11, 13, 12])
ax.set_title("Equity Curve")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
plt.close(fig)


# In WeasyPrint: base64-embed in HTML
import base64
img_b64 = base64.b64encode(buf.getvalue()).decode()
html = f'<img src="data:image/png;base64,{img_b64}" width="600">'


# In ReportLab:
from reportlab.platypus import Image as RLImage
img = RLImage(buf, width=6 * inch, height=3 * inch)
story.append(img)
```

For vector charts, render to SVG instead. ReportLab has native SVG support via `svglib`.

## Page numbers and headers

Both libraries support page-level templating:

```python
# ReportLab
from reportlab.lib.pagesizes import letter


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(0.5 * inch, 0.3 * inch, f"Page {doc.page}")
    canvas.drawRightString(letter[0] - 0.5 * inch, 0.3 * inch, "Confidential")
    canvas.restoreState()


doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
```

For WeasyPrint, use CSS `@page` rules:

```css
@page {
  @top-left { content: "My Report"; }
  @bottom-right { content: counter(page); }
  margin: 0.5in;
}
```

## A worked example: a full daily report

```python
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


def generate_daily_report(pnl_data, output_path):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("daily_report.html.j2")
    html = template.render(
        report_date=date.today().isoformat(),
        pnl_data=pnl_data,
        total_pnl=sum(row["pnl"] for row in pnl_data),
    )
    HTML(string=html, base_url=str(Path(__file__).parent)).write_pdf(output_path)


# templates/daily_report.html.j2:
"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: sans-serif; }
    h1 { color: #2c3e50; border-bottom: 2px solid #305496; padding-bottom: 0.3em; }
    table { border-collapse: collapse; width: 100%; margin-top: 1em; }
    th { background: #305496; color: white; padding: 8px; }
    td { padding: 8px; border-bottom: 1px solid #eee; }
    .negative { color: #c0392b; }
    .total { font-weight: bold; border-top: 2px solid #333; }
  </style>
</head>
<body>
  <h1>Daily P&L — {{ report_date }}</h1>
  <table>
    <tr><th>Strategy</th><th>P&L</th><th>Notional</th></tr>
    {% for row in pnl_data %}
    <tr>
      <td>{{ row.strategy }}</td>
      <td class="{% if row.pnl < 0 %}negative{% endif %}">${{ '{:+,.2f}'.format(row.pnl) }}</td>
      <td>${{ '{:,.0f}'.format(row.notional) }}</td>
    </tr>
    {% endfor %}
    <tr class="total"><td>Total</td><td>${{ '{:+,.2f}'.format(total_pnl) }}</td><td></td></tr>
  </table>
</body>
</html>
"""
```

Clean separation of template + data. The same pattern produces invoices, statements, audit reports.

## Embedding fonts and avoiding licensing issues

PDFs can embed fonts. For consistent rendering across systems, embed:

```python
# ReportLab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("InterRegular", "fonts/Inter-Regular.ttf"))
pdfmetrics.registerFont(TTFont("InterBold", "fonts/Inter-Bold.ttf"))
```

Use open-licensed fonts (Inter, Source Sans, IBM Plex) — bundling commercial fonts in a redistributable PDF often violates licenses.

## Pitfalls

!!! warning "WeasyPrint dependencies"
    WeasyPrint depends on Cairo and Pango (native libraries). On Linux: `apt install libpango-1.0-0 libcairo2 libffi-dev`. On macOS: `brew install pango cairo`. Often the source of installation pain.

!!! warning "ReportLab's coordinate system"
    Origin is at *bottom-left*, not top-left. Counter-intuitive for anyone coming from web layout.

!!! warning "Fonts that don't have your glyphs"
    A font that doesn't include Cyrillic / CJK / emoji renders missing-glyph boxes. Embed a font with full Unicode coverage (e.g., Noto) if you support international text.

!!! warning "Page size confusion"
    `letter` (US) vs `A4` (rest of the world) is 8.5"×11" vs 210×297mm. Default to A4 unless you have a US-specific requirement.

## Bottom line

For PDF generation:

- **WeasyPrint + Jinja2** for HTML-template-driven reports — fastest path for new code.
- **ReportLab** for fine-grained programmatic PDFs.
- **Embed charts** as PNG (from matplotlib) or SVG.
- **Embed fonts** for consistent rendering.
- **Test on the target output device** — fonts and colours can drift.

Continue to **[Image fundamentals with Pillow](03-pillow.md)**.
