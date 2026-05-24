# Reading PDFs with PyMuPDF

PDFs come in three flavours: text-based (selectable text), image-based (scans, no extractable text), and hybrid (mix). The right extraction tool depends on which you have. **PyMuPDF** (imported as `fitz`) is the best general-purpose reader; for image-only PDFs you'll add OCR (chapter 5).

## The basics

```python
import fitz                    # PyMuPDF


doc = fitz.open("invoice.pdf")
print(f"Pages: {len(doc)}")
print(f"Metadata: {doc.metadata}")

for page_num, page in enumerate(doc):
    text = page.get_text()
    print(f"--- Page {page_num + 1} ---")
    print(text[:500])
```

`get_text()` extracts text as the PDF stores it — usually in reading order, but PDFs are weird. For sophisticated layout handling, see below.

## Extracting structured text

```python
page = doc[0]

# Get text with positions
text_dict = page.get_text("dict")
for block in text_dict["blocks"]:
    if block["type"] == 0:        # text block (vs image)
        for line in block["lines"]:
            for span in line["spans"]:
                print(f"({span['bbox']}) [{span['font']}, {span['size']}] {span['text']}")
```

`get_text("dict")` returns nested blocks → lines → spans. Each span has bounding box, font, size. For invoices, receipts, financial statements where layout matters, this is the right tool.

## Extracting tables

PDF tables are not first-class structures; they're just text positioned to look tabular. PyMuPDF has a table detector:

```python
page = doc[0]
tables = page.find_tables()
print(f"Found {len(tables.tables)} tables")

for tab in tables.tables:
    df = tab.to_pandas()
    print(df)
```

Works well on clean PDFs with visible borders. For noisier PDFs, [`camelot`](https://camelot-py.readthedocs.io/) or [`pdfplumber`](https://github.com/jsvine/pdfplumber) often work better. Try all three; pick the one with the cleanest output for your input.

## Bookmarks and structure

```python
toc = doc.get_toc()
for level, title, page_num in toc:
    print(f"{'  ' * (level - 1)}{title} (p.{page_num})")
```

Useful for big documents where you only want certain sections.

## Extracting images

```python
for page_num, page in enumerate(doc):
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base = doc.extract_image(xref)
        ext = base["ext"]
        data = base["image"]
        Path(f"page_{page_num}_img_{img_index}.{ext}").write_bytes(data)
```

Extracts every embedded image. For invoices with logos, scanned receipts in a PDF wrapper, this gets you the raw image to OCR.

## Searching

```python
page = doc[0]
matches = page.search_for("Invoice Number")
for rect in matches:
    print(f"Found at: {rect}")
    # rect is a fitz.Rect; you can extract nearby text
    nearby = page.get_text("text", clip=rect + (0, 0, 200, 0))
    print(f"Context: {nearby}")
```

Layout-aware text search. The pattern: find a label, then read text immediately after it. Useful for "extract the invoice number from this PDF."

## Annotations

```python
for page in doc:
    for annot in page.annots() or []:
        print(annot.type, annot.info)
```

Catches highlights, notes, form-field values — useful when your input PDFs have user-added markup.

## A worked example: invoice field extraction

```python
import fitz
import re


def extract_invoice_fields(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    page = doc[0]
    text = page.get_text()

    patterns = {
        "invoice_number": r"Invoice (?:#|Number)[:\s]+(\S+)",
        "date": r"Date[:\s]+(\d{4}-\d{2}-\d{2})",
        "total": r"Total[:\s]+\$([\d,]+\.\d{2})",
        "vendor": r"From[:\s]+(.+?)(?:\n|$)",
    }
    result = {}
    for field, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        result[field] = m.group(1).strip() if m else None
    doc.close()
    return result


fields = extract_invoice_fields("invoice_2024-11.pdf")
print(fields)
# {'invoice_number': 'INV-12345', 'date': '2024-11-15', 'total': '1,250.00', 'vendor': 'Vendor Co'}
```

For a few hundred invoices a month, regex on extracted text works fine. For thousands, train a layout-aware model (e.g., `donut`) — but only if regex isn't enough.

## When text extraction fails

Symptoms:

- `page.get_text()` returns empty string.
- Text comes out scrambled or as Unicode private-use glyphs.

Causes:

- **Image-only PDF.** Use OCR (chapter 5).
- **Embedded weird fonts.** Try `page.get_text("rawdict")` to see what's there.
- **DRM / encrypted PDF.** `doc.is_encrypted` checks; `doc.authenticate(password)` unlocks if you have the password.

## Rendering pages to images

For OCR or visual diff:

```python
page = doc[0]
pix = page.get_pixmap(dpi=300)        # 300 DPI = high quality
pix.save("page1.png")
```

DPI tuning matters: 150 is fine for screen display, 300 for OCR, 600 for archival.

## Modifying PDFs (briefly)

PyMuPDF can also write PDFs:

```python
doc = fitz.open("input.pdf")
page = doc[0]
page.insert_text((100, 100), "ADDED TEXT", fontsize=14, color=(1, 0, 0))
doc.save("output.pdf")
doc.close()
```

For richer generation (charts, tables, styled reports), use ReportLab or WeasyPrint (next chapter).

## `pdfplumber` for layout-heavy work

For tables that PyMuPDF struggles with, try `pdfplumber`:

```python
import pdfplumber

with pdfplumber.open("statement.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(row)
```

Different algorithm; different strengths. For bank statements and financial PDFs, often better than PyMuPDF.

## Pitfalls

!!! warning "Memory on huge PDFs"
    A 1000-page PDF can OOM. Iterate page-by-page; don't materialise all text in memory.

!!! warning "OCR-quality PDFs that look like text"
    Some PDFs have invisible OCR layers on top of scanned images. The text is extractable but inaccurate. Cross-check key fields against the image.

!!! warning "Unicode normalisation"
    Extracted text often has unusual whitespace, non-breaking spaces, ligatures. Normalise with `unicodedata.normalize("NFKC", text)` for consistent regex matching.

!!! warning "Encrypted PDFs**
    Many vendor invoices arrive encrypted with the recipient's name as password. Check `doc.is_encrypted`; pass the password explicitly.

## Bottom line

For reading PDFs:

- **PyMuPDF** (`fitz`) as the default — fast, accurate, handles most cases.
- **`page.get_text("dict")`** for layout-aware extraction.
- **`page.find_tables()`** then `pdfplumber` then `camelot` for tables.
- **OCR (chapter 5)** for image-only PDFs.
- **Normalise Unicode** before regex.

Continue to **[Generating PDFs](02-pdf-generation.md)**.
