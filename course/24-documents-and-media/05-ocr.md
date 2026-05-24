# OCR with Tesseract

For image-only PDFs, scanned receipts, screenshots — anything without an extractable text layer — OCR (Optical Character Recognition) is the bridge. **Tesseract** is the open-source standard; modern alternatives like Donut, TrOCR, and PaddleOCR can be more accurate but require ML stack overhead.

## Tesseract setup

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng
# For other languages: tesseract-ocr-fra, tesseract-ocr-deu, etc.

# Verify
tesseract --version
```

```bash
pip install pytesseract pillow
```

## Basic OCR

```python
from PIL import Image
import pytesseract


img = Image.open("receipt.jpg")
text = pytesseract.image_to_string(img)
print(text)
```

For most clean scans, that's it. For noisy scans, preprocessing dramatically improves accuracy.

## Preprocessing for accuracy

```python
import cv2
import numpy as np


def preprocess_for_ocr(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    # 3. Threshold (binarise) — Otsu auto-picks the threshold
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 4. Optional: morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned
```

Then OCR on the preprocessed:

```python
preprocessed = preprocess_for_ocr("receipt.jpg")
text = pytesseract.image_to_string(preprocessed, lang="eng")
```

Preprocessing typically lifts accuracy from 80% to 95%+ on phone-photographed receipts.

## Page segmentation mode

Tesseract has several PSM (page segmentation modes):

| PSM | Use case |
|---|---|
| 3 | default — auto page segmentation |
| 4 | single column of text of varying sizes |
| 6 | single uniform block of text |
| 7 | single text line |
| 8 | single word |
| 11 | sparse text (e.g., a screenshot with scattered labels) |

```python
text = pytesseract.image_to_string(img, config="--psm 6")
```

For receipts (one column, varying line sizes), PSM 4. For business cards (sparse text), PSM 11. Worth experimenting.

## Structured output

```python
# Get word-level bounding boxes
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
for i, word in enumerate(data["text"]):
    if word.strip():
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf = data["conf"][i]
        print(f"({x},{y},{w},{h}) conf={conf} text={word!r}")
```

Useful for "find the total amount" — locate the word "Total", then read text immediately to the right.

## A worked example: receipt parsing

```python
import re
import pytesseract
from PIL import Image


def parse_receipt(image_path: str) -> dict:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config="--psm 4")

    fields = {
        "total": None,
        "date": None,
        "vendor": None,
    }
    # Total
    m = re.search(r"total[:\s]+\$?([0-9.,]+)", text, re.IGNORECASE)
    if m:
        fields["total"] = float(m.group(1).replace(",", ""))
    # Date
    m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
    if m:
        fields["date"] = m.group(1)
    # Vendor: usually first non-trivial line
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 3 and not any(c.isdigit() for c in line):
            fields["vendor"] = line
            break
    return fields
```

For 80% of business receipts, this works. The remaining 20% (handwritten, very low contrast, rotated) need either better preprocessing or a deeper model.

## OCR on PDFs

```python
import fitz
import pytesseract
from PIL import Image
from io import BytesIO


def ocr_pdf(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)            # 300 DPI for OCR quality
        img = Image.open(BytesIO(pix.tobytes("png")))
        pages.append(pytesseract.image_to_string(img))
    return pages
```

Render → OCR per page. Use 300 DPI; lower hurts accuracy.

## Multiple languages

```bash
sudo apt install tesseract-ocr-fra tesseract-ocr-deu
```

```python
text = pytesseract.image_to_string(img, lang="eng+fra+deu")
```

Tesseract handles multilingual documents in one pass. Each `+`-separated language adds processing time.

## When Tesseract isn't enough

Modern alternatives:

- **PaddleOCR** — much better on Chinese / Japanese / multilingual; competitive on English.
- **EasyOCR** — easier API; modern deep-learning-based.
- **Donut** (Hugging Face) — end-to-end document understanding; transformer-based.
- **AWS Textract / Google Document AI / Azure Form Recogniser** — managed cloud services with much higher accuracy on forms / tables.

For a forms-heavy production workflow (invoices, tax forms, bank statements at scale), the cloud services often pay for themselves through better accuracy.

## Confidence-based filtering

```python
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
high_conf = [
    word for word, conf in zip(data["text"], data["conf"], strict=True)
    if word.strip() and int(conf) > 80
]
```

For automated workflows, filter low-confidence reads to a human queue rather than acting on them.

## Pitfalls

!!! warning "Skewed / rotated images"
    Tesseract handles small rotations but struggles past ~5°. Pre-deskew via OpenCV's `cv2.minAreaRect` + rotation.

!!! warning "Mixed languages without specifying**
    English-only Tesseract on Cyrillic text outputs gibberish. Set `lang=` correctly.

!!! warning "Numbers vs letters confusion"
    `O` vs `0`, `1` vs `l`, `5` vs `S`. For numeric fields, restrict the character set: `config='-c tessedit_char_whitelist=0123456789.,$- '`.

!!! warning "Tesseract performance**
    A 300-DPI page takes 1-3 seconds. For batch OCR of thousands of pages, parallelise with ProcessPoolExecutor.

## Bottom line

For OCR:

- **Tesseract via pytesseract** for general work.
- **Preprocess** (grayscale, denoise, Otsu threshold) for noisy scans.
- **PSM mode 4 or 6** for most document layouts.
- **Confidence filtering** for human review of uncertain reads.
- **Cloud services** (Textract / Document AI) when stakes are high.

Continue to **[Report generation](06-report-generation.md)**.
