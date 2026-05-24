# Image pipelines

For batch image work — thumbnails, watermarking, conversion, content-aware filtering — Pillow is fine but slow at scale. **OpenCV** is faster, has more algorithms, and is the standard in computer vision. **`pyvips`** is the fastest streaming-friendly option for very large images.

This chapter covers the working patterns.

## OpenCV — when Pillow isn't enough

```python
import cv2
import numpy as np


img = cv2.imread("input.jpg")                # returns BGR (not RGB!)
print(img.shape, img.dtype)                  # (H, W, 3) uint8

# Resize
small = cv2.resize(img, (400, 225), interpolation=cv2.INTER_AREA)

# Color conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Save
cv2.imwrite("output.png", img)
```

OpenCV is BGR by default (legacy from OpenCV's C++ roots). When you pass arrays to matplotlib or Pillow, convert to RGB first: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

## Edge / contour detection

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("boxed.png", img)
```

Useful for "find the regions of interest" — extracting tables from screenshots, finding objects, etc.

## Template matching

```python
template = cv2.imread("logo.png", cv2.IMREAD_GRAYSCALE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
if max_val > 0.8:
    print(f"Found at {max_loc} with confidence {max_val:.2f}")
```

"Find this small image in this big image" — useful for UI test automation, watermark detection, etc.

## Watermarking

```python
def add_watermark(img_path: str, watermark_path: str, output: str,
                   position: str = "bottom-right", alpha: float = 0.5):
    img = cv2.imread(img_path)
    wm = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)   # keep alpha if PNG

    h, w = img.shape[:2]
    wh, ww = wm.shape[:2]

    positions = {
        "top-left": (10, 10),
        "top-right": (w - ww - 10, 10),
        "bottom-left": (10, h - wh - 10),
        "bottom-right": (w - ww - 10, h - wh - 10),
        "center": ((w - ww) // 2, (h - wh) // 2),
    }
    x, y = positions[position]

    # If the watermark has an alpha channel, use it; else simple alpha blend
    if wm.shape[2] == 4:
        wm_rgb = wm[:, :, :3]
        wm_a = wm[:, :, 3] / 255.0 * alpha
        roi = img[y:y+wh, x:x+ww]
        for c in range(3):
            roi[:, :, c] = (1 - wm_a) * roi[:, :, c] + wm_a * wm_rgb[:, :, c]
    else:
        roi = img[y:y+wh, x:x+ww]
        cv2.addWeighted(roi, 1 - alpha, wm, alpha, 0, roi)

    cv2.imwrite(output, img)
```

Bottom-right at 50% opacity is the canonical watermark placement.

## Batch resizing pipeline

```python
import asyncio
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def resize_one(args):
    src, dst, size = args
    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w >= h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return dst


def batch_resize(src_dir: Path, dst_dir: Path, size: int = 1024):
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png"))
    tasks = [(p, dst_dir / p.name, size) for p in paths]
    with ProcessPoolExecutor() as pool:
        pool.map(resize_one, tasks)
```

Image processing is CPU-bound, GIL-released (OpenCV does C work outside the GIL). Process pool gives near-linear scaling.

## `pyvips` — for very large images

For images > 100MP, `pyvips` is streaming-friendly:

```python
import pyvips

image = pyvips.Image.new_from_file("huge.tif", access="sequential")
thumbnail = image.thumbnail_image(1024)
thumbnail.write_to_file("thumb.jpg[Q=85]")
```

`access="sequential"` streams instead of loading the whole image. For TIFFs, satellite imagery, gigapixel scans, this is the only library that handles them efficiently.

## Computer vision basics

For object detection / face detection / OCR-like tasks:

- **Haar cascades** — old but cheap; `cv2.CascadeClassifier`. Face detection in 5 lines.
- **DNN module** — load pretrained models (Caffe, TF, PyTorch, ONNX). `cv2.dnn.readNetFromONNX("model.onnx")`.
- **`ultralytics` (YOLO)** — modern object detection. `pip install ultralytics`; one-liner inference.

For "find faces in this image":

```python
import cv2

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

Not state-of-the-art but very fast. For production face detection, use YOLO or a dedicated model.

## A worked example: redact faces in screenshots

```python
import cv2


def redact_faces(input_path: str, output_path: str):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (51, 51), 30)
        img[y:y+h, x:x+w] = blurred
    cv2.imwrite(output_path, img)
```

Useful for sanitising user-submitted screenshots before storage.

## Image deduplication via perceptual hashing

```python
from imagehash import phash
from PIL import Image
from pathlib import Path


def find_duplicates(directory: Path):
    hashes = {}
    for path in directory.glob("*.jpg"):
        h = phash(Image.open(path))
        hashes.setdefault(h, []).append(path)
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}
```

`phash` is a perceptual hash — visually similar images have similar hashes (Hamming distance), unlike md5. Catches near-duplicates with different compression / resolution.

## Pitfalls

!!! warning "OpenCV BGR vs RGB"
    Forgetting the conversion produces "blue-tinted" images when displayed with matplotlib / Pillow.

!!! warning "Memory on huge batches"
    Even with process pool, a 1000-image batch can hit memory limits. Stream the iteration; chunk the work.

!!! warning "OpenCV install pain"
    `pip install opencv-python` is the headless+gui version; `opencv-python-headless` is for servers (no GUI deps). Mixing both in the same env breaks.

!!! warning "Haar cascades give false positives"
    Detected "faces" in noise textures. Tune `scaleFactor` and `minNeighbors`; verify on representative data.

## Bottom line

For image pipelines:

- **Pillow** for simple ops; **OpenCV** for CV; **pyvips** for very large.
- **Process pool** for batch — image work is CPU-bound but GIL-released.
- **Perceptual hashing** for deduplication.
- **Always BGR-aware** when using OpenCV.

Continue to **[OCR with Tesseract](05-ocr.md)**.
