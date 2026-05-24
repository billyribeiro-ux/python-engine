# Image fundamentals with Pillow

[Pillow](https://pillow.readthedocs.io/) (the maintained fork of PIL) is Python's general-purpose image library. Opening, resizing, cropping, drawing text, format conversion — all the basics + a lot of the advanced.

## Basics

```python
from PIL import Image


img = Image.open("input.png")
print(img.size, img.mode)        # (1920, 1080) 'RGB'

# Resize
small = img.resize((400, 225), Image.LANCZOS)
small.save("output.png", optimize=True)

# Crop (left, upper, right, lower)
crop = img.crop((100, 100, 500, 400))
crop.save("crop.png")

# Convert format
img.save("output.jpg", quality=85)
img.save("output.webp", quality=85)
```

`Image.LANCZOS` is the high-quality downsampling filter; `Image.NEAREST` is fastest but pixelated; defaults are reasonable.

## Modes

| Mode | Meaning |
|---|---|
| `L` | 8-bit grayscale |
| `RGB` | 3-channel colour |
| `RGBA` | RGB + alpha (transparency) |
| `CMYK` | print colour space |
| `1` | binary (1-bit) |

Convert with `img.convert("RGB")`. For drawing or saving JPEG, you need `RGB` (JPEG doesn't support alpha).

## Drawing

```python
from PIL import Image, ImageDraw, ImageFont


img = Image.new("RGB", (800, 200), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 48)

draw.text((20, 50), "Hello, world", fill="black", font=font)
draw.rectangle((20, 120, 200, 180), outline="red", width=3)
draw.line((200, 150, 600, 150), fill="blue", width=2)

img.save("hello.png")
```

The `ImageDraw` API has all the primitives: `text`, `rectangle`, `ellipse`, `line`, `polygon`, `arc`.

For text with measured size:

```python
# Pillow 10+
bbox = draw.textbbox((0, 0), "hello", font=font)
width = bbox[2] - bbox[0]
height = bbox[3] - bbox[1]
```

Useful for centring text or laying out tables.

## Working with thumbnails

```python
from PIL import Image


img = Image.open("big.jpg")
img.thumbnail((300, 300))         # in-place, preserves aspect
img.save("thumb.jpg")
```

`thumbnail` mutates the image; it's the cheapest "make me a thumbnail" call.

## EXIF metadata

```python
from PIL import Image
from PIL.ExifTags import TAGS


img = Image.open("photo.jpg")
exif = img.getexif()
for tag_id, value in exif.items():
    tag = TAGS.get(tag_id, tag_id)
    print(f"{tag}: {value}")
```

Useful when photos arrive with embedded GPS / camera info you want to strip before publishing.

To strip:

```python
img = Image.open("photo.jpg")
img.getexif().clear()
img.save("clean.jpg")
```

## Colour adjustments

```python
from PIL import ImageEnhance


img = Image.open("input.png")
img = ImageEnhance.Brightness(img).enhance(1.2)       # 20% brighter
img = ImageEnhance.Contrast(img).enhance(1.5)         # 50% more contrast
img = ImageEnhance.Color(img).enhance(1.3)            # 30% more saturated
img.save("adjusted.png")
```

For more sophisticated colour science, `colour` or `colorspacious` libraries.

## Filters

```python
from PIL import ImageFilter


img = Image.open("input.png")
blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
sharpened = img.filter(ImageFilter.SHARPEN)
edges = img.filter(ImageFilter.FIND_EDGES)
```

For more, OpenCV (chapter 4).

## NumPy interop

Often you want to manipulate pixels directly:

```python
import numpy as np
from PIL import Image


img = Image.open("photo.jpg")
arr = np.array(img)                                   # H × W × 3
print(arr.shape, arr.dtype)                            # (1080, 1920, 3) uint8

# Threshold to monochrome
gray = arr.mean(axis=2).astype("uint8")
bw = np.where(gray > 128, 255, 0).astype("uint8")
Image.fromarray(bw, mode="L").save("bw.png")
```

`np.array(img)` is one of the most useful interop points in scientific Python.

## Batch processing

```python
from pathlib import Path
from PIL import Image


def make_thumbnails(input_dir: Path, output_dir: Path, size=(300, 300)):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.glob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        with Image.open(path) as img:
            img.thumbnail(size)
            img.save(output_dir / path.name)
```

The `with Image.open(...)` pattern ensures the file handle closes — important when batch-processing thousands.

## Compositing — layering images

```python
base = Image.open("base.png").convert("RGBA")
logo = Image.open("logo.png").convert("RGBA")
position = (base.width - logo.width - 20, base.height - logo.height - 20)
base.paste(logo, position, logo)       # third arg: use logo's alpha as mask
base.save("composite.png")
```

For watermarking, the third argument (alpha mask) is critical — without it, you'd paste a opaque rectangle.

## Format conversion at scale

```python
def png_to_webp(input_dir: Path, output_dir: Path, quality: int = 85):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.glob("*.png"):
        with Image.open(path) as img:
            img.save(output_dir / f"{path.stem}.webp", "WEBP", quality=quality)
```

WebP is typically 30-50% smaller than PNG at similar visual quality. For websites and reports, worth converting.

## Pitfalls

!!! warning "Saving JPEG with mode=RGBA"
    JPEG doesn't support transparency. `img.convert("RGB")` before saving as JPEG, or use PNG / WebP.

!!! warning "EXIF rotation"
    Photos from phones often have EXIF orientation tags. `img.size` may say `(4000, 3000)` even though the photo is portrait. `ImageOps.exif_transpose(img)` rotates per the EXIF tag.

!!! warning "Pillow ≠ PIL"
    PIL was abandoned long ago. Pillow is the maintained fork. `pip install pillow` (not `pip install PIL`).

!!! warning "Memory on huge images"
    A 100-megapixel image is 400 MB in memory as `uint8`. Streaming-friendly libraries (libvips via `pyvips`) handle these better.

## Bottom line

For images:

- **Pillow** for general work; **OpenCV** when you need real CV (next chapter).
- **`LANCZOS`** for high-quality downsampling.
- **NumPy interop** via `np.array(img)` ↔ `Image.fromarray(arr)`.
- **Strip EXIF** before publishing photos.
- **WebP** for size.

Continue to **[Image pipelines](04-image-pipelines.md)**.
