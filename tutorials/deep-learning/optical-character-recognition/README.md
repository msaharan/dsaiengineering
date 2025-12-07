# README

## Resources for image to text conversion on local machine

---

## 1. Tesseract-based (most common & fully local)

### `pytesseract`

* **What it is:** Python wrapper around the Tesseract OCR engine.
* **Pros:** Mature, works fully offline, supports many languages, easy to use.
* **Cons:** Not as good as newer DL models on messy handwriting or weird layouts.

```bash
# Install the engine (example: Ubuntu)
sudo apt-get install tesseract-ocr

# Then install the Python wrapper
pip install pytesseract pillow
```

Minimal example:

```python
from PIL import Image
import pytesseract

img = Image.open("page.png")
text = pytesseract.image_to_string(img, lang="eng")
print(text)
```

### `tesserocr`

* Another wrapper for Tesseract, using its C++ API directly.
* Slightly faster / more control than `pytesseract`, but a bit trickier to install.

```bash
pip install tesserocr
```

---

## 2. Deep-learning OCR libraries

These usually perform better on noisy images, multiple scripts, and complex fonts.

### `easyocr`

* Uses PyTorch and pretrained models for many languages.
* Very simple interface.

```bash
pip install easyocr
```

```python
import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext("page.png", detail=0)  # just text
print("\n".join(result))
```

### `paddleocr`

* Part of Baidu’s PaddlePaddle ecosystem.
* Strong performance, lots of languages, layout support.

```bash
pip install "paddleocr>=2.0.1"  # also installs paddlepaddle if possible
```

### `python-doctr` (docTR)

* OCR + layout analysis (detects blocks, words, etc.).
* Good if you care about structure, not just raw text.

```bash
pip install python-doctr[torch]  # or [tensorflow]
```

---

## 3. Helpful companions

### `opencv-python`

* Not an OCR engine, but great for preprocessing (binarization, deskewing, denoising).

```bash
pip install opencv-python
```

### `ocrmypdf`

* Uses Tesseract under the hood to add a text layer to PDFs.
* Perfect if you’re starting with scanned PDFs rather than images.

```bash
pip install ocrmypdf
```

---

## 4. How to choose

* **Simple, reliable, offline text from images:**
  → `pytesseract` (+ `opencv-python` for preprocessing).

* **Need better accuracy on messy images or multiple scripts/languages:**
  → `easyocr` or `paddleocr`.

* **Care about document layout / bounding boxes / structured output:**
  → `python-doctr`, `paddleocr` (with detection), or `easyocr` (with `detail=True`).

* **Scanned PDFs → searchable PDFs:**
  → `ocrmypdf` (internally uses Tesseract).

---

## 5. LLM-based OCR

### OpenAI Vision API

```bash
pip install openai
```

```python
import openai

client = openai.OpenAI()
```
Read more [here](https://platform.openai.com/docs/guides/images-vision).
