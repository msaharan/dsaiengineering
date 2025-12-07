# Copyright 2025 Mohit Saharan (github.com/msaharan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract book titles/authors from shelf photos using PaddleOCR.

Requires the `dsaie` conda environment with paddleocr installed:
    conda run -n dsaie pip install "paddleocr>=2.7.0" opencv-python

Usage (from this folder):
    conda run -n dsaie python extract_books.py
"""

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from paddleocr import PaddleOCR  # type: ignore


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
CSV_PATH = OUTPUT_DIR / "books.csv"
MIN_CONF = 0.7
TARGET_BOOKS = 19


def build_ocr() -> PaddleOCR:
    # Keep angle classification for rotated spines; avoid doc pipeline params
    # that are not supported in this PaddleOCR version.
    return PaddleOCR(use_angle_cls=True, lang="en")


def run_ocr(ocr: PaddleOCR, image_path: Path) -> List[Dict]:
    """
    Run PaddleOCR on an image and return line-level dicts with geometry.
    """
    results = ocr.ocr(str(image_path), cls=True)
    lines: List[Dict] = []
    for page in results:
        for bbox, (text, conf) in page:
            if not text.strip() or conf < MIN_CONF:
                continue
            xs, ys = zip(*bbox)
            top = min(ys)
            bottom = max(ys)
            left = min(xs)
            height = bottom - top
            center_y = (top + bottom) / 2
            lines.append(
                {
                    "text": " ".join(text.split()),
                    "conf": conf,
                    "top": top,
                    "height": height,
                    "center_y": center_y,
                    "left": left,
                }
            )
    return lines


def cluster_spines(lines: Sequence[Dict]) -> List[List[Dict]]:
    """
    Group lines into spines by clustering on vertical position.
    We split along the largest vertical gaps to target roughly TARGET_BOOKS groups.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda x: x["center_y"])
    gaps = [
        (sorted_lines[i + 1]["center_y"] - sorted_lines[i]["center_y"], i)
        for i in range(len(sorted_lines) - 1)
    ]
    gaps.sort(reverse=True, key=lambda x: x[0])
    split_count = max(0, min(TARGET_BOOKS - 1, len(gaps)))
    split_points = sorted(idx for _, idx in gaps[:split_count])

    groups: List[List[Dict]] = []
    start = 0
    for idx in split_points:
        groups.append(sorted_lines[start : idx + 1])
        start = idx + 1
    groups.append(sorted_lines[start:])

    # left-to-right order inside each spine
    for g in groups:
        g.sort(key=lambda x: x["left"])

    return groups


def select_title_author(spine: Sequence[Dict]) -> Tuple[str, str]:
    """
    Simple extraction: longest line -> title; remaining -> author blob.
    """
    if not spine:
        return "", ""
    texts = [l["text"] for l in spine if l["text"]]
    if not texts:
        return "", ""

    # Pick title as the longest text line; author as the next longest (if any).
    sorted_texts = sorted(texts, key=len, reverse=True)
    longest = sorted_texts[0]
    author_candidates = sorted_texts[1:3]  # up to two remaining lines
    others = [t for t in author_candidates if t != longest]
    title = longest.strip()
    author = " | ".join(others).strip()
    return title, author


def process_image(ocr: PaddleOCR, image_path: Path) -> List[Dict]:
    lines = run_ocr(ocr, image_path)
    spines = cluster_spines(lines)
    rows = []
    for spine in spines:
        text = " | ".join(l["text"] for l in spine)
        rows.append({"book_text": text})
    return rows


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    images = sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise SystemExit(f"No images found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    ocr = build_ocr()

    all_rows: List[Dict] = []
    for image_path in images:
        if image_path.stat().st_size == 0:
            print(f"Skipping empty file: {image_path.name}")
            continue
        print(f"OCR: {image_path.name}")
        all_rows.extend(process_image(ocr, image_path))

    if not all_rows:
        raise SystemExit("No text extracted.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["book_text"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
