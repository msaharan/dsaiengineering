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
LLM-based book spine extractor using OpenAI vision models.

Requirements:
    pip install openai
    export OPENAI_API_KEY=...

Usage:
    conda run -n dsaiecv python llm_extract_books.py \\
        --images data \\
        --output output/books_llm.csv \\
        --model gpt-4o-mini   # or gpt-4o if available
"""

import argparse
import base64
import json
import csv
from pathlib import Path
from typing import List, Dict

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"

PROMPT = """You are given a photo of a stack of books. Extract a clean, ordered list of ALL visible books with fields:
- title (full book title exactly as on the spine; fix casing/punctuation)
- author (primary author(s) from the spine; if absent, empty string)

Return JSON only, shaped exactly like:
{"books": [{"title": "...", "author": "..."}]}

Requirements:
- Keep the list in physical top-to-bottom order.
- Include every distinct spine you can see; do not drop uncertain ones—provide your best guess even if partially legible.
- Exclude publishers/series words unless part of the main title.
- Preserve edition words if on the spine (e.g., "Third Edition").
- Do not add explanatory text outside the JSON.
"""


def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision(model: str, image_path: Path) -> Dict:
    client = OpenAI()
    img_b64 = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def process_image(model: str, image_path: Path) -> List[Dict]:
    data = call_vision(model, image_path)
    books = data.get("books", [])
    rows = []
    for book in books:
        rows.append(
            {
                "title": book.get("title", "").strip(),
                "author": book.get("author", "").strip(),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract books via LLM vision.")
    parser.add_argument("--images", type=Path, default=Path("data"), help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("output/books_llm.csv"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.images.is_dir():
        image_paths = sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        image_paths = [args.images]

    if not image_paths:
        raise SystemExit(f"No images found at {args.images}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for img in image_paths:
        print(f"LLM OCR: {img.name}")
        rows = process_image(args.model, img)
        all_rows.extend(rows)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "author"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
