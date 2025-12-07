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

from __future__ import annotations

from typing import Dict, List, Optional

CATEGORY_KEYWORDS = {
    "seafood": ["seafood", "fish", "lobster", "grill"],
    "japanese": ["sushi", "japanese", "maki", "nigiri"],
    "italian": ["pizza", "pasta", "italian", "trattoria"],
    "mexican": ["taco", "burrito", "salsa", "mexican"],
    "indian": ["curry", "tandoor", "naan", "indian"],
    "thai": ["thai", "curry", "stir fry"],
    "american": ["burger", "fries", "american"],
    "vegan": ["vegan", "plant-based", "plant based"],
    "vegetarian": ["vegetarian", "veggie"],
}

DIETARY_KEYWORDS = {
    "vegan": ["vegan", "plant-based", "plant based"],
    "vegetarian": ["vegetarian", "veggie"],
    "gluten_free": ["gluten-free", "gluten free"],
}


def extract_attributes(
    description: str, cuisine: Optional[str] = None, price_range: Optional[str] = None
) -> Dict[str, str | bool | List[str]]:
    """
    Structured-ish attribute extractor to mirror an ontology/attribute pipeline (heuristic).
    Returns a dict with category, cuisine, dietary tags, price_level, and boolean flags.
    """
    text = (description or "").lower()
    attrs: Dict[str, str | bool | List[str]] = {}

    # Cuisine/category from provided cuisine field.
    if cuisine:
        attrs["cuisine"] = cuisine.lower()

    # Category from keywords.
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords) or (cuisine and cat in cuisine.lower()):
            attrs["category"] = cat
            break

    # Dietary tags.
    dietary_tags: List[str] = []
    for tag, keywords in DIETARY_KEYWORDS.items():
        if any(k in text for k in keywords):
            dietary_tags.append(tag)
    if dietary_tags:
        attrs["dietary"] = dietary_tags
    if "vegan" in dietary_tags:
        attrs["is_vegan_friendly"] = True
    if "gluten_free" in dietary_tags:
        attrs["gluten_free"] = True

    # Price level from price_range if available.
    if price_range:
        attrs["price_level"] = str(price_range).lower()

    return attrs
