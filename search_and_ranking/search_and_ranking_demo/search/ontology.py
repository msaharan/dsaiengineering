from __future__ import annotations

from typing import Dict


def extract_attributes(description: str) -> Dict[str, str | bool]:
    """Heuristic attribute extractor to mirror the ontology section (no external LLM)."""
    text = description.lower()
    attrs: Dict[str, str | bool] = {}
    if any(k in text for k in ["vegan", "plant"]):
        attrs["dietary"] = "vegan"
        attrs["is_vegan_friendly"] = True
    if "gluten" in text:
        attrs["gluten_free"] = True
    if any(k in text for k in ["seafood", "fish", "lobster"]):
        attrs["category"] = "seafood"
    if any(k in text for k in ["sushi", "japanese"]):
        attrs["category"] = "japanese"
    if any(k in text for k in ["pizza", "pasta", "italian"]):
        attrs["category"] = "italian"
    return attrs
