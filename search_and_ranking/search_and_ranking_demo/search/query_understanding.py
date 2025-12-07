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

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .spell import SpellCorrector


def normalize_query(text: str) -> str:
    """Lowercase and strip extra punctuation."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class IntentClassifier:
    """Thin wrapper around a TF-IDF + Logistic Regression classifier."""

    def __init__(self, max_features: int = 800):
        self.pipeline: Pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        self.is_fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        self.pipeline.fit(texts, labels)
        self.is_fitted = True

    def predict(self, queries: Sequence[str]) -> List[str]:
        if not self.is_fitted:
            # Default to product_search to stay permissive.
            return ["product_search" for _ in queries]
        return list(self.pipeline.predict(queries))


def build_cuisine_lexicon(catalog: pd.DataFrame) -> set[str]:
    cuisines = set(catalog["cuisine"].str.lower().unique())
    cuisines.update({"vegan", "vegetarian"})
    return cuisines


SYNONYMS: Dict[str, List[str]] = {
    "vegan": ["plant based", "plant-based"],
    "vegetarian": ["veggie"],
    "cheap": ["budget", "affordable"],
    "expensive": ["premium", "fancy"],
}


def expand_query_with_synonyms(query: str) -> List[str]:
    expansions: List[str] = []
    lowered = query.lower()
    for term, syns in SYNONYMS.items():
        if term in lowered:
            expansions.extend(syns)
    return expansions


DIETARY_TAGS = {"vegan", "vegetarian", "gluten free", "gluten-free"}


def extract_dietary_tags(query: str) -> List[str]:
    tokens = normalize_query(query)
    tags = []
    for tag in DIETARY_TAGS:
        if tag.replace("-", " ") in tokens:
            tags.append(tag)
    return tags


def extract_price_range(query: str) -> str | None:
    tokens = normalize_query(query).split()
    if any(t in tokens for t in ["cheap", "budget", "affordable", "low"]):
        return "cheap"
    if any(t in tokens for t in ["expensive", "premium", "fancy", "high"]):
        return "expensive"
    if "medium" in tokens or "mid" in tokens:
        return "medium"
    return None


def extract_cuisine_entities(query: str, cuisines: Iterable[str]) -> List[str]:
    tokens = set(normalize_query(query).split())
    return [c for c in cuisines if c in tokens]


@dataclass
class UnderstoodQuery:
    raw: str
    normalized: str
    corrected: str
    intent: str
    cuisines: List[str]
    expansions: List[str]
    dietary_tags: List[str]
    price_hint: str | None


def understand_query(
    query: str,
    spell_corrector: SpellCorrector,
    intent_classifier: IntentClassifier,
    cuisines: Iterable[str],
) -> UnderstoodQuery:
    normalized = normalize_query(query)
    corrected = spell_corrector.correct(normalized)
    intent = intent_classifier.predict([corrected])[0]
    entities = extract_cuisine_entities(corrected, cuisines)
    expansions = expand_query_with_synonyms(corrected)
    dietary_tags = extract_dietary_tags(corrected)
    price_hint = extract_price_range(corrected)
    return UnderstoodQuery(
        raw=query,
        normalized=normalized,
        corrected=corrected,
        intent=intent,
        cuisines=entities,
        expansions=expansions,
        dietary_tags=dietary_tags,
        price_hint=price_hint,
    )
