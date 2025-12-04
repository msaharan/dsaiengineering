from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

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
    return UnderstoodQuery(
        raw=query,
        normalized=normalized,
        corrected=corrected,
        intent=intent,
        cuisines=entities,
    )
