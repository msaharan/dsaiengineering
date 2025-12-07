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

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .personalization import price_bucket, UserProfiles
from .query_understanding import (
    extract_cuisine_entities,
    extract_dietary_tags,
    extract_price_range,
)
from .retrieval import HybridRetriever


FEATURE_COLUMNS = [
    "lexical_score",
    "semantic_score",
    "rating",
    "popularity",
    "is_vegan_friendly",
    "delivery_time_minutes",
    "price_bucket",
    "user_pref_score",
    "price_affinity",
    "user_item_bias",
    "ontology_dietary_match",
    "ontology_category_match",
    "price_hint_match",
    "cuisine_match",
    "intent_code",
]

INTENT_MAP = {"product_search": 0, "faq_search": 1, "local_search": 2}


@dataclass
class FeatureRow:
    query_id: str
    query: str
    user_id: str
    item_id: int
    features: Dict[str, float]
    label: float


def build_feature_rows(
    labeled_data: pd.DataFrame,
    catalog: pd.DataFrame,
    retriever: HybridRetriever,
    user_profiles: UserProfiles,
    intent_predictor,
    cuisines: Sequence[str],
) -> List[FeatureRow]:
    """Assemble per-(query,item) feature rows from retrieval scores, user prefs, ontology cues, and labels."""
    feature_rows: List[FeatureRow] = []
    catalog_by_id = catalog.set_index("item_id")

    for _, row in labeled_data.iterrows():
        item = catalog_by_id.loc[row["item_id"]]
        lexical_score, semantic_score = retriever.pair_scores(row["query"], row["item_id"])
        if retriever.semantic is None:
            semantic_score = lexical_score  # fallback to lexical when semantic is disabled
        user_pref = user_profiles.score(row["user_id"], item["cuisine"])
        price_affinity = user_profiles.price_affinity(
            row["user_id"], price_bucket(item["price_range"])
        )
        bias = user_profiles.item_bias(row["user_id"], int(row["item_id"]))
        diet_tags = extract_dietary_tags(row["query"])
        price_hint = extract_price_range(row["query"])
        intent = intent_predictor.predict([row["query"]])[0]
        intent_code = INTENT_MAP.get(intent, 0)
        cuisines_in_query = extract_cuisine_entities(row["query"], cuisines)
        cuisine_match = 1.0 if item["cuisine"].lower() in cuisines_in_query else 0.0
        ontology = {}
        if hasattr(item, "get"):
            ontology = item.get("ontology_attrs", {}) or {}
        dietary_attr = False
        category_attr = None
        if isinstance(ontology, dict):
            diet_field = ontology.get("dietary", [])
            if isinstance(diet_field, str):
                diet_field = [diet_field]
            dietary_attr = bool(ontology.get("is_vegan_friendly", False)) or (
                set(diet_field).intersection(set(diet_tags)) if diet_tags else False
            )
            category_attr = ontology.get("category")
        ontology_dietary_match = 1.0 if dietary_attr or (diet_tags and bool(item.get("is_vegan_friendly", False))) else 0.0
        ontology_category_match = 1.0 if category_attr and category_attr.lower() in cuisines_in_query else 0.0
        price_hint_match = 1.0 if price_hint and price_hint.lower() == str(item["price_range"]).lower() else 0.0

        features = {
            "lexical_score": float(lexical_score),
            "semantic_score": float(semantic_score),
            "rating": float(item["rating"]),
            "popularity": float(item["popularity"]),
            "is_vegan_friendly": float(bool(item["is_vegan_friendly"])),
            "delivery_time_minutes": float(item["delivery_time_minutes"]),
            "price_bucket": float(price_bucket(item["price_range"])),
            "user_pref_score": float(user_pref),
            "price_affinity": float(price_affinity),
            "user_item_bias": float(bias),
            "ontology_dietary_match": float(ontology_dietary_match),
            "ontology_category_match": float(ontology_category_match),
            "price_hint_match": float(price_hint_match),
            "cuisine_match": float(cuisine_match),
            "intent_code": float(intent_code),
        }
        feature_rows.append(
            FeatureRow(
                query_id=str(row["query_id"]),
                query=row["query"],
                user_id=row["user_id"],
                item_id=int(row["item_id"]),
                features=features,
                label=float(row["relevance"]),
            )
        )
    return feature_rows


def build_matrices(rows: List[FeatureRow]) -> Tuple[np.ndarray, np.ndarray, List[int], List[FeatureRow]]:
    """Convert feature rows into X, y, and group sizes (by query_id) for ranking models."""
    X = np.array([[row.features[col] for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([row.label for row in rows], dtype=float)
    group_counts: Dict[str, int] = {}
    for row in rows:
        group_counts[row.query_id] = group_counts.get(row.query_id, 0) + 1
    group = list(group_counts.values())
    return X, y, group, rows


class Ranker:
    """Use XGBRanker if available, otherwise a RandomForest regressor fallback."""

    def __init__(self):
        self.model = None
        self.uses_xgb = False
        try:
            from xgboost import XGBRanker  # type: ignore

            self.model = XGBRanker(
                objective="rank:pairwise",
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            self.uses_xgb = True
        except ImportError:
            self.model = RandomForestRegressor(n_estimators=120, random_state=42)
            self.uses_xgb = False

    def fit(self, X: np.ndarray, y: np.ndarray, group: Sequence[int]) -> None:
        if self.uses_xgb:
            self.model.fit(X, y, group=group)
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.asarray(preds)
