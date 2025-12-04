from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .personalization import price_bucket, UserProfiles
from .query_understanding import extract_cuisine_entities
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
    feature_rows: List[FeatureRow] = []
    catalog_by_id = catalog.set_index("item_id")

    for _, row in labeled_data.iterrows():
        item = catalog_by_id.loc[row["item_id"]]
        lexical_score, semantic_score = retriever.pair_scores(row["query"], row["item_id"])
        user_pref = user_profiles.score(row["user_id"], item["cuisine"])
        intent = intent_predictor.predict([row["query"]])[0]
        intent_code = INTENT_MAP.get(intent, 0)
        cuisines_in_query = extract_cuisine_entities(row["query"], cuisines)
        cuisine_match = 1.0 if item["cuisine"].lower() in cuisines_in_query else 0.0

        features = {
            "lexical_score": float(lexical_score),
            "semantic_score": float(semantic_score),
            "rating": float(item["rating"]),
            "popularity": float(item["popularity"]),
            "is_vegan_friendly": float(bool(item["is_vegan_friendly"])),
            "delivery_time_minutes": float(item["delivery_time_minutes"]),
            "price_bucket": float(price_bucket(item["price_range"])),
            "user_pref_score": float(user_pref),
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
