from __future__ import annotations

from typing import Dict

import pandas as pd


PRICE_BUCKET = {"cheap": 0, "medium": 1, "expensive": 2}


def price_bucket(value: str) -> int:
    return PRICE_BUCKET.get(str(value).lower(), 1)


class UserProfiles:
    """Tiny user-profile builder based on cuisine preferences."""

    def __init__(self, interactions: pd.DataFrame, catalog: pd.DataFrame):
        merged = interactions.merge(
            catalog[["item_id", "cuisine"]], how="left", on="item_id"
        )
        scores = (
            merged.groupby(["user_id", "cuisine"])["relevance"]
            .sum()
            .reset_index()
        )

        profiles: Dict[str, Dict[str, float]] = {}
        for _, row in scores.iterrows():
            user = row["user_id"]
            cuisine = str(row["cuisine"]).lower()
            profiles.setdefault(user, {})[cuisine] = float(row["relevance"])

        # Normalize per user to keep scores in [0,1].
        for user, prefs in profiles.items():
            total = sum(prefs.values()) or 1.0
            for cuisine in list(prefs.keys()):
                prefs[cuisine] = prefs[cuisine] / total
        self.profiles = profiles

    def score(self, user_id: str, cuisine: str) -> float:
        return self.profiles.get(user_id, {}).get(str(cuisine).lower(), 0.0)
