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

from typing import Dict

import pandas as pd


PRICE_BUCKET = {"cheap": 0, "medium": 1, "expensive": 2}


def price_bucket(value: str) -> int:
    return PRICE_BUCKET.get(str(value).lower(), 1)


class UserProfiles:
    """Tiny user-profile builder based on cuisine preferences, price affinity, and per-item bias."""

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

        # Price affinity: average price bucket per user.
        merged_price = interactions.merge(
            catalog[["item_id", "price_range"]], how="left", on="item_id"
        )
        price_pref = (
            merged_price.assign(price_bucket=merged_price["price_range"].map(price_bucket))
            .groupby("user_id")["price_bucket"]
            .mean()
            .to_dict()
        )
        self.price_pref = {k: float(v) for k, v in price_pref.items()}

        # User-item bias: total relevance per (user, item).
        self.user_item_bias = (
            interactions.groupby(["user_id", "item_id"])["relevance"].sum().to_dict()
        )

    def score(self, user_id: str, cuisine: str) -> float:
        return self.profiles.get(user_id, {}).get(str(cuisine).lower(), 0.0)

    def price_affinity(self, user_id: str, item_price_bucket: int) -> float:
        pref = self.price_pref.get(user_id)
        if pref is None:
            return 0.0
        # Higher when closer to preferred bucket.
        diff = abs(pref - item_price_bucket)
        return max(0.0, 1.0 - diff / 2.0)

    def item_bias(self, user_id: str, item_id: int) -> float:
        return float(self.user_item_bias.get((user_id, item_id), 0.0))
