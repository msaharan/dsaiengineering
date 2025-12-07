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

from typing import List, Tuple

import pandas as pd

from .query_understanding import UnderstoodQuery
from .ranking import FeatureRow


def apply_business_rules(
    scored_rows: List[Tuple[float, FeatureRow]],
    catalog: pd.DataFrame,
    understood: UnderstoodQuery,
    vegan_boost: float = 0.2,
    diversity_penalty: float = 0.1,
) -> List[Tuple[float, FeatureRow]]:
    """Simple reranker that boosts vegan-friendly results for vegan queries and enforces cuisine diversity."""
    seen_cuisines = set()
    adjusted = []
    wants_vegan = "vegan" in understood.cuisines or "vegan" in understood.dietary_tags

    for score, row in scored_rows:
        item = catalog[catalog["item_id"] == row.item_id].iloc[0]
        cuisine = str(item["cuisine"]).lower()
        new_score = score

        if wants_vegan and bool(item.get("is_vegan_friendly", False)):
            new_score += vegan_boost

        if cuisine in seen_cuisines:
            new_score -= diversity_penalty
        else:
            seen_cuisines.add(cuisine)

        adjusted.append((new_score, row))

    adjusted.sort(key=lambda x: x[0], reverse=True)
    return adjusted
