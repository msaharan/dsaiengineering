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

from typing import Dict, List

import numpy as np

from .ranking import FeatureRow


def dcg_at_k(rels: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    rels = np.asarray(rels)[:k]
    gains = 2 ** rels - 1
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(rels: List[float], k: int) -> float:
    """Compute Normalized DCG@k, returning 0.0 when ideal DCG is zero."""
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def mrr_at_k(rels: List[float], k: int) -> float:
    """Compute Mean Reciprocal Rank@k for a single ranked list of labels."""
    for idx, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def evaluate_predictions(rows: List[FeatureRow], preds: np.ndarray, k: int = 3) -> Dict[str, float]:
    """Aggregate per-query NDCG@k and MRR@k over predicted scores and return the means."""
    per_query: Dict[str, List[tuple[float, float]]] = {}
    for row, pred in zip(rows, preds):
        per_query.setdefault(row.query_id, []).append((pred, row.label))

    ndcgs = []
    mrrs = []
    for _, pairs in per_query.items():
        pairs.sort(key=lambda x: x[0], reverse=True)
        labels_sorted = [p[1] for p in pairs]
        ndcgs.append(ndcg_at_k(labels_sorted, k=k))
        mrrs.append(mrr_at_k(labels_sorted, k=k))

    return {
        "mean_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }
