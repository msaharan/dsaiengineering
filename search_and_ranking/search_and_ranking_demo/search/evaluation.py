from __future__ import annotations

from typing import Dict, List

import numpy as np

from .ranking import FeatureRow


def dcg_at_k(rels: List[float], k: int) -> float:
    rels = np.asarray(rels)[:k]
    gains = 2 ** rels - 1
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(rels: List[float], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def mrr_at_k(rels: List[float], k: int) -> float:
    for idx, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def evaluate_predictions(rows: List[FeatureRow], preds: np.ndarray, k: int = 3) -> Dict[str, float]:
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
