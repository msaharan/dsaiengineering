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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .ann_index import ANNIndex

@dataclass
class ScoredItem:
    item_id: int
    score: float
    source: str


class LexicalRetriever:
    """TF-IDF based lexical retriever."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.doc_matrix = self.vectorizer.fit_transform(catalog["text"])
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(catalog["item_id"])}

    def query(self, text: str, top_k: int = 5) -> List[ScoredItem]:
        q_vec = self.vectorizer.transform([text])
        scores = linear_kernel(q_vec, self.doc_matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]
        results = [
            ScoredItem(
                item_id=int(self.catalog.iloc[idx]["item_id"]),
                score=float(scores[idx]),
                source="lexical",
            )
            for idx in top_indices
        ]
        return results

    def score_pair(self, text: str, item_id: int) -> float:
        idx = self.id_to_idx[item_id]
        q_vec = self.vectorizer.transform([text])
        score = linear_kernel(q_vec, self.doc_matrix[idx])
        return float(score[0, 0])


class SemanticRetriever:
    """SentenceTransformer-based semantic retriever with optional ANN index."""

    def __init__(self, catalog: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2", use_ann: bool = False):
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not available. Install optional dependencies to enable semantic retrieval."
            ) from exc

        self.util = util
        self.model = SentenceTransformer(model_name)
        self.catalog = catalog
        self.doc_embeddings = self.model.encode(
            catalog["text"].tolist(), convert_to_tensor=True, show_progress_bar=False
        )
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(catalog["item_id"])}
        self.ann = None
        if use_ann:
            self.ann = ANNIndex(self.doc_embeddings.cpu().numpy())

    def query(self, text: str, top_k: int = 5) -> List[ScoredItem]:
        q_emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        k = min(top_k, len(self.catalog))
        if self.ann:
            scores, indices = self.ann.search(q_emb.cpu().numpy()[None, :], k)
            flat_scores = scores[0]
            flat_indices = indices[0]
        else:
            scores_tensor = self.util.cos_sim(q_emb, self.doc_embeddings)[0]
            top_k_scores, top_k_indices = scores_tensor.topk(k=k)
            flat_scores = top_k_scores
            flat_indices = top_k_indices

        results = []
        for score, idx in zip(flat_scores, flat_indices):
            results.append(
                ScoredItem(
                    item_id=int(self.catalog.iloc[int(idx)]["item_id"]),
                    score=float(score),
                    source="semantic",
                )
            )
        return results

    def score_pair(self, text: str, item_id: int) -> float:
        idx = self.id_to_idx[item_id]
        q_emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        item_emb = self.doc_embeddings[idx]
        return float(self.util.cos_sim(q_emb, item_emb)[0])


class DualEncoderRetriever:
    """Dual-encoder semantic retriever with optional ANN index."""

    def __init__(self, catalog: pd.DataFrame, model, use_ann: bool = True):
        self.model = model
        self.catalog = catalog
        self.doc_embeddings = self.model.encode(
            catalog["text"].tolist(), convert_to_tensor=False, show_progress_bar=False
        )
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(catalog["item_id"])}
        self.ann = ANNIndex(np.array(self.doc_embeddings)) if use_ann else None

    def query(self, text: str, top_k: int = 5) -> List[ScoredItem]:
        q_emb = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        k = min(top_k, len(self.catalog))
        if self.ann:
            scores, indices = self.ann.search(np.array([q_emb]), k)
            flat_scores = scores[0]
            flat_indices = indices[0]
        else:
            doc_mat = np.stack(self.doc_embeddings)
            scores = doc_mat @ np.array(q_emb)
            flat_indices = scores.argsort()[::-1][:k]
            flat_scores = scores[flat_indices]

        results = []
        for score, idx in zip(flat_scores, flat_indices):
            results.append(
                ScoredItem(
                    item_id=int(self.catalog.iloc[int(idx)]["item_id"]),
                    score=float(score),
                    source="dual_encoder",
                )
            )
        return results

    def score_pair(self, text: str, item_id: int) -> float:
        q_emb = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        idx = self.id_to_idx[item_id]
        item_emb = self.doc_embeddings[idx]
        return float(np.dot(q_emb, item_emb))


class HybridRetriever:
    """Combine lexical and semantic scores with a simple weighted sum."""

    def __init__(
        self,
        lexical: LexicalRetriever,
        semantic: Optional[SemanticRetriever] = None,
        semantic_weight: float = 0.5,
    ):
        self.lexical = lexical
        self.semantic = semantic
        self.semantic_weight = semantic_weight

    def retrieve(self, text: str, top_k: int = 5) -> List[ScoredItem]:
        lexical_results = self.lexical.query(text, top_k=top_k * 2)
        if self.semantic:
            semantic_results = self.semantic.query(text, top_k=top_k * 2)
        else:
            semantic_results = []

        scores: Dict[int, Dict[str, float]] = {}
        for item in lexical_results:
            scores.setdefault(item.item_id, {})["lexical"] = item.score
        for item in semantic_results:
            scores.setdefault(item.item_id, {})["semantic"] = item.score

        # Normalize scores to [0,1] to make the weighted sum more stable.
        def normalize(values):
            if not values:
                return {}
            arr = np.array(list(values.values()))
            denom = arr.max() - arr.min() + 1e-6
            return {k: (v - arr.min()) / denom for k, v in values.items()}

        normalized_scores = normalize(
            {k: v.get("lexical", 0.0) for k, v in scores.items()}
        )
        normalized_semantic = normalize(
            {k: v.get("semantic", 0.0) for k, v in scores.items()}
        )

        combined = []
        for item_id in scores.keys():
            lexical_score = normalized_scores.get(item_id, 0.0)
            semantic_score = normalized_semantic.get(item_id, 0.0)
            total = lexical_score + self.semantic_weight * semantic_score
            combined.append(ScoredItem(item_id=item_id, score=float(total), source="hybrid"))

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]

    def pair_scores(self, text: str, item_id: int) -> tuple[float, float]:
        lexical_score = self.lexical.score_pair(text, item_id)
        semantic_score = 0.0
        if self.semantic:
            semantic_score = self.semantic.score_pair(text, item_id)
        return lexical_score, semantic_score
