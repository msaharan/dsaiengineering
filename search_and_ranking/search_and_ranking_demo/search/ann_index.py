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

from typing import List, Sequence, Tuple

import numpy as np


class ANNIndex:
    """Small ANN wrapper: prefers faiss, falls back to sklearn NearestNeighbors."""

    def __init__(self, vectors: np.ndarray):
        self.uses_faiss = False
        self.index = None
        self.vectors = vectors.astype("float32")
        try:
            import faiss  # type: ignore

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            faiss.normalize_L2(self.vectors)
            self.index.add(self.vectors)
            self.uses_faiss = True
        except ImportError:
            try:
                from sklearn.neighbors import NearestNeighbors
            except ImportError as exc:
                raise ImportError("No ANN backend available. Install faiss or scikit-learn.") from exc
            self.index = NearestNeighbors(metric="cosine")
            self.index.fit(self.vectors)

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.uses_faiss:
            import faiss  # type: ignore

            q = query_vectors.astype("float32")
            faiss.normalize_L2(q)
            scores, indices = self.index.search(q, top_k)
            return scores, indices
        else:
            distances, indices = self.index.kneighbors(query_vectors, n_neighbors=top_k)
            scores = 1.0 - distances  # cosine distance -> similarity
            return scores, indices
