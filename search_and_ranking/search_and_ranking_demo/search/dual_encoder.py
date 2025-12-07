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

from typing import List

import pandas as pd


def build_mnr_training_data(labeled_pairs: pd.DataFrame, catalog: pd.DataFrame):
    """Build training examples for MultipleNegativesRankingLoss using (query, item_text) pairs."""
    try:
        from sentence_transformers import InputExample  # type: ignore
    except ImportError as exc:
        raise ImportError("sentence-transformers required for dual-encoder training.") from exc

    catalog_by_id = catalog.set_index("item_id")
    examples = []
    for _, row in labeled_pairs.iterrows():
        if row["relevance"] <= 0:
            continue
        item_text = catalog_by_id.loc[row["item_id"]]["text"]
        examples.append(InputExample(texts=[row["query"], item_text]))
    return examples


def train_dual_encoder(
    labeled_pairs: pd.DataFrame,
    catalog: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    epochs: int = 1,
    batch_size: int = 16,
):
    """
    Minimal in-batch negatives dual-encoder training.
    Returns a fine-tuned SentenceTransformer.
    """
    try:
        from sentence_transformers import SentenceTransformer, losses  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as exc:
        raise ImportError("sentence-transformers required for dual-encoder training.") from exc

    examples = build_mnr_training_data(labeled_pairs, catalog)
    if not examples:
        raise ValueError("No positive relevance labels found for dual-encoder training.")

    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=False,
    )
    return model
