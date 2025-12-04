from __future__ import annotations

from typing import List

import pandas as pd


def build_mnr_training_data(labeled_pairs: pd.DataFrame) -> List:
    """Build training examples for MultipleNegativesRankingLoss using (query, positive) pairs."""
    try:
        from sentence_transformers import InputExample  # type: ignore
    except ImportError as exc:
        raise ImportError("sentence-transformers required for dual-encoder training.") from exc

    examples = []
    for _, row in labeled_pairs.iterrows():
        if row["relevance"] <= 0:
            continue
        examples.append(InputExample(texts=[row["query"], str(row["item_id"])]))
    return examples


def train_dual_encoder(
    labeled_pairs: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2", epochs: int = 1, batch_size: int = 16
):
    """
    Optional stub for in-batch negatives dual-encoder training to align with the article.
    Returns a fine-tuned SentenceTransformer.
    """
    try:
        from sentence_transformers import SentenceTransformer, losses  # type: ignore
    except ImportError as exc:
        raise ImportError("sentence-transformers required for dual-encoder training.") from exc

    examples = build_mnr_training_data(labeled_pairs)
    if not examples:
        raise ValueError("No positive relevance labels found for dual-encoder training.")

    model = SentenceTransformer(model_name)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(examples, train_loss)],
        epochs=epochs,
        train_batch_size=batch_size,
        show_progress_bar=False,
    )
    return model
