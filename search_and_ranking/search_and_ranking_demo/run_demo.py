from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from search.business_rules import apply_business_rules
from search.data_loader import (
    get_data_dir,
    load_catalog,
    load_query_doc_labels,
    load_query_intents,
)
from search.evaluation import evaluate_predictions
from search.personalization import UserProfiles
from search.query_understanding import (
    IntentClassifier,
    build_cuisine_lexicon,
    understand_query,
)
from search.ranking import Ranker, build_feature_rows, build_matrices
from search.retrieval import HybridRetriever, LexicalRetriever, SemanticRetriever
from search.spell import SpellCorrector


def train_pipeline(enable_semantic: bool = False):
    base_dir = Path(__file__).resolve().parent
    data_dir = get_data_dir(base_dir)

    catalog = load_catalog(data_dir)
    labeled_pairs = load_query_doc_labels(data_dir)
    intent_data = load_query_intents(data_dir)

    intent_classifier = IntentClassifier()
    intent_classifier.fit(intent_data["text"], intent_data["label"])

    vocab = list(catalog["name"]) + list(catalog["cuisine"])
    # Keep max_edit_distance small to avoid aggressive corrections like "lunch" -> "curry".
    spell_corrector = SpellCorrector(vocab, max_edit_distance=1)
    cuisines = build_cuisine_lexicon(catalog)

    lexical = LexicalRetriever(catalog)
    semantic = None
    if enable_semantic:
        semantic = SemanticRetriever(catalog, use_ann=True)
    hybrid = HybridRetriever(lexical, semantic)

    user_profiles = UserProfiles(labeled_pairs, catalog)

    train_qids, test_qids = train_test_split(
        labeled_pairs["query_id"].unique(), test_size=0.3, random_state=42
    )
    train_df = labeled_pairs[labeled_pairs["query_id"].isin(train_qids)]
    test_df = labeled_pairs[labeled_pairs["query_id"].isin(test_qids)]

    train_rows = build_feature_rows(
        train_df, catalog, hybrid, user_profiles, intent_classifier, cuisines
    )
    test_rows = build_feature_rows(
        test_df, catalog, hybrid, user_profiles, intent_classifier, cuisines
    )

    X_train, y_train, group_train, _ = build_matrices(train_rows)
    X_test, y_test, _, test_rows_meta = build_matrices(test_rows)

    ranker = Ranker()
    ranker.fit(X_train, y_train, group=group_train)
    preds = ranker.predict(X_test)
    metrics = evaluate_predictions(test_rows_meta, preds, k=3)

    print("\nOffline evaluation (held-out queries)")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")

    demo_queries = ["vegan sushi", "fast lunch", "seafood dinner"]
    for q in demo_queries:
        run_demo_query(
            q,
            catalog,
            hybrid,
            spell_corrector,
            intent_classifier,
            cuisines,
            ranker,
            user_profiles,
        )


def run_demo_query(
    query: str,
    catalog: pd.DataFrame,
    hybrid: HybridRetriever,
    spell_corrector: SpellCorrector,
    intent_classifier: IntentClassifier,
    cuisines,
    ranker: Ranker,
    user_profiles: UserProfiles,
):
    understood = understand_query(query, spell_corrector, intent_classifier, cuisines)
    print("\n---")
    print(f"Query: {query}")
    print(f"Normalized/Corrected: '{understood.corrected}' | Intent: {understood.intent}")

    retrieval_text = understood.corrected + " " + " ".join(understood.expansions)
    candidates = hybrid.retrieve(retrieval_text.strip(), top_k=8)
    feature_rows = []
    for scored in candidates:
        user_id = "u_demo"
        row = {
            "query_id": "demo",
            "query": understood.corrected,
            "user_id": user_id,
            "item_id": scored.item_id,
            "relevance": 0,  # placeholder for feature builder
        }
        feature_rows.append(row)

    # Build features for demo ranking
    temp_df = pd.DataFrame(feature_rows)
    demo_feature_rows = build_feature_rows(
        temp_df, catalog, hybrid, user_profiles, intent_classifier, cuisines
    )
    X_demo, _, _, demo_rows = build_matrices(demo_feature_rows)
    demo_preds = ranker.predict(X_demo)

    scored_rows = sorted(
        [(pred, row) for pred, row in zip(demo_preds, demo_rows)],
        key=lambda x: x[0],
        reverse=True,
    )
    reranked = apply_business_rules(scored_rows, catalog, understood)

    for rank, (pred, row) in enumerate(reranked[:5], start=1):
        item = catalog[catalog["item_id"] == row.item_id].iloc[0]
        print(
            textwrap.dedent(
                f"""
                #{rank}: {item['name']} (score={pred:.3f})
                  cuisine={item['cuisine']} | price={item['price_range']} | rating={item['rating']}
                  description={item['description']}
                """
            ).strip()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hands-on search & ranking demo.")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic retrieval (requires sentence-transformers).",
    )
    args = parser.parse_args()
    train_pipeline(enable_semantic=args.semantic)
