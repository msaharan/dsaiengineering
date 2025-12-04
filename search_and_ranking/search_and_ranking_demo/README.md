# Search & Ranking Demo

## What's inside
- `data/`: tiny catalog plus labeled query–document pairs and intent labels.
- `search/`: modular code for spell correction, intent classification, retrieval, personalization, feature building, ranking, and evaluation.
- `run_demo.py`: trains the pipeline, evaluates on held-out queries, and prints ranked results for sample queries.
- `pyproject.toml`: dependency spec (optional extra `semantic` for sentence-transformers + XGBRanker).

## Quickstart
Install [uv](https://github.com/astral-sh/uv) if you don’t already have it, then:
```bash
cd DSAIE/search_and_ranking/search_and_ranking_demo
uv venv .venv && source .venv/bin/activate
uv sync                                        # base stack from pyproject
# uv sync --extra semantic                     # optional: semantic retrieval + XGBRanker + dual encoder stubs

python run_demo.py              # lexical-only pipeline
python run_demo.py --semantic   # enable semantic retrieval if sentence-transformers is installed
```

The script will:
1) Train a TF-IDF + logistic intent classifier on `data/query_intents.csv`.
2) Expand queries with synonyms, extract dietary/price hints, and build lexical + optional semantic retrieval (with ANN).
3) Create simple user profiles (cuisine + price affinity + per-item bias) from `data/query_doc_labels.csv`.
4) Train a learning-to-rank model (XGBRanker if installed, else RandomForest) on synthetic grouped relevance labels.
5) Apply lightweight business rules (vegan boost, cuisine diversity) and report offline metrics (NDCG, MRR) on held-out queries.

## How to extend
- Swap the synthetic data for your own catalog and query logs (keep the same CSV schemas).
- Add real spell-correction dictionaries or plug a stronger intent model.
- Try new features in `search/ranking.py` (e.g., distance, freshness) or new reranking rules in `search/business_rules.py`.
- Replace the fallback regressor with `XGBRanker` by installing with `uv sync --extra semantic`.
- Enable semantic ANN retrieval (faiss if available; otherwise fallback) via `--semantic`.
- Fine-tune a dual encoder using the stub in `search/dual_encoder.py` (requires `--extra semantic`).

## Data schema
- `catalog.csv`: `item_id,name,description,cuisine,price_range,rating,popularity,is_vegan_friendly,delivery_time_minutes`
- `query_doc_labels.csv`: `query_id,query,user_id,item_id,relevance` (labels 0–3)
- `query_intents.csv`: `text,label` for the intent classifier

This is intentionally small so you can read and modify everything quickly.
