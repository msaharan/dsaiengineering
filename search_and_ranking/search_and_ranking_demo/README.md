# Search & Ranking Demo

## What's inside
- `data/`: tiny catalog plus labeled query–document pairs and intent labels.
- `search/`: modular code for spell correction, intent classification, retrieval, personalization, feature building, ranking, and evaluation.
- `run_demo.py`: trains the pipeline, evaluates on held-out queries, and prints ranked results for sample queries.
- `pyproject.toml`: dependency spec (optional extra `semantic` for sentence-transformers + XGBRanker).

## Quickstart (Docker-only)
This project is intended to run in Docker with the semantic stack baked in:
```bash
cd DSAIE/search_and_ranking/search_and_ranking_demo
docker build -t search-ranking-demo .
docker run --rm -it search-ranking-demo        # runs semantic pipeline by default
# Override command if you want lexical-only:
# docker run --rm -it search-ranking-demo python run_demo.py
# Train/use dual-encoder + ANN:
# docker run --rm -it search-ranking-demo python run_demo.py --semantic --dual
```
Docker builds install with `uv sync --locked --extra semantic` (consuming `uv.lock`); torch may pull CUDA wheels. No local `uv sync` needed unless you want to run outside Docker. ANN indexing is optional; the demo uses transformer dot-product retrieval by default. The dual-encoder path (`--semantic --dual`) trains a small dual encoder and enables ANN (faiss required).

If you do run `uv sync` locally, match the lockfile’s timestamp cutoff by exporting:

```bash
export UV_EXCLUDE_NEWER=2024-12-31T23:00:00Z
uv sync --locked --extra semantic
```

Without that env var, `uv` will consider the lock stale and refuse to install. Alternatively, drop the cutoff from `uv.lock` and remove the Dockerfile `UV_EXCLUDE_NEWER` line, then regenerate the lock with `uv lock --python 3.11 --upgrade`.

The script will:
1) Train a TF-IDF + logistic intent classifier on `data/query_intents.csv`.
2) Expand queries with synonyms, extract dietary/price hints, and build lexical + optional semantic retrieval (with ANN). Ontology enrichment derives structured dietary/category/price hints from catalog text/cuisine; it's still a lightweight heuristic, not a full ontology service.
3) Create simple user profiles (cuisine + price affinity + per-item bias) from `data/query_doc_labels.csv`.
4) Train a learning-to-rank model (XGBRanker if installed, else RandomForest) on synthetic grouped relevance labels.
5) Apply lightweight business rules (vegan boost, cuisine diversity) and report offline metrics (NDCG, MRR) on held-out queries. Demo scores printed are min–max normalized for display only; ranking uses raw model scores.

## How to extend
- Swap the synthetic data for your own catalog and query logs (keep the same CSV schemas).
- Add real spell-correction dictionaries or plug a stronger intent model.
- Try new features in `search/ranking.py` (e.g., distance, freshness) or new reranking rules in `search/business_rules.py`.
- Fine-tune a dual encoder using the stub in `search/dual_encoder.py` (requires transformer deps).

## Data schema
- `catalog.csv`: `item_id,name,description,cuisine,price_range,rating,popularity,is_vegan_friendly,delivery_time_minutes`
- `query_doc_labels.csv`: `query_id,query,user_id,item_id,relevance` (labels 0–3)
- `query_intents.csv`: `text,label` for the intent classifier

This is intentionally small so you can read and modify everything quickly.
