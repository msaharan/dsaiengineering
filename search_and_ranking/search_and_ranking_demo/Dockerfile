FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Keep dependency resolution in sync with the lockfile timestamp cutoff.
ENV UV_EXCLUDE_NEWER=2024-12-31T23:00:00Z

# Install uv.
RUN pip install --no-cache-dir uv

# Copy project files and code before syncing so editable install can find the package.
COPY pyproject.toml uv.lock README.md run_demo.py ./ 
COPY search ./search
COPY data ./data

# Install dependencies from project metadata/lockfile into .venv.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra semantic
ENV PATH="/app/.venv/bin:${PATH}"

# Pre-download the sentence-transformers model to avoid runtime fetch.
ENV HF_HOME=/root/.cache/huggingface
RUN .venv/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

CMD ["python", "run_demo.py", "--semantic"]
