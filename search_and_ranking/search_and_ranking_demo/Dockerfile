FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install uv.
RUN pip install --no-cache-dir uv

# Prefer CPU wheels for torch during resolution (CPU index first, PyPI as fallback).
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_EXTRA_INDEX_URL=https://pypi.org/simple

# Copy project files and code before syncing so editable install can find the package.
COPY pyproject.toml README.md run_demo.py ./ 
COPY search ./search
COPY data ./data

# Install dependencies from project metadata into .venv (lock refresh disabled here).
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --extra semantic
ENV PATH="/app/.venv/bin:${PATH}"J

# Pre-download the sentence-transformers model to avoid runtime fetch.
ENV HF_HOME=/root/.cache/huggingface
RUN .venv/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

CMD ["python", "run_demo.py", "--semantic"]
