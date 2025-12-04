FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install uv.
RUN pip install --no-cache-dir uv

# Install dependencies from project metadata and lockfile (cached layer).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --system --extra semantic

# Pre-download the sentence-transformers model to avoid runtime fetch.
ENV HF_HOME=/root/.cache/huggingface
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

# Copy application code.
COPY search ./search
COPY data ./data
COPY run_demo.py ./run_demo.py

# Install the package (no deps; already installed).
RUN uv pip install --system --no-deps .

CMD ["python", "run_demo.py", "--semantic"]
