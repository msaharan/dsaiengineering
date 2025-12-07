# Docker Crash Course

## 0. Prereqs & quick sanity check

You’ll need:

- Docker Desktop (or Docker Engine) installed.
- Basic Python + terminal familiarity.

Quick check:

```bash
docker --version
docker run hello-world
```

If the second command prints a friendly message and exits, you’re good.

Platform note (Apple Silicon → x86_64): add `--platform linux/amd64` to `docker build` / `docker run` / `docker compose` if you need to target an x86 prod environment.

------

## 1. Mental model & 80/20 commands

**How to picture it:**

- **Image** = a frozen environment you can run later.
- **Container** = a running process that came from an image.
- **Dockerfile** = the recipe to make an image.
- **Volume** = a place on disk that survives when containers die.
- **Port mapping** = how you reach web apps inside the container.

**Commands you’ll use a lot:**

```bash
# Build image from Dockerfile
docker build -t my-image-name .

# Run a container from an image
docker run --rm my-image-name

# Map a folder and a port
docker run --rm -v "$(pwd)/data:/app/data" -p 8000:8000 my-image-name

# List containers
docker ps        # running only
docker ps -a     # all (including stopped)

# View logs
docker logs <container_id_or_name>

# Get a shell inside a running container
docker exec -it <container_id_or_name> bash
```

> On Windows PowerShell, replace `$(pwd)` with `${PWD}`.

------

## 2. Lab 1 – Containerize a batch data cleaning job

**Scenario:**
 Nightly/weekly ETL that reads raw CSV, cleans it, writes a clean CSV. Nothing fancy, just something you can hand to a teammate or cron without “works on my machine.”

### 2.1 Project layout

Create this structure:

```text
batch-job/
  Dockerfile
  requirements.txt
  clean_data.py
  data/           # (can be empty; will hold raw.csv / clean.csv)
```

### 2.2 `clean_data.py`

This script:

- Generates a synthetic `raw.csv` if it doesn’t exist.
- Cleans missing values and extreme incomes.
- Adds an `age_bucket` column.
- Writes `clean.csv`.

```python
# clean_data.py
import os
from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "raw.csv"
CLEAN_PATH = DATA_DIR / "clean.csv"


def generate_raw_if_missing():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        return
    rng = np.random.default_rng(42)
    n = 100
    ages = rng.integers(18, 70, size=n)
    incomes = rng.normal(50000, 15000, size=n)
    churn = (ages > 50).astype(int)
    df = pd.DataFrame({"age": ages, "income": incomes, "churn": churn})
    # introduce some missing values
    df.loc[rng.choice(n, size=5, replace=False), "income"] = np.nan
    df.to_csv(RAW_PATH, index=False)
    print(f"Generated synthetic raw dataset at {RAW_PATH}")


def clean():
    generate_raw_if_missing()
    print(f"Reading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Simple cleaning: drop rows with missing income, clip extreme values
    before = len(df)
    df = df.dropna(subset=["income"])
    after = len(df)
    print(f"Dropped {before - after} rows with missing income")

    df["income"] = df["income"].clip(lower=0, upper=200000)
    df["age_bucket"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 80],
                              labels=["18-30", "31-45", "46-60", "61+"])

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")


if __name__ == "__main__":
    clean()
```

### 2.3 `requirements.txt`

```text
pandas
numpy
```

### 2.4 `Dockerfile`

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Make Python output unbuffered (nicer logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clean_data.py ./clean_data.py

# Data will live in /app/data (we'll mount a volume here)
RUN mkdir -p data

# Add a non-root user so mounted files aren’t owned by root on your host
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "clean_data.py"]
```

### 2.5 `.dockerignore` (same folder)

```text
__pycache__
*.pyc
.git
.venv
data/*
```

### 2.6 Build & run

From inside `batch-job/`:

```bash
# Build image
docker build -t batch-cleaner .

# Run with a volume so data persists on your host
# macOS / Linux:
docker run --rm -v "$(pwd)/data:/app/data" batch-cleaner

# Windows PowerShell:
docker run --rm -v "${PWD}/data:/app/data" batch-cleaner
```

Check the output files on your host:

```bash
ls data
# raw.csv  clean.csv

head data/clean.csv
```

That’s a basic data pipeline in a container.

------

## 3. Lab 2 – Reproducible Jupyter / experiment environment

**Scenario:**
 A shared DS stack (pandas / sklearn / plotting) you can start as Jupyter Lab without version fights on every laptop.

### 3.1 Project layout

```text
ds-lab/
  Dockerfile
  requirements.txt
  notebooks/
```

(Leave `notebooks/` empty for now; that’s where you’ll save work.)

### 3.2 `requirements.txt`

```text
jupyterlab
pandas
numpy
matplotlib
scikit-learn
seaborn
```

### 3.3 `Dockerfile`

```dockerfile
# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# Add a non-root user so notebooks saved to the mounted volume keep your host UID/GID
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /workspace
USER appuser

# Start Jupyter Lab when the container runs
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
```

### 3.4 `.dockerignore` (same folder)

```text
__pycache__
*.pyc
.git
.venv
notebooks/.ipynb_checkpoints
```

### 3.5 Build & run

From `ds-lab/`:

```bash
# Build
docker build -t ds-lab .

# macOS / Linux:
docker run --rm -p 8888:8888 -v "$(pwd)/notebooks:/workspace" ds-lab

# Windows PowerShell:
docker run --rm -p 8888:8888 -v "${PWD}/notebooks:/workspace" ds-lab
```

Watch the logs; you’ll see something like:

```text
    Or copy and paste one of these URLs:
        http://127.0.0.1:8888/lab?token=...
```

Open that URL in your browser.

Anything you create in Jupyter Lab under `/workspace` is actually being written to your host `notebooks/` directory thanks to the volume.

> This pattern (one container = one dev environment) is super common for DS/ML: you can pin versions, share the Dockerfile, and be confident you’re all using the same stack.

------

## 4. Lab 3 – Train & serve an ML model with `docker compose`

**Scenario:**

- One container trains an ML model and writes it to a shared volume.
- Another container loads that model and serves predictions via a FastAPI endpoint.
- Use `docker compose` so the two play nicely together.

### 4.1 Project layout

```text
ml-prod-demo/
  docker-compose.yml
  train/
    Dockerfile
    requirements.txt
    train.py
  serve/
    Dockerfile
    requirements.txt
    app.py
```

Add a `.dockerignore` inside both `train/` and `serve/`:

```text
__pycache__
*.pyc
.git
.venv
```

------

### 4.2 Training service

#### `train/train.py`

```python
# train/train.py
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_DIR = Path("/models")
MODEL_PATH = MODEL_DIR / "iris_logreg.joblib"


def main():
    print("Loading iris dataset...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training LogisticRegression...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target_names": iris.target_names}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

#### `train/requirements.txt`

```text
scikit-learn
joblib
pandas
```

#### `train/Dockerfile`

```dockerfile
# train/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

RUN mkdir -p /models
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app /models
USER appuser

CMD ["python", "train.py"]
```

------

### 4.3 Serving service (FastAPI)

#### `serve/app.py`

```python
# serve/app.py
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path("/models/iris_logreg.joblib")

app = FastAPI(title="Iris Classifier API")

model = None
target_names = None


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
def load_model():
    global model, target_names
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Did you run the training container?"
        )
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    target_names = bundle["target_names"]
    print(f"Loaded model from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    pred_idx = int(model.predict(X)[0])
    return {
        "predicted_class_index": pred_idx,
        "predicted_class_name": str(target_names[pred_idx]),
        "input": features.dict(),
    }
```

#### `serve/requirements.txt`

```text
fastapi
uvicorn[standard]
scikit-learn
joblib
```

#### `serve/Dockerfile`

```dockerfile
# serve/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

RUN mkdir -p /models
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app /models
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

------

### 4.4 `docker-compose.yml`

At the root (`ml-prod-demo/`):

```yaml
# docker-compose.yml
version: "3.9"

services:
  train:
    build: ./train
    volumes:
      - model_store:/models

  api:
    build: ./serve
    ports:
      - "8000:8000"
    volumes:
      - model_store:/models
    depends_on:
      - train
    entrypoint: >
      sh -c "while [ ! -f /models/iris_logreg.joblib ]; do echo 'Waiting for model...'; sleep 2; done; exec uvicorn app:app --host 0.0.0.0 --port 8000"

volumes:
  model_store:
```

The `model_store` volume is shared: trainer writes `/models/iris_logreg.joblib`, API reads it. The API container now waits until the model file exists before starting uvicorn, so you don’t get a “model missing” race.

------

### 4.5 Build, train, serve

From inside `ml-prod-demo/`:

```bash
# Build both images
docker compose build      # or: docker-compose build

# Run training job (one-off)
docker compose run --rm train

# Start the API
docker compose up api
```

If you skip the one-off training run, `docker compose up api` will still start `train`; the API stays in a short wait loop until `/models/iris_logreg.joblib` appears.

You should see logs like:

- From `train`: accuracy + “Saved model to /models/iris_logreg.joblib”
- From `api`: “Loaded model from /models/iris_logreg.joblib”

Now test the API.

**Example request (macOS / Linux / WSL):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

You should get a JSON response like:

```json
{
  "predicted_class_index": 0,
  "predicted_class_name": "setosa",
  "input": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

That’s the simple train/serve loop I reach for when I need a quick demo.

------

## 5. Practical patterns & best practices (DS/ML flavored)

### 5.1 Use a `.dockerignore`

I forget this occasionally and regret it when builds crawl. Add one near the Dockerfile to avoid sending junk into the build context:

```text
__pycache__
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3

.env
.venv
venv
.git
.gitignore

data/raw/*
notebooks/.ipynb_checkpoints
```

This speeds up builds and avoids accidentally baking secrets into images.

------

### 5.2 Layer caching & fast builds

Typical Python pattern:

```dockerfile
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
```

- When you only change application code, Docker reuses the `pip install` layer → faster builds.
- If you change `requirements.txt`, Docker reruns pip (which is what you want).

------

### 5.3 Configuration via environment variables

Instead of hard-coding config in code:

```bash
docker run --rm \
  -e MODEL_PATH=/models/other_model.joblib \
  -v "$(pwd)/models:/models" \
  iris-api
```

In Python:

```python
import os
model_path = os.getenv("MODEL_PATH", "/models/iris_logreg.joblib")
```

This is very common in production (e.g., selecting model versions, turning on debug logs, etc.).

------

### 5.4 GPUs (high-level pointer)

For GPU training/inference, the pattern is basically the same, just with a CUDA base image:

- Use an image with CUDA (e.g. PyTorch or TensorFlow official Docker images).
- Run with `--gpus` (requires the NVIDIA Container Toolkit installed on host):

```bash
docker run --gpus all my-gpu-image python train.py
```

Inside `Dockerfile`, you’d base off something like `pytorch/pytorch:...-cuda...` instead of `python:3.11-slim`.

------

### 5.5 Housekeeping (free disk fast)

List what’s hanging around, then prune unused stuff. I run these when Docker starts eating disk:

```bash
docker ps -a
docker images
docker volume ls

docker system prune -f        # remove stopped containers + unused networks
docker system prune -a -f     # also remove unused images (more aggressive)
docker volume rm <name>       # delete stray volumes once you’re sure they’re unused
```

Stop containers you still need before pruning so you don’t delete their resources.
