FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py batch_predict.py .

ENV MODEL_PATH=/models/model.joblib

RUN useradd -m appuser && mkdir -p /models && chown -R appuser /models /app
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
