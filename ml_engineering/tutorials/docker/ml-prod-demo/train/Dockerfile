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
