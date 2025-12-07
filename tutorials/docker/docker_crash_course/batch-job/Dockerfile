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
