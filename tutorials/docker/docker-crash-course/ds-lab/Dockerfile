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
