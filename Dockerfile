# Backend image. The API needs a real filesystem: SQLite, the FTS5 index and
# the embedding matrices are files, and ingestion writes to them for minutes.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/data/fastembed

WORKDIR /srv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    "fastapi>=0.117" "uvicorn[standard]>=0.30"

COPY app ./app
COPY scripts ./scripts

# mounted disk; the DB and vectors live here so they survive a redeploy
VOLUME ["/data"]
ENV DATA_DIR=/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]
