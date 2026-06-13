# ── World Cup 2026 Sentiment Analysis — Dockerfile ──────────────────────────
# Multi-stage build: dependencies first, then runtime.

FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Download spaCy models
RUN python -m spacy download es_core_news_sm && \
    python -m spacy download en_core_web_sm

COPY . .

# Default: run the full pipeline
CMD ["python", "-m", "src.pipeline"]
