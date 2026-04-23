# Multi-stage build for a slim production image.
# Heavy ML deps (lightgbm, prophet) are kept; spaCy model + transformers
# are pulled at build time only if you uncomment them in requirements.txt.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps for prophet (cmdstanpy / pystan), lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt
RUN python -m spacy download en_core_web_sm || true   # best-effort

# ---- final stage ---------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 botuser

COPY --from=builder /root/.local /root/.local

WORKDIR /app
COPY --chown=botuser:botuser . .

RUN mkdir -p data models_store && chown -R botuser:botuser data models_store
USER botuser

EXPOSE 8000

# Default entrypoint runs the API. Override CMD to run the scheduler:
#   docker run weather-bot python -m src.scheduler
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
