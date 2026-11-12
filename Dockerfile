# syntax=docker/dockerfile:1

FROM python:3.10-slim AS builder

ARG EMBEDDING_MODEL=thenlper/gte-small
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .
# Install CPU-only PyTorch first to avoid ~2.5 GB CUDA wheel bloat on CPU VMs
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
    && pip install -r requirements-prod.txt

# Pre-cache embedding and cross-encoder weights at build time for fast offline startup
RUN mkdir -p /app/.cache/huggingface \
    && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"


FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/.cache /app/.cache
COPY --chown=appuser:appuser . .

RUN mkdir -p /app/documents /app/metadata \
    && chown -R appuser:appuser /app \
    && chmod -R u+rwX,g+rwX /app/documents /app/metadata /app/.cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
