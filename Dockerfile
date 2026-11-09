# ── Stage 1: Build/dependency resolution ─────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Copy dependency files first (layer cache)
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies into /app/.venv
RUN uv sync --no-dev --frozen

# ── Stage 2: Model download (cached layer) ───────────────────────────────────
FROM builder AS models

ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Pre-download models at build time so container starts fast
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embed model...'); \
SentenceTransformer('${EMBED_MODEL}'); \
print('Downloading rerank model...'); \
CrossEncoder('${RERANK_MODEL}'); \
print('Models ready.')"

# ── Stage 3: Final image ──────────────────────────────────────────────────────
FROM python:3.12-slim AS final

WORKDIR /app

# Non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Copy venv and models from builder stages
COPY --from=models /app/.venv /app/.venv
COPY --from=models /root/.cache/huggingface /home/raguser/.cache/huggingface

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/golden/ ./data/golden/

# Set ownership
RUN chown -R raguser:raguser /app /home/raguser

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV TRANSFORMERS_CACHE=/home/raguser/.cache/huggingface

USER raguser

# Default: run the API
EXPOSE 8080
CMD ["uvicorn", "hybrid_rag.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
