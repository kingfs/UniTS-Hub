# =========================
# builder stage
# =========================
FROM python:3.12-slim-bookworm AS builder

# ---- proxy args (optional, for local build) ----
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv 
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

RUN uv pip install -r pyproject.toml

# ---- bake model weights ----
## 下面脚本将模型下载至/app/models
RUN python scripts/download_models.py

# =========================
# runtime stage
# =========================
FROM python:3.12-slim-bookworm

LABEL maintainer="kingfs"
LABEL description="Unified serving for time-series foundation models (CPU-only)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ---- copy python runtime ----
COPY --from=builder /bin/uv* /bin/
COPY --from=builder /app/models /app/models

# ---- copy application ----
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
# COPY --from=builder /app/README.md /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]