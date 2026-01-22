# =========================
# builder stage
# =========================
FROM debian:bookworm-slim AS builder

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
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/tmp/hf_cache

WORKDIR /build

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# ---- install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN uv pip install --system hatchling

COPY . .
RUN uv pip install --system --no-build-isolation .

# ---- bake model weights ----
## 下面脚本将模型下载至/app/models
RUN python3 scripts/download_models.py

# =========================
# runtime stage
# =========================
FROM debian:bookworm-slim

LABEL maintainer="kingfs"
LABEL description="Unified serving for time-series foundation models (CPU-only)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# ---- copy python runtime ----
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/models /app/models

# ---- copy application ----
COPY --from=builder /build/app /app/app
COPY --from=builder /build/README.md /app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]