FROM python:3.12-slim-bookworm AS builder

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG MODEL_TYPE=chronos
ARG KRONOS_RUNTIME_REPO=https://github.com/shiyu-coder/Kronos.git
ARG KRONOS_RUNTIME_REF=master

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/tmp/hf_cache \
    MODEL_TYPE=${MODEL_TYPE} \
    KRONOS_RUNTIME_REPO=${KRONOS_RUNTIME_REPO} \
    KRONOS_RUNTIME_REF=${KRONOS_RUNTIME_REF} \
    KRONOS_RUNTIME_PATH=/opt/kronos-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

RUN uv pip install -r pyproject.toml
RUN mkdir -p /opt/kronos-runtime
RUN if [ "${MODEL_TYPE}" = "kronos" ]; then sh scripts/install_kronos_runtime.sh; fi
RUN python scripts/download_models.py --model ${MODEL_TYPE}

FROM python:3.12-slim-bookworm

ARG MODEL_TYPE=chronos
ARG KRONOS_RUNTIME_REF=master

LABEL maintainer="kingfs"
LABEL description="UniTS-Hub single-model serving image"
LABEL org.opencontainers.image.title="UniTS-Hub"
LABEL org.opencontainers.image.description="Single-model time-series foundation model serving image"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_TYPE=${MODEL_TYPE} \
    KRONOS_RUNTIME_PATH=/opt/kronos-runtime \
    KRONOS_RUNTIME_REF=${KRONOS_RUNTIME_REF} \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/models /app/models
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /opt/kronos-runtime /opt/kronos-runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
