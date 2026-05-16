# syntax=docker/dockerfile:1.7
# ---- builder: resolve and install dependencies with uv -----------------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Third-party dependencies first for layer caching.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# Then the project itself.
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# ---- runtime: minimal image, non-root user ----------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp: OpenMP runtime required by XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib \
    FRAUD_ML_MODEL_DIR=/app/models \
    FRAUD_ML_DATA_DIR=/app/data \
    FRAUD_ML_REPORTS_DIR=/app/reports \
    FRAUD_ML_MLFLOW_ENABLED=false \
    FRAUD_ML_LOG_FORMAT=json \
    FRAUD_ML_LOG_LEVEL=INFO

RUN groupadd --system fraudml && useradd --system --gid fraudml --home /app fraudml

WORKDIR /app
COPY --from=builder --chown=fraudml:fraudml /app /app
COPY --chown=fraudml:fraudml docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /app/models /app/data /app/reports \
    && chown -R fraudml:fraudml /app/models /app/data /app/reports

USER fraudml
VOLUME ["/app/models", "/app/data", "/app/reports"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=5 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status == 200 else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["fraud-ml", "serve", "--host", "0.0.0.0", "--port", "8000"]
