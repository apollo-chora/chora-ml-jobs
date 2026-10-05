#
# Multi-stage build for chora-ml-jobs (Python ML batch jobs).
#
# Build context = this repository. The image runs one batch job per invocation;
# the job is selected at runtime via the JOB environment variable.
#
# Usage:
#   docker build -t chora-ml-jobs .
#   docker run --rm -e JOB=drift_detector -e DATABASE_URL=... chora-ml-jobs
#   docker run --rm -e JOB=eval_runner -e DATABASE_URL=... chora-ml-jobs
#   docker run --rm -e JOB=billing_aggregator chora-ml-jobs

ARG SERVICE_NAME=chora-ml-jobs
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

############################
# Stage 1 — build
############################
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

ARG SERVICE_NAME

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Project metadata first for layer caching.
COPY pyproject.toml uv.lock ./

# Resolve + install dependencies (without the project itself).
RUN uv sync --no-dev --no-install-project

# Full source + golden dataset fixtures, then install the project.
COPY src ./src
COPY fixtures ./fixtures
RUN uv sync --no-dev

############################
# Stage 2 — runtime
############################
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME

LABEL org.opencontainers.image.title="${SERVICE_NAME}" \
      org.opencontainers.image.source="https://github.com/apollo-chora/chora-ml-jobs" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.vendor="Chora Platform" \
      org.opencontainers.image.licenses="UNLICENSED" \
      io.chora.service="${SERVICE_NAME}" \
      io.chora.runtime="python" \
      io.chora.git-sha="${GIT_SHA}" \
      io.chora.build-time="${BUILD_TIME}"

# Non-root user (uid 65532 matches distroless-nonroot semantics).
RUN groupadd --system --gid 65532 nonroot \
    && useradd --system --uid 65532 --gid 65532 --no-create-home nonroot

WORKDIR /app

# Bring the venv with installed dependencies + sources + fixtures.
COPY --from=build /app /app
RUN chown -R nonroot:nonroot /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JOB="drift_detector" \
    FIXTURES_DIR="/app/fixtures" \
    SERVICE_NAME=${SERVICE_NAME} \
    GIT_SHA=${GIT_SHA} \
    BUILD_TIME=${BUILD_TIME}

# Batch job — no ports are listened on.
USER nonroot:nonroot
ENTRYPOINT ["python", "-m", "chora_ml_jobs.entrypoint"]
