# syntax=docker/dockerfile:1.7
#
# multimodal-vision-pipeline: production image.
#
#   builder  locked, bytecode-compiled virtualenv built with uv (no dev group)
#   runtime  slim Debian + poppler, the venv copied in, non-root, read-only-rootfs ready
#
# Build:  docker build -t multimodal-vision-pipeline .
# Run:    see docker-compose.yml (read-only rootfs, dropped capabilities, tmpfs /tmp).

# Base images are pinned by tag *and* digest; bump both together. Builder and runtime
# must share PYTHON_IMAGE: the venv's interpreter symlinks into /usr/local/bin.
ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------
# Builder: resolve exactly uv.lock into /opt/venv
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies first: this layer is reused until pyproject.toml or uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# The project itself, as a regular (non-editable) install: /build is not shipped.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="multimodal-vision-pipeline" \
      org.opencontainers.image.description="Async document vision & extraction microservice" \
      org.opencontainers.image.licenses="MIT"

# poppler-utils: pdfinfo/pdftoppm for PDF rasterization (pdf2image shells out to them).
# libopenjp2-7, libtiff6: JPEG 2000 and TIFF codecs used by poppler.
# tini: PID 1 that forwards SIGTERM to uvicorn and reaps orphaned poppler processes.
# Then strip setuid/setgid bits so nothing in the image can regain privileges.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        poppler-utils \
        libopenjp2-7 \
        libtiff6 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin appuser \
    && find / -xdev -perm /6000 -type f -exec chmod a-s {} +

COPY --from=builder /opt/venv /opt/venv
COPY docker/uvicorn-logging.json /app/uvicorn-logging.json

# Uvicorn reads UVICORN_<OPTION> for every CLI option, so these are overridable at run time.
# Keep UVICORN_WORKERS=1: the job store lives in process memory, and a GET for a job can
# land on a worker that never saw it. Scale with replicas behind sticky routing instead;
# worker_max_concurrency already overlaps jobs inside one process.
# UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN bounds connection draining on SIGTERM; the lifespan then
# drains jobs for MVP_WORKER_SHUTDOWN_GRACE_SECONDS. The stop timeout must cover both.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TMPDIR=/tmp \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=1 \
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=10 \
    UVICORN_LOG_LEVEL=info

# Code and venv stay root-owned and read-only to the service user.
WORKDIR /app
USER 10001:10001

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "multimodal_vision_pipeline.healthcheck", "/healthz"]

# No `tini -g`: SIGTERM goes to uvicorn only, so in-flight pdftoppm calls finish while
# the worker drains instead of being killed with the process group.
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "multimodal_vision_pipeline.api:create_app", "--factory", \
     "--log-config", "/app/uvicorn-logging.json", "--no-access-log", "--no-server-header"]
