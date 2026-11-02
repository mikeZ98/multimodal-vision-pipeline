# CLAUDE.md

## Git commit policy

Maintain clean, conventional git commit messages adhering to repository guidelines with unified
author metadata. Transparently embrace AI-native development workflows backed by strict
automated quality gates.

- Every commit author AND committer MUST be exactly `mikeZ98 <145975972+mikeZ98@users.noreply.github.com>`.
  Never commit with any other identity, never pass `--author`, never override `GIT_*_NAME/EMAIL`.
- Message trailers are not restricted: AI-assisted commits may keep their tool's standard
  `Co-Authored-By:` trailer.
- NEVER use `git commit --no-verify` or `git push --no-verify`. If a hook fails, fix the cause.
- Policy source of truth: `.githooks/_policy.sh`. Audit: `scripts/audit-attribution.sh`.
- Full rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Project

`multimodal-vision-pipeline`: async FastAPI microservice that rasterizes PDF/TIFF uploads to images
(`pdf2image`/Pillow), extracts structured data with a vision LLM under strict Pydantic schema
enforcement, and validates the result with mathematical post-processing checks (totals, sums,
cross-field consistency). Python 3.12, uv, `src/` layout.

## Environment

- Always work with direnv loaded (`direnv allow`). Caches, the venv (`.cache/venv`), temp files and
  Colima live under `./.cache/`. Never create `.venv/` or write caches elsewhere.
- Install/sync: `uv sync`. Add deps: `uv add <pkg>` / `uv add --group dev <pkg>`. Commit `uv.lock`.

## Commands

```bash
ruff check . && ruff format --check .
mypy                               # --strict via pyproject
bandit -c pyproject.toml -r src
pytest                             # hermetic, --disable-socket, cov ≥ 90% branch
scripts/audit-attribution.sh       # identity audit of HEAD history
```

## Conventions

- Pydantic v2 models: `model_config = ConfigDict(extra="forbid", frozen=True, strict=True)`
  (inherit `schemas._base.PipelineModel`).
- Money is integer minor units (`Cents`); rates and quantities are `Decimal`. Never use `float` for
  amounts. Round derived values once via `ReconciliationPolicy.to_cents`.
- Tests must be hermetic and in-memory: no outbound network (pytest-socket enforced), no
  subprocesses (never invoke poppler/`pdftoppm` from tests; inject a fake rasterizer), no GUI
  (never call `Image.show()`). Mock the LLM HTTP API with `httpx.MockTransport`.
  Use `core.rasterizer.FakeRasterizer`; exercise the PDF path by monkeypatching
  `pdf2image.pdfinfo_from_bytes`/`convert_from_bytes` (see `tests/test_core.py`). Unix sockets are
  allowed only for the asyncio self-pipe; network sockets stay blocked.
- Job state changes go through `core.job_manager.JobManager`; never mutate a `JobStatusResponse`.
- `tools.vision_client.VisionClient` calls the Messages API over raw `httpx` (`anthropic-version`
  `2026-10-21`). In tests pass `transport=httpx.MockTransport(...)`, a recording `sleep` and a
  seeded `rng`; never let a test wait on real backoff.
- API tests: `create_app(settings, rasterizer=..., extractor=...)`, drive the lifespan with
  `app.router.lifespan_context(app)` and send requests through `httpx.ASGITransport`. Every error
  response is `application/problem+json`; raise a `PipelineError` subclass, never return ad-hoc JSON.
- Logging: use module loggers and put structured data in `extra=`; job/stage/correlation ids come
  from `observability.context.bind`, never from string formatting. Never log document bytes, page
  images, request bodies, extracted values or credentials (the redactor is a safety net, not a
  licence). Metrics live in a per-app `PipelineMetrics` registry; label values must be bounded
  (route templates, enum values), never raw paths or ids.
- Evals (`evals/`): synthetic invoices are drawn with Pillow's bundled font, so labels must be
  printable ASCII; every perturbation keeps size/mode and maps anchors via `box_transform`;
  undefined ratios are `None`, never 0 or 1. `python -m multimodal_vision_pipeline.evals` calls the
  real API (costs money): never from tests, which use an in-process `BenchmarkExtractor` stub or
  `VisionClient` with `httpx.MockTransport`.
- Containers: `Dockerfile` (uv builder → slim runtime, UID/GID 10001, tini, read-only-rootfs
  ready), `docker-compose.yml`, allowlist `.dockerignore`. `tests/test_container.py` checks them
  statically; never run Docker from tests. Every `MVP_*` setting must appear in the compose
  `environment` with the code default; new build inputs must be re-included in `.dockerignore`.
  Keep `UVICORN_WORKERS=1` (the job store is per process); health probes go through
  `python -m multimodal_vision_pipeline.healthcheck` (no curl in the image).
- CI (`.github/workflows/ci.yml`) runs the Commands above plus a Docker build, BuildKit checks
  and a compose smoke test; `tests/test_ci.py` checks it statically. Pin every action to a full
  commit SHA with a `# vX.Y.Z` comment, keep `permissions: contents: read`, use no secrets, and
  pass `${{ }}` values to scripts via `env:`. A new local gate must also be added to CI.
  CI audits with `audit-attribution.sh --history-only`, so GitHub-made merge/squash commits and
  bot PRs (e.g. Dependabot) fail it: integrate locally (rebase/fast-forward) and push.
- Docs (`README.md`, `docs/architecture.md`, `docs/runbook.md`) are verified against the code by
  `tests/test_docs.py`: settings and defaults, routes behind every `curl`, the state diagram vs
  `TRANSITIONS`, error slugs and statuses, metric names, quoted limits, CLI flags. Changing any of
  those means updating the docs in the same commit.
- Conventional Commits, atomic. The blueprint order starts: chore(config) → feat(schemas) → feat(core) → …
