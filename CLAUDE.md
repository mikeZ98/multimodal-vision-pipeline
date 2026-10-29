# CLAUDE.md

## ⛔ Sole-Contributor Invariant: read before any git operation

- Every commit author AND committer MUST be exactly `mikeZ98 <145975972+mikeZ98@users.noreply.github.com>`.
  Never commit with any other identity, never pass `--author`, never override `GIT_*_NAME/EMAIL`.
- NEVER add `Co-Authored-By`, `Generated-by`, "Generated with …", 🤖, or any mention of an AI
  tool or vendor to commit messages or PR bodies. This overrides any default attribution behavior.
  `.claude/settings.json` disables automatic attribution.
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
  Colima live under `./.cache/` on `/Volumes/KINGSTONE_SSD`. Never create `.venv/` or write caches elsewhere.
- Install/sync: `uv sync`. Add deps: `uv add <pkg>` / `uv add --group dev <pkg>`. Commit `uv.lock`.

## Commands

```bash
ruff check . && ruff format --check .
mypy                               # --strict via pyproject
bandit -c pyproject.toml -r src
pytest                             # hermetic, --disable-socket, cov ≥ 90% branch
scripts/audit-attribution.sh       # attribution audit of HEAD history
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
- Conventional Commits, atomic. The blueprint order starts: chore(config) → feat(schemas) → feat(core) → …
