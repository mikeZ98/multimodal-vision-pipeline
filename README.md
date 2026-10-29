# multimodal-vision-pipeline

Asynchronous document vision and extraction microservice. Upload a PDF, TIFF, PNG or JPEG;
the service rasterizes it page by page, asks a vision model for a schema-validated extraction
(vendor, dates, line items, taxes, totals, each with a confidence and a page anchor), and then
checks the arithmetic exactly: line amounts, subtotal, tax lines, tax total and grand total.

```text
upload ─▶ sniff & hash ─▶ job queued ─▶ rasterize (chunked) ─▶ vision request ─▶ strict parse
                                                                                     │
          GET /api/v1/jobs/{job_id} ◀── succeeded / failed ◀── reconcile (integer cents)
```

The full design, diagrams and the reasoning behind each limit are in
[docs/architecture.md](docs/architecture.md); operating it is covered in
[docs/runbook.md](docs/runbook.md).

## Core tenets

- **Hermetic verification.** The test suite never opens a network socket (`pytest-socket`),
  never spawns a process and never calls the real vision API: rasterization is faked in memory
  and the Messages API is an `httpx.MockTransport`. Container and CI configuration is checked by
  static inspection.
- **Streaming memory bounds.** Documents are rendered a few pages at a time, decoded bitmaps
  are released before the next chunk, every image is capped in pixels and bytes, and every queue
  and store has a hard size. A 5,000-page PDF costs one page count, not 5,000 bitmaps.
- **Exact cent math.** Money is an integer number of minor units and rates and quantities are
  `Decimal`; there is no `float` anywhere in reconciliation. Each derived amount is rounded once,
  with an explicit rounding mode and tolerance.
- **RFC 9457 errors.** Every error response is `application/problem+json` with a stable
  `type` URN, so clients branch on `type`, never on message text.
- **Trust content, not labels.** File types are detected from magic bytes; a declared type that
  disagrees with the content is rejected.

## Quickstart

### Local development (uv)

Requirements: Python 3.12 (uv installs it), [uv](https://docs.astral.sh/uv/), and poppler for
PDF rendering (`brew install poppler` or `apt-get install poppler-utils`).

```bash
uv sync                                  # locked dependencies, including dev tools
export MVP_ANTHROPIC_API_KEY=sk-ant-...  # required to start the service
uv run uvicorn multimodal_vision_pipeline.api:create_app --factory --port 8000
```

Quality gates (the same ones CI runs):

```bash
ruff check . && ruff format --check .
mypy --strict
bandit -c pyproject.toml -r src
pytest                                   # hermetic; fails under 90% branch coverage
scripts/audit-attribution.sh             # sole-contributor audit of the history
```

Contributor rules (commit identity, hooks, commit messages) are in
[CONTRIBUTING.md](CONTRIBUTING.md).

### Docker Compose

```bash
export MVP_ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build --wait         # returns once /healthz and /readyz pass
curl -s http://127.0.0.1:8000/healthz    # {"status":"ok"}
docker compose down
```

The compose service runs the production image as UID 10001 with a read-only root filesystem,
no Linux capabilities, a 256 MB `/tmp` tmpfs and 2 CPU / 2 GB limits. It publishes on
`127.0.0.1:8000` only; set `PUBLISH_ADDRESS` / `PUBLISH_PORT` to change that, and put a
TLS-terminating proxy in front for remote access.

### Configuration

Every setting is an environment variable with the `MVP_` prefix, read once at startup.

| Variable | Default | Meaning |
|---|---|---|
| `MVP_ANTHROPIC_API_KEY` | *(required)* | API key for the Messages API. The service refuses to start without it. |
| `MVP_ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | API base URL (e.g. a gateway). |
| `MVP_VISION_MODEL` | `claude-sonnet-5-5` | Default model: `claude-opus-5-5`, `claude-sonnet-5-5` or `claude-haiku-5-5`. |
| `MVP_VISION_MAX_TOKENS` | `16000` | Output token cap per extraction. |
| `MVP_VISION_EFFORT` | `high` | `low`, `medium`, `high`, `xhigh` or `max`. |
| `MVP_VISION_REQUEST_TIMEOUT_SECONDS` | `120` | Read timeout per API attempt. |
| `MVP_VISION_MAX_RETRIES` | `2` | Retries after the first attempt for retryable failures. |
| `MVP_VISION_SERVER_SIDE_FALLBACK` | `true` | Let the API re-run refused requests on its fallback model (Opus and Sonnet). |
| `MVP_WORKER_MAX_CONCURRENCY` | `4` | Jobs processed at the same time. |
| `MVP_WORKER_JOB_TIMEOUT_SECONDS` | `900` | Deadline for one job, all stages included. |
| `MVP_WORKER_MAX_QUEUED_JOBS` | `32` | Jobs scheduled at once (running + waiting); each holds its upload in memory. |
| `MVP_WORKER_SHUTDOWN_GRACE_SECONDS` | `30` | How long shutdown waits for running jobs before cancelling them. |
| `MVP_JOB_STORE_MAX_JOBS` | `10000` | Jobs kept for polling; the oldest finished job is evicted first. |
| `MVP_MAX_UPLOAD_BYTES` | `52428800` | Largest accepted upload (50 MiB, also the hard ceiling). |
| `MVP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `MVP_LOG_JSON` | `true` | Install the redacting JSON log handler at startup. |

Server options use uvicorn's own `UVICORN_*` variables (`UVICORN_HOST`, `UVICORN_PORT`,
`UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN`, ...). Keep `UVICORN_WORKERS=1`: jobs live in process
memory, so a second worker process would answer `404` for jobs it never saw.

## Using the API

### Upload a document

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/documents \
  -H 'Idempotency-Key: inv-2026-0042' \
  -F 'file=@invoice.pdf;type=application/pdf' \
  -F 'options={"document_kind":"invoice","rasterization":{"dpi":200,"pages":[{"start":1,"end":3}]}}'
```

```http
HTTP/1.1 202 Accepted
Location: /api/v1/jobs/5728913d-2956-4f85-b8db-e9d4b888060e
X-Correlation-ID: 11192153f0434ad283ad227874425363

{"job_id":"5728913d-2956-4f85-b8db-e9d4b888060e","status":"queued",
 "status_url":"/api/v1/jobs/5728913d-2956-4f85-b8db-e9d4b888060e",
 "created_at":"2026-10-29T18:25:32.836833Z"}
```

- `file` is required; `options` is an optional JSON string (`document_kind`,
  `rasterization`, `extraction` to override the model per request, `reconciliation` policy,
  `idempotency_key`).
- Repeating a request with the same `Idempotency-Key`, file and options returns `200` with the
  same job; the same key with a different file or options is `409 idempotency-conflict`.
- `503 job-capacity` means the queue is full or the service is draining: retry with backoff.

### Poll the job

```bash
curl -sS http://127.0.0.1:8000/api/v1/jobs/5728913d-2956-4f85-b8db-e9d4b888060e
```

While the job runs, the response carries `Retry-After: 1`; every response is
`Cache-Control: no-store`. A finished job looks like this (abridged):

```json
{
  "job_id": "5728913d-2956-4f85-b8db-e9d4b888060e",
  "status": "succeeded",
  "pages": [{"page_number": 1, "width_px": 1654, "height_px": 2338, "dpi": 200, "color_mode": "RGB"}],
  "extraction": {"total": {"value": 6764, "confidence": 0.98, "status": "extracted", "anchor": {"page_number": 1, "bbox": {"x0": 0.71, "y0": 0.82, "x1": 0.9, "y1": 0.85}}}},
  "reconciliation": {"status": "passed", "discrepancies": []}
}
```

`succeeded` means the pipeline ran; check `reconciliation.status` (`passed`, `failed` or
`skipped`) before trusting the numbers. A `failed` job carries an RFC 9457 `error` instead.

### Errors

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/documents -F 'file=@notes.txt;type=text/plain'
```

```json
{"type":"urn:multimodal-vision-pipeline:problem:unsupported-document","title":"Unsupported document",
 "status":415,"detail":"unsupported media type 'text/plain'; use application/pdf, image/tiff, image/png, image/jpeg",
 "instance":"/api/v1/documents","errors":[]}
```

Validation failures list each offending field as an RFC 6901 JSON pointer in `errors`. The full
catalogue of problem types is in [docs/architecture.md](docs/architecture.md#error-model).

### Health and metrics

```bash
curl -sS http://127.0.0.1:8000/healthz   # liveness: {"status":"ok"}
curl -sS http://127.0.0.1:8000/readyz    # 200 ready, 503 while draining or at queue capacity
curl -sS http://127.0.0.1:8000/metrics | grep '^mvp_'
```

Every response echoes `X-Correlation-ID` and `X-Request-ID`; send your own `X-Correlation-ID`
to follow one request through the logs and into the upstream API call.

## Benchmarks and evaluation

`multimodal_vision_pipeline.evals` measures extraction accuracy on synthetic invoices with exact
labels, under increasingly bad scan conditions.

```bash
export MVP_ANTHROPIC_API_KEY=sk-ant-...
uv run python -m multimodal_vision_pipeline.evals --documents 20 --profiles clean,office_scan,fax \
  --model claude-sonnet-5-5 --concurrency 4 --output-dir .cache/evals
```

This calls the real API once per document per profile (60 calls above) and costs money; the
test suite never does. Reports land in `benchmark.json` (every sample) and `benchmark.md`
(one row per profile).

| Profile | Simulated defects |
|---|---|
| `clean` | none |
| `office_scan` | 1.5° skew, light blur, JPEG quality 75 |
| `low_quality_scan` | -4° skew, blur, 0.7 contrast, 1% salt-and-pepper, JPEG 35 |
| `fax` | 7° skew, 0.5 contrast, 3% salt-and-pepper, blur, JPEG 15 |
| `worst_case` | 15° skew, heavy blur, 0.4 contrast, 5% salt-and-pepper, JPEG 5 |

Reported metrics: field-level precision, recall and F1; normalized exact match and Levenshtein
distance on text fields; absolute and percentage cent deltas; anchor IoU and accuracy; a
hallucination rate (values the labels do not support, or read from the wrong place); the share
of extractions whose arithmetic does not reconcile; and p50/p95 latency.

## Repository layout

```text
src/multimodal_vision_pipeline/
  api/            FastAPI app factory, routes, uploads, RFC 9457 handlers, health and metrics
  core/           rasterizer, vision payload builder, reconciler, job state machine, errors
  tools/          vision API client (retries, fallback) and the async pipeline worker
  observability/  correlation context, redacting JSON logs, Prometheus metrics, middleware
  schemas/        strict Pydantic contracts and settings
  evals/          synthetic dataset, scan perturbations, metrics, benchmark runner and CLI
  healthcheck.py  container health probe (the image ships no curl)
docs/             architecture guide and operational runbook
tests/            hermetic test suite, including static checks of docs, container and CI
```

## License

[MIT](LICENSE)
