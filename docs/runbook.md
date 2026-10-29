# Operational runbook

How to deploy, size, observe and troubleshoot `multimodal-vision-pipeline` in production. The
design behind these recommendations is in [architecture.md](architecture.md).

- [Deployment](#deployment)
- [Health probing](#health-probing)
- [Resource sizing](#resource-sizing)
- [Restarts, deploys and shutdown](#restarts-deploys-and-shutdown)
- [Log aggregation](#log-aggregation)
- [Metrics and alerting](#metrics-and-alerting)
- [Troubleshooting](#troubleshooting)
- [Routine maintenance](#routine-maintenance)

## Deployment

**Checklist**

- [ ] Build from the repository `Dockerfile` (multi-stage, locked dependencies, non-root). The
      final stage is `runtime`.
- [ ] Inject `MVP_ANTHROPIC_API_KEY` from a secret store as an environment variable. Never bake
      it into an image or commit it; the service will not start without it.
- [ ] Run **one uvicorn process per container** (`UVICORN_WORKERS=1`, the default). Jobs live in
      process memory; a second worker process would answer `404` for jobs it did not accept.
- [ ] Keep the hardening from `docker-compose.yml` on any orchestrator: read-only root
      filesystem, a writable tmpfs at `/tmp`, all capabilities dropped, `no-new-privileges`,
      UID/GID 10001.
- [ ] Terminate TLS in front of the service. The compose file publishes on `127.0.0.1` only.
- [ ] Give the container a stop timeout of at least
      `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN + MVP_WORKER_SHUTDOWN_GRACE_SECONDS` plus a margin
      (45 s with the defaults).
- [ ] Scrape `GET /metrics` and load the alert rules below.

**Scaling out.** There is no shared job store, so a job can only be polled on the replica that
accepted it. Either run a single replica (one replica handles `MVP_WORKER_MAX_CONCURRENCY` jobs
at a time; vision calls dominate job time), or use sticky sessions at the load balancer so a
client's upload and its polls reach the same replica. A replica restart loses its jobs; clients
should resubmit with the same `Idempotency-Key`.

**Kubernetes example**

```yaml
spec:
  terminationGracePeriodSeconds: 45
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
  containers:
    - name: api
      image: multimodal-vision-pipeline:1.0.0
      ports:
        - containerPort: 8000
      envFrom:
        - secretRef:
            name: mvp-anthropic
      resources:
        requests: {cpu: 500m, memory: 512Mi}
        limits: {cpu: "2", memory: 2Gi}
      securityContext:
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
      volumeMounts:
        - {name: tmp, mountPath: /tmp}
      livenessProbe:
        httpGet: {path: /healthz, port: 8000}
        periodSeconds: 15
        failureThreshold: 3
      readinessProbe:
        httpGet: {path: /readyz, port: 8000}
        periodSeconds: 5
        failureThreshold: 1
  volumes:
    - name: tmp
      emptyDir: {medium: Memory, sizeLimit: 256Mi}
```

## Health probing

| Endpoint | Use as | 200 | 503 |
|---|---|---|---|
| `GET /healthz` | liveness | the process and event loop are responsive: `{"status":"ok"}` | never; no answer means restart |
| `GET /readyz` | readiness | accepting work | draining, or `MVP_WORKER_MAX_QUEUED_JOBS` jobs already scheduled |

`/readyz` returns the worker's view either way, which is useful when debugging saturation:

```json
{"status":"ready","worker":{"accepting":true,"active_jobs":0,"max_concurrency":4,"max_queued_jobs":32},"stored_jobs":1,"max_jobs":10000}
```

- Do **not** use `/readyz` for liveness: a full queue is a reason to stop routing traffic, not to
  restart the container (which would drop every queued job).
- The image has no curl. Container-level probes run
  `python -m multimodal_vision_pipeline.healthcheck /healthz /readyz` (exit 0 only if every path
  answers 200). The image's own `HEALTHCHECK` checks `/healthz` only; the compose healthcheck
  checks both, so `docker compose ps` shows `unhealthy` while the queue is full.

## Resource sizing

Memory has three parts. With the defaults (`MVP_WORKER_MAX_CONCURRENCY=4`,
`MVP_WORKER_MAX_QUEUED_JOBS=32`, `MVP_MAX_UPLOAD_BYTES=50 MiB`, 200 DPI RGB):

| Component | Formula | Default worst case |
|---|---|---|
| Uploads held by scheduled jobs | queued jobs x upload limit | 32 x 50 MiB = 1.6 GiB |
| Bitmaps being rendered | concurrency x 4 pages x page bitmap | 4 x 4 x 11.6 MB = 186 MB |
| Encoded pages and requests | concurrency x (selected pages' PNGs + request of at most 30 MiB) | a few hundred MB |
| Spool files in `/tmp` (tmpfs, counts as memory) | concurrency x upload limit | 4 x 50 MiB = 200 MiB |
| Interpreter, libraries, job store | fixed | about 150 MB |

The compose defaults (2 GiB limit, 512 MiB reservation) fit typical traffic, but not the
theoretical worst case of 32 queued 50 MiB uploads. For hostile or bulk traffic, lower
`MVP_WORKER_MAX_QUEUED_JOBS` or `MVP_MAX_UPLOAD_BYTES` until the worst case fits the limit.
Running out means an OOM kill, which loses every in-memory job.

**CPU.** Rendering and image encoding are CPU-bound and run in threads; two cores serve four
concurrent jobs comfortably. More concurrency mainly helps when vision calls dominate.

**Upstream quota.** Each job is one Messages API request (plus retries). Peak request rate is
about `MVP_WORKER_MAX_CONCURRENCY / median job time`; check it against your rate limits before
raising concurrency.

## Restarts, deploys and shutdown

On SIGTERM (forwarded by `tini`):

1. uvicorn stops accepting connections and drains open ones for
   `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN` (10 s).
2. The worker stops admitting jobs (`/readyz` turns 503, uploads get `job-capacity`) and lets
   running jobs finish for `MVP_WORKER_SHUTDOWN_GRACE_SECONDS` (30 s).
3. Remaining jobs are cancelled and end as `job-cancelled`; the process exits.

Every job, finished or not, is lost when the process exits. For zero-loss deploys, take the
replica out of rotation, wait until `mvp_active_jobs` is 0 and clients have collected their
results, then stop it.

## Log aggregation

Logs are JSON lines on stderr; ship them with any collector (Fluent Bit, Vector, Promtail) and
parse them as JSON. The compose file rotates them with the `json-file` driver (10 MB x 5).

```json
{"timestamp":"2026-10-29T18:25:33.112Z","level":"info","logger":"multimodal_vision_pipeline.tools.pipeline_worker","message":"stage extraction completed","correlation_id":"11192153f0434ad283ad227874425363","request_id":"11192153f0434ad283ad227874425363","job_id":"5728913d-2956-4f85-b8db-e9d4b888060e","stage":"extraction","latency_ms":8412.07}
```

| Field | Use |
|---|---|
| `correlation_id` | Follow one upload end to end, including the upstream call (sent as `X-Correlation-ID`). |
| `job_id` | Everything the worker did for one job. |
| `stage` | `rasterization`, `extraction` or `reconciliation`. |
| `logger` | `multimodal_vision_pipeline.access` holds one access line per request with `status_code` and `latency_ms`. |
| `error_type`, `traceback` | Present on logged exceptions, already redacted. |

- Index `correlation_id`, `job_id`, `level` and `logger`; do not index `message`.
- Logs never contain document bytes, page images, extracted values or credentials, and the
  redactor is a second line of defence (keys, bearer tokens, base64 payloads, IBANs, emails,
  card numbers). Treat logs as internal data anyway: file names and upstream error messages can
  carry customer information.
- `MVP_LOG_LEVEL=DEBUG` is safe to enable temporarily; it adds detail, not payloads.
- Set `MVP_LOG_JSON=false` only when the host process configures logging itself.

## Metrics and alerting

Scrape `GET /metrics` every 15 to 30 seconds. Metric names, types and labels are listed in
[architecture.md](architecture.md#observability).

**Recommended alert rules** (Prometheus; tune thresholds to your traffic):

```yaml
groups:
  - name: multimodal-vision-pipeline
    rules:
      - alert: MvpJobFailureRateHigh
        expr: |
          sum(rate(mvp_jobs_total{status="failed"}[15m]))
            / sum(rate(mvp_jobs_total[15m])) > 0.10
        for: 15m
        labels:
          severity: page
        annotations:
          summary: More than 10% of jobs are failing
          runbook: Check the error slug breakdown in the logs (error_slug) and the troubleshooting table.

      - alert: MvpUpstreamErrorsHigh
        expr: |
          sum(rate(mvp_upstream_requests_total{status_code!~"2.."}[10m]))
            / sum(rate(mvp_upstream_requests_total[10m])) > 0.20
        for: 10m
        labels:
          severity: page
        annotations:
          summary: More than 20% of vision API attempts fail
          runbook: Split by status_code; 429 and 529 mean quota or overload, timeout and connection_error mean network.

      - alert: MvpUpstreamLatencyHigh
        expr: |
          histogram_quantile(0.95,
            sum by (le, model) (rate(mvp_upstream_latency_seconds_bucket[15m]))) > 90
        for: 15m
        labels:
          severity: ticket
        annotations:
          summary: p95 vision API latency above 90 s
          runbook: Expect job-timeout failures next; consider a lower effort or a smaller model.

      - alert: MvpUploadsRejectedForCapacity
        expr: |
          sum(rate(mvp_http_requests_total{route="/api/v1/documents", status_code="503"}[5m])) > 0
        for: 10m
        labels:
          severity: ticket
        annotations:
          summary: Uploads are being turned away (job-capacity)
          runbook: The queue is saturated; add replicas with sticky routing or raise concurrency within upstream quota.

      - alert: MvpHttpServerErrors
        expr: |
          sum(rate(mvp_http_requests_total{status_code=~"5..", status_code!="503"}[5m]))
            / sum(rate(mvp_http_requests_total[5m])) > 0.05
        for: 10m
        labels:
          severity: page
        annotations:
          summary: More than 5% of HTTP responses are server errors
          runbook: Look for logger=multimodal_vision_pipeline.api.errors entries with tracebacks.

      - alert: MvpStageSlow
        expr: |
          histogram_quantile(0.95,
            sum by (le, stage) (rate(mvp_job_duration_seconds_bucket[30m]))) > 300
        for: 30m
        labels:
          severity: ticket
        annotations:
          summary: p95 duration of a pipeline stage above 5 minutes
          runbook: Rasterization means large or high-DPI documents; extraction means upstream latency or retries.

      - alert: MvpReconciliationDiscrepanciesRising
        expr: |
          sum(rate(mvp_reconciliation_discrepancies_total[6h]))
            / sum(rate(mvp_jobs_total{status="succeeded"}[6h])) > 0.25
        for: 1h
        labels:
          severity: ticket
        annotations:
          summary: More than 0.25 arithmetic discrepancies per successful job
          runbook: Split by rule_id; run the evals benchmark to see whether extraction quality regressed.

      - alert: MvpWorkerStuck
        expr: |
          max(mvp_active_jobs) > 0
            and sum(rate(mvp_jobs_total[30m])) == 0
        for: 30m
        labels:
          severity: page
        annotations:
          summary: Jobs are active but none has finished in 30 minutes
          runbook: Jobs should hit MVP_WORKER_JOB_TIMEOUT_SECONDS long before this; capture logs and restart.
```

Dashboards worth having: jobs per minute by `status`; stage p50/p95 from
`mvp_job_duration_seconds`; upstream attempts by `status_code`; `mvp_active_jobs` against
`MVP_WORKER_MAX_CONCURRENCY`; discrepancies by `rule_id`; HTTP rate and latency by `route`;
`route="unmatched"` traffic (scanners and misconfigured clients).

## Troubleshooting

Failed jobs carry the cause as an RFC 9457 `type`; the worker logs the same value as
`error_slug`.

| Symptom (slug) | Likely cause | Action |
|---|---|---|
| `job-capacity` on upload | Queue full, store full of running jobs, or draining | Retry with backoff; check `/readyz`; scale out or raise `MVP_WORKER_MAX_QUEUED_JOBS` within memory. |
| `upstream-rate-limited`, `upstream-overloaded` | Quota or API capacity | Lower concurrency, spread load, or raise the quota; retries already used `retry-after`. |
| `upstream-rejected` | Bad or revoked key, billing, invalid request | Check the key and account; an invalid-request reply after a deploy points at a payload regression. |
| `upstream-timeout`, `job-timeout` | Slow model on large requests | Lower `effort`, select fewer pages, or raise `MVP_VISION_REQUEST_TIMEOUT_SECONDS` / `MVP_WORKER_JOB_TIMEOUT_SECONDS`. |
| `upstream-unavailable` | Network, DNS, proxy, egress policy | Test egress to `MVP_ANTHROPIC_BASE_URL` from the pod. |
| `extraction-invalid` | The reply did not match the schema | Usually transient; if persistent for one model, compare with another model via `extraction.model`. |
| `vision-refusal` | The model declined the content | Expected for some content; server-side fallback is on by default for Opus and Sonnet. |
| `vision-payload-too-large` | More than 100 pages selected, or extreme pages | Split the document with `rasterization.pages` across several jobs. |
| `document-too-large` | Upload over the limit, or a page over 30,000 px per side | Lower `rasterization.dpi`; compress or split the file. |
| `rasterization-failed` | Corrupt, empty or unreadable document | Open the file locally; re-export the PDF. |
| `unsupported-document` | Not PDF/TIFF/PNG/JPEG, or the declared type is wrong | Fix the client's content type, or omit it and let the service detect it. |
| `page-selection` | Ranges beyond the last page | The problem's `detail` states how many pages the document has. |
| `reconciliation-overflow` | Absurd extracted amounts | Inspect the extraction; usually a misread number. |
| Job succeeded, `reconciliation.status: failed` | The document's arithmetic does not add up as extracted | Inspect `discrepancies`: a single-cent delta suggests the rounding mode or tolerance; larger deltas are misreads or genuinely wrong documents. |
| `404` on a job that existed | Evicted from the store, a restart, or polling another replica | Raise `MVP_JOB_STORE_MAX_JOBS`, collect results sooner, check sticky routing. |

## Routine maintenance

- **Model quality.** Before changing `MVP_VISION_MODEL` or `MVP_VISION_EFFORT`, run the benchmark
  on both settings and compare the `benchmark.md` tables (this calls the paid API):
  `python -m multimodal_vision_pipeline.evals --documents 20 --model claude-haiku-5-5`.
- **Key rotation.** Update the secret and restart replicas one at a time; the key is read once
  at startup.
- **Base images and actions.** Base images (Dockerfile) and GitHub Actions are pinned by digest
  or commit SHA and are not updated by a bot (bot commits would fail the sole-contributor audit).
  Bump the tag and digest together, by hand, and let CI rebuild and smoke-test the image.
- **Dependencies.** `uv lock --upgrade`, then run every quality gate locally before pushing;
  CI installs strictly from `uv.lock`.
