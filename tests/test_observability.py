"""Phase 6 verification: metrics exposition, correlation propagation, JSON logs, redaction.

Hermetic: logs go to an in-memory stream, metrics live in per-test registries,
upstream calls hit ``httpx.MockTransport`` and requests go through ``ASGITransport``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr

from multimodal_vision_pipeline.api import create_app
from multimodal_vision_pipeline.core import (
    FakeRasterizer,
    JobManager,
    RasterizedPage,
    VisionPayloadBuilder,
)
from multimodal_vision_pipeline.core.vision_builder import VisionRequest
from multimodal_vision_pipeline.observability import (
    ContextFilter,
    CorrelationMiddleware,
    JsonFormatter,
    PipelineMetrics,
    bind,
    configure_logging,
    current_context,
    redact_text,
    sanitize,
)
from multimodal_vision_pipeline.observability.context import accept_id
from multimodal_vision_pipeline.observability.logging import rfc3339
from multimodal_vision_pipeline.observability.redaction import MAX_STRING_CHARS, REDACTED
from multimodal_vision_pipeline.schemas import (
    ColorMode,
    DocumentDescriptor,
    ExtractionModelConfig,
    IngestionOptions,
    JobStatus,
    PageMetadata,
    VisionSettings,
)
from multimodal_vision_pipeline.tools import PipelineWorker, VisionClient, VisionResponse
from tests.factories import descriptor_data, messages_api_body, model_reply, parse

API_KEY = "sk-ant-api03-Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0Lz"
PDF = b"%PDF-1.7\n1 0 obj<<>>endobj\n%%EOF\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample(exposition: str, name: str, **labels: str) -> float:
    """Value of one sample in a Prometheus text exposition (0.0 when absent)."""
    for family in text_string_to_metric_families(exposition):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return float(s.value)
    return 0.0


class Upstream:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses) or [
            httpx.Response(200, json=messages_api_body(model_reply()))
        ]
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses[min(len(self.requests), len(self.responses)) - 1]


async def no_sleep(delay: float) -> None:
    return None


def observed_app(
    upstream: Upstream | None = None, *, extractor: Any = None, **overrides: Any
) -> tuple[FastAPI, PipelineMetrics]:
    metrics = PipelineMetrics()
    if extractor is None:
        extractor = VisionClient(
            api_key=SecretStr(API_KEY),
            transport=httpx.MockTransport(upstream or Upstream()),
            metrics=metrics,
            sleep=no_sleep,
        )
    settings = VisionSettings(anthropic_api_key=SecretStr(API_KEY), log_json=False, **overrides)
    app = create_app(
        settings,
        rasterizer=FakeRasterizer(page_count=2, page_size_inches=(1.0, 1.5)),
        extractor=extractor,
        metrics=metrics,
    )
    return app, metrics


@asynccontextmanager
async def running(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def submit_and_wait(
    client: httpx.AsyncClient, *, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", PDF, "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    for _ in range(500):
        body: dict[str, Any] = (await client.get(response.headers["location"])).json()
        if JobStatus(body["status"]).is_terminal:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


LogReader = Callable[[], list[dict[str, Any]]]


@pytest.fixture
def logs() -> Iterator[LogReader]:
    root = logging.getLogger()
    previous_level = root.level
    buffer = io.StringIO()
    handler = configure_logging("DEBUG", stream=buffer)
    try:
        yield lambda: [json.loads(line) for line in buffer.getvalue().splitlines()]
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


# ===========================================================================
# Metrics exposition
# ===========================================================================
async def test_metrics_endpoint_after_a_successful_job() -> None:
    app, _ = observed_app()
    async with running(app) as client:
        await submit_and_wait(client)
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    text = response.text
    assert sample(text, "mvp_jobs_total", status="succeeded", mime_type="application/pdf") == 1
    assert sample(text, "mvp_rasterized_pages_total") == 2
    for stage in ("rasterization", "extraction", "reconciliation"):
        assert sample(text, "mvp_job_duration_seconds_count", stage=stage) == 1
        assert sample(text, "mvp_job_duration_seconds_sum", stage=stage) >= 0
    model = "claude-sonnet-5-5"
    assert sample(text, "mvp_upstream_requests_total", model=model, status_code="200") == 1
    assert sample(text, "mvp_upstream_latency_seconds_count", model=model) == 1
    assert sample(text, "mvp_active_jobs") == 0
    assert sample(text, "mvp_reconciliation_discrepancies_total") == 0
    assert (
        sample(
            text,
            "mvp_http_requests_total",
            method="POST",
            route="/api/v1/documents",
            status_code="202",
        )
        == 1
    )
    assert sample(text, "mvp_http_requests_total", route="/api/v1/jobs/{job_id}") >= 1


async def test_metrics_available_before_startup() -> None:
    app, metrics = observed_app()
    metrics.rasterized_pages.inc(3)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        text = (await client.get("/metrics")).text
    assert sample(text, "mvp_rasterized_pages_total") == 3


async def test_retries_failures_and_discrepancies_are_counted() -> None:
    overloaded = httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})
    wrong_total = model_reply().replace('"value": 6764', '"value": 9999')
    upstream = Upstream(overloaded, httpx.Response(200, json=messages_api_body(wrong_total)))
    app, metrics = observed_app(upstream)
    async with running(app) as client:
        final = await submit_and_wait(client)
    text = metrics.render()[0].decode()
    assert final["reconciliation"]["status"] == "failed"
    model = "claude-sonnet-5-5"
    assert sample(text, "mvp_upstream_requests_total", model=model, status_code="529") == 1
    assert sample(text, "mvp_upstream_requests_total", model=model, status_code="200") == 1
    assert sample(text, "mvp_upstream_latency_seconds_count", model=model) == 2
    assert sample(text, "mvp_reconciliation_discrepancies_total", rule_id="grand_total") == 1


async def test_failed_job_is_counted_once_with_its_stage_timing() -> None:
    rejected = httpx.Response(
        400, json={"type": "error", "error": {"type": "invalid_request_error"}}
    )
    app, metrics = observed_app(Upstream(rejected))
    async with running(app) as client:
        final = await submit_and_wait(client)
    text = metrics.render()[0].decode()
    assert final["status"] == "failed"
    assert sample(text, "mvp_jobs_total", status="failed", mime_type="application/pdf") == 1
    assert sample(text, "mvp_jobs_total", status="succeeded") == 0
    assert sample(text, "mvp_job_duration_seconds_count", stage="extraction") == 1
    assert sample(text, "mvp_job_duration_seconds_count", stage="reconciliation") == 0


@pytest.mark.parametrize(
    ("failure", "label"),
    [(httpx.ReadTimeout("slow"), "timeout"), (httpx.ConnectError("refused"), "connection_error")],
)
async def test_transport_failures_are_labelled(failure: Exception, label: str) -> None:
    metrics = PipelineMetrics()

    def fail(request: httpx.Request) -> httpx.Response:
        raise failure

    client = VisionClient(
        api_key=SecretStr(API_KEY),
        transport=httpx.MockTransport(fail),
        metrics=metrics,
        sleep=no_sleep,
    )
    request = _vision_request(max_retries=1)
    with pytest.raises(Exception, match="attempt 2"):
        await client.extract(request)
    text = metrics.render()[0].decode()
    assert sample(text, "mvp_upstream_requests_total", status_code=label) == 2


def _vision_request(*, max_retries: int) -> VisionRequest:
    page = io.BytesIO()
    Image.new("RGB", (40, 60), "white").save(page, format="PNG")
    metadata = PageMetadata(
        page_number=1, width_px=40, height_px=60, dpi=200, color_mode=ColorMode.RGB
    )
    return VisionPayloadBuilder().build(
        [RasterizedPage(metadata=metadata, png=page.getvalue())],
        ExtractionModelConfig(max_retries=max_retries),
        page_count=1,
    )


def active(metrics: PipelineMetrics) -> float:
    return metrics.registry.get_sample_value("mvp_active_jobs") or 0.0


class GatedExtractor:
    def __init__(self, metrics: PipelineMetrics) -> None:
        self.metrics = metrics
        self.release = asyncio.Event()
        self.peak_active = 0.0

    async def extract(self, request: VisionRequest) -> VisionResponse:
        self.peak_active = max(self.peak_active, active(self.metrics))
        await self.release.wait()
        return VisionResponse(
            text=model_reply(),
            model="claude-sonnet-5-5",
            stop_reason="end_turn",
            request_id=None,
            attempts=1,
        )


async def test_active_jobs_gauge_tracks_running_work() -> None:
    metrics = PipelineMetrics()
    gate = GatedExtractor(metrics)
    jobs = JobManager()
    worker = PipelineWorker(
        jobs=jobs,
        rasterizer=FakeRasterizer(page_count=1, page_size_inches=(1.0, 1.0)),
        extractor=gate,
        metrics=metrics,
        max_concurrency=3,
    )
    descriptor = parse(DocumentDescriptor, descriptor_data())
    ids = [jobs.submit(descriptor, IngestionOptions()).job.job_id for _ in range(7)]
    tasks = [worker.submit(job_id, PDF) for job_id in ids]
    for _ in range(100):
        if active(metrics) == 3:
            break
        await asyncio.sleep(0.01)
    assert active(metrics) == 3  # bounded by the semaphore
    gate.release.set()
    await asyncio.gather(*tasks)
    text = metrics.render()[0].decode()
    assert sample(text, "mvp_active_jobs") == 0
    assert gate.peak_active <= 3
    assert sample(text, "mvp_jobs_total", status="succeeded") == 7


async def test_unmatched_routes_do_not_explode_label_cardinality() -> None:
    app, metrics = observed_app()
    async with running(app) as client:
        for n in range(5):
            await client.get(f"/no/such/path/{n}")
        await client.get(f"/api/v1/jobs/{UUID(int=1)}")
    text = metrics.render()[0].decode()
    assert sample(text, "mvp_http_requests_total", route="unmatched", status_code="404") == 5
    assert sample(text, "mvp_http_requests_total", route="/api/v1/jobs/{job_id}") == 1
    assert "/no/such/path" not in text


def test_concurrent_metric_updates_are_atomic() -> None:
    metrics = PipelineMetrics()
    threads, per_thread = 16, 2000
    barrier = threading.Barrier(threads)

    def hammer(_: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            metrics.job_finished(status="succeeded", mime_type="image/png")
            metrics.rasterized_pages.inc()
            metrics.upstream_attempt(model="m", status_code="200", seconds=0.01)
            metrics.http_request(method="GET", route="/x", status_code=200, seconds=0.001)
            with metrics.tracking_active_job(), metrics.stage("reconciliation"):
                metrics.discrepancies.labels(rule_id=f"r{i % 3}").inc()

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(hammer, range(threads)))
    text = metrics.render()[0].decode()
    total = threads * per_thread
    assert sample(text, "mvp_jobs_total", status="succeeded") == total
    assert sample(text, "mvp_rasterized_pages_total") == total
    assert sample(text, "mvp_upstream_requests_total") == total
    assert sample(text, "mvp_upstream_latency_seconds_count") == total
    assert sample(text, "mvp_http_requests_total") == total
    assert sample(text, "mvp_job_duration_seconds_count", stage="reconciliation") == total
    assert (
        sum(
            sample(text, "mvp_reconciliation_discrepancies_total", rule_id=f"r{k}")
            for k in range(3)
        )
        == total
    )
    assert sample(text, "mvp_active_jobs") == 0


def test_registries_are_isolated() -> None:
    first, second = PipelineMetrics(), PipelineMetrics()
    first.rasterized_pages.inc(5)
    assert sample(second.render()[0].decode(), "mvp_rasterized_pages_total") == 0


# ===========================================================================
# Correlation
# ===========================================================================
async def test_correlation_id_propagates_to_logs_and_upstream(logs: LogReader) -> None:
    upstream = Upstream()
    app, _ = observed_app(upstream)
    async with running(app) as client:
        await submit_and_wait(client, headers={"X-Correlation-ID": "corr-123"})
        echo = await client.get("/healthz", headers={"X-Correlation-ID": "corr-123"})
    assert echo.headers["x-correlation-id"] == "corr-123"
    assert re.fullmatch(r"[0-9a-f]{32}", echo.headers["x-request-id"])
    assert upstream.requests[0].headers["x-correlation-id"] == "corr-123"

    records = logs()
    worker_records = [r for r in records if r.get("job_id")]
    assert worker_records, "worker emitted no logs"
    assert all(r["correlation_id"] == "corr-123" for r in worker_records)
    stages = {r["stage"] for r in worker_records if "stage" in r}
    assert stages == {"rasterization", "extraction", "reconciliation"}
    completed = [r for r in worker_records if r["message"].startswith("stage ")]
    assert all(isinstance(r["latency_ms"], float) for r in completed)
    access = [r for r in records if r["logger"] == "multimodal_vision_pipeline.access"]
    assert any(r["correlation_id"] == "corr-123" and r["status_code"] == 202 for r in access)
    assert all("request_id" in r for r in access)


async def test_ids_are_generated_or_preserved() -> None:
    app, _ = observed_app()
    async with running(app) as client:
        fresh = await client.get("/healthz")
        kept = await client.get("/healthz", headers={"X-Request-ID": "req-from-gateway"})
        rejected = await client.get(
            "/healthz", headers={"X-Correlation-ID": "has spaces", "X-Request-ID": "x" * 300}
        )
    assert fresh.headers["x-correlation-id"] == fresh.headers["x-request-id"]
    assert kept.headers["x-request-id"] == "req-from-gateway"
    assert kept.headers["x-correlation-id"] == "req-from-gateway"
    assert rejected.headers["x-correlation-id"] != "has spaces"
    assert len(rejected.headers["x-request-id"]) == 32


async def test_problem_responses_carry_correlation_headers() -> None:
    app, _ = observed_app()
    async with running(app) as client:
        response = await client.get("/api/v1/jobs/not-a-uuid", headers={"X-Correlation-ID": "c-9"})
    assert response.status_code == 422
    assert response.headers["x-correlation-id"] == "c-9"


async def test_correlation_middleware_without_metrics() -> None:
    sent: list[Any] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message: Any) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    await CorrelationMiddleware(app)(scope, None, send)  # type: ignore[arg-type]
    names = {name for name, _ in sent[0]["headers"]}
    assert {b"x-correlation-id", b"x-request-id"} <= names


async def test_correlation_middleware_passes_other_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    await CorrelationMiddleware(app)({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan"]


async def test_context_is_isolated_between_tasks(logs: LogReader) -> None:
    log = logging.getLogger("multimodal_vision_pipeline.test")

    async def work(correlation_id: str) -> None:
        with bind(correlation_id=correlation_id):
            for _ in range(5):
                log.info("tick")
                await asyncio.sleep(0)

    await asyncio.gather(work("a"), work("b"))
    ticks = [r["correlation_id"] for r in logs() if r["message"] == "tick"]
    assert sorted(ticks) == ["a"] * 5 + ["b"] * 5
    assert current_context() == {}


def test_bind_restores_previous_values() -> None:
    with bind(correlation_id="outer", job_id="j1"):
        with bind(correlation_id="inner", stage="extraction"):
            assert current_context() == {
                "correlation_id": "inner",
                "job_id": "j1",
                "stage": "extraction",
            }
        assert current_context() == {"correlation_id": "outer", "job_id": "j1"}
    assert current_context() == {}
    with pytest.raises(ValueError, match="unknown context fields"), bind(tenant="x"):
        pass


@pytest.mark.parametrize(
    ("raw", "accepted"),
    [
        ("abc-123", True),
        ("a.b:c_d", True),
        ("", False),
        ("has space", False),
        ("-lead", False),
        ("x\r\ninjected", False),
        ("y" * 129, False),
        (None, False),
    ],
)
def test_accept_id(raw: str | None, accepted: bool) -> None:
    assert (accept_id(raw) is not None) is accepted


# ===========================================================================
# JSON logging
# ===========================================================================
def test_json_log_line_shape(logs: LogReader) -> None:
    log = logging.getLogger("multimodal_vision_pipeline.test")
    with bind(correlation_id="c-1", job_id="j-1", stage="extraction"):
        log.warning("stage %s took long", "extraction", extra={"latency_ms": 12.5, "pages": 3})
    (record,) = logs()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", record["timestamp"])
    assert record["level"] == "warning"
    assert record["logger"] == "multimodal_vision_pipeline.test"
    assert record["message"] == "stage extraction took long"
    assert (record["correlation_id"], record["job_id"], record["stage"]) == (
        "c-1",
        "j-1",
        "extraction",
    )
    assert (record["latency_ms"], record["pages"]) == (12.5, 3)


def test_rfc3339_formatting() -> None:
    assert rfc3339(0) == "1970-01-01T00:00:00.000Z"
    assert rfc3339(1_793_275_200.1234) == "2026-10-29T12:00:00.123Z"


def test_exceptions_are_logged_redacted(logs: LogReader) -> None:
    log = logging.getLogger("multimodal_vision_pipeline.test")

    def call_upstream() -> None:
        raise RuntimeError(f"upstream said x-api-key: {API_KEY}")

    try:
        call_upstream()
    except RuntimeError:
        log.exception("call failed")
    (record,) = logs()
    assert record["error_type"] == "RuntimeError"
    assert "RuntimeError" in record["traceback"]
    assert API_KEY not in json.dumps(record)


def test_explicit_extra_wins_over_context() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", (), None)
    record.job_id = "explicit"
    with bind(job_id="from-context", correlation_id="c"):
        ContextFilter().filter(record)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["job_id"] == "explicit"
    assert entry["correlation_id"] == "c"


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    level = root.level
    try:
        first = configure_logging("INFO", stream=io.StringIO())
        second = configure_logging("WARNING", stream=io.StringIO())
        ours = [h for h in root.handlers if h.get_name() == "mvp-json"]
        assert ours == [second]
        assert first not in root.handlers
        assert root.level == logging.WARNING
    finally:
        root.removeHandler(second)
        root.setLevel(level)


async def test_lifespan_installs_json_logging() -> None:
    root = logging.getLogger()
    level = root.level
    app = create_app(
        VisionSettings(anthropic_api_key=SecretStr(API_KEY), log_json=True, log_level="WARNING"),
        rasterizer=FakeRasterizer(),
        extractor=GatedExtractor(PipelineMetrics()),
    )
    try:
        async with running(app):
            assert any(h.get_name() == "mvp-json" for h in root.handlers)
            assert root.level == logging.WARNING
    finally:
        for handler in [h for h in root.handlers if h.get_name() == "mvp-json"]:
            root.removeHandler(handler)
        root.setLevel(level)


# ===========================================================================
# Redaction
# ===========================================================================
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4).decode()


@pytest.mark.parametrize(
    ("text", "secret", "marker"),
    [
        (f"key={API_KEY}", API_KEY, "[REDACTED_API_KEY]"),
        ("x-api-key: sk-test-0000-abcd", "sk-test-0000-abcd", REDACTED),
        (
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
            "eyJhbGciOiJIUzI1NiJ9",
            REDACTED,
        ),
        ("token Bearer abcdefghijklmnop", "abcdefghijklmnop", REDACTED),
        (f"image {PNG_B64} end", PNG_B64[:60], "[REDACTED_BASE64]"),
        (f"src=data:image/png;base64,{PNG_B64[:40]}", PNG_B64[:40], "[REDACTED_DATA_URI]"),
        ("pay to PL61 1090 1014 0000 0712 1981 2874 today", "1090 1014", "[REDACTED_IBAN]"),
        ("iban PL61109010140000071219812874", "PL61109010140000071219812874", "[REDACTED_IBAN]"),
        ("card 4111 1111 1111 1111 exp", "4111 1111 1111 1111", "[REDACTED_PAN]"),
        ("pan 4111111111111111.", "4111111111111111", "[REDACTED_PAN]"),
        ("amex 3782-822463-10005", "822463", "[REDACTED_PAN]"),
        ("contact billing@acme.example for help", "billing@acme.example", "[REDACTED_EMAIL]"),
    ],
)
def test_redact_text(text: str, secret: str, marker: str) -> None:
    cleaned = redact_text(text)
    assert secret not in cleaned
    assert marker in cleaned


@pytest.mark.parametrize(
    "text",
    [
        "job 6f1c2a52-5d0e-4c3e-9b7a-1f2e3d4c5b6a reached stage extraction (HTTP 529)",
        "uuid 00000000-0000-0000-0000-000000000007",
        "order 1234567890123 shipped",  # 13 digits, fails the Luhn check
        "invoice FV/2026/10/0042 total 6764",
    ],
)
def test_identifiers_that_are_not_secrets_survive(text: str) -> None:
    assert redact_text(text) == text


def test_long_text_is_truncated() -> None:
    cleaned = redact_text("word " * 2000)
    assert len(cleaned) < MAX_STRING_CHARS + 40
    assert cleaned.endswith("chars]")


def test_sanitize_structures() -> None:
    payload = {
        "model": "claude-sonnet-5-5",
        "api_key": "anything",
        "Authorization": "whatever",
        "messages": [{"role": "user", "content": [{"type": "image", "data": PNG_B64}]}],
        "raw": b"%PDF-1.7 secret bytes",
        "amount": Decimal("12.50"),
        "ids": {UUID(int=7)},
        "nested": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}},
        "many": list(range(80)),
    }
    cleaned = sanitize(payload)
    assert isinstance(cleaned, dict)
    assert cleaned["model"] == "claude-sonnet-5-5"
    assert cleaned["api_key"] == REDACTED
    assert cleaned["Authorization"] == REDACTED
    assert cleaned["messages"][0]["content"] == REDACTED
    assert cleaned["raw"] == "[REDACTED_BYTES len=21]"
    assert cleaned["amount"] == "12.50"
    assert cleaned["ids"] == ["00000000-0000-0000-0000-000000000007"]
    assert "[TRUNCATED_DEPTH]" in json.dumps(cleaned["nested"])
    assert len(cleaned["many"]) == 51
    wide = sanitize({f"k{i}": i for i in range(60)})
    assert isinstance(wide, dict)
    assert wide["…"] == "10 more"


def test_sensitive_extras_never_reach_the_log(logs: LogReader) -> None:
    log = logging.getLogger("multimodal_vision_pipeline.test")
    log.info(
        "sending %s",
        f"Bearer {API_KEY}",
        extra={
            "body": {"data": PNG_B64},
            "document": b"%PDF...",
            "headers": {"x-api-key": API_KEY},
        },
    )
    line = json.dumps(logs())
    assert API_KEY not in line
    assert PNG_B64[:60] not in line
    assert "%PDF" not in line


async def test_full_pipeline_logs_leak_nothing(logs: LogReader) -> None:
    upstream = Upstream()
    app, _ = observed_app(upstream)
    async with running(app) as client:
        await submit_and_wait(client)
    sent_body = upstream.requests[0].content.decode()
    image_data = json.loads(sent_body)["messages"][0]["content"][1]["source"]["data"]
    everything = json.dumps(logs())
    assert API_KEY not in everything
    assert image_data[:80] not in everything
    assert "PL5250001009" not in everything  # vendor tax id from the extraction
    assert "%PDF" not in everything
