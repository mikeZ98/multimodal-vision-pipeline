"""Phase 4 verification: resilient vision client and the async pipeline worker.

Hermetic: every upstream call goes through ``httpx.MockTransport``; backoff sleeps
are recorded rather than awaited; pages come from the in-memory fake rasterizer.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from multimodal_vision_pipeline.core import (
    ExtractionParseError,
    FakeRasterizer,
    InvalidTransitionError,
    JobManager,
    PipelineError,
    RasterizedPage,
    VisionPayloadBuilder,
    VisionRequest,
)
from multimodal_vision_pipeline.core.errors import (
    PROBLEM_TYPE_PREFIX,
    UpstreamError,
    UpstreamOverloadedError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    VisionRefusalError,
)
from multimodal_vision_pipeline.core.rasterizer import CORRUPT_MARKER
from multimodal_vision_pipeline.schemas import (
    ColorMode,
    DocumentDescriptor,
    ExtractionModelConfig,
    IngestionOptions,
    JobStatus,
    JobStatusResponse,
    PageMetadata,
    RasterizationConfig,
    ReconciliationStatus,
    VisionModelId,
    VisionSettings,
)
from multimodal_vision_pipeline.tools import (
    API_VERSION,
    FALLBACK_BETA,
    PipelineWorker,
    RetryPolicy,
    VisionClient,
    VisionResponse,
    error_from_response,
    parse_message,
    parse_retry_after,
)
from tests.factories import descriptor_data, ef, invoice_data, parse

API_KEY = "sk-test-0000"
Responder = Callable[[httpx.Request], Awaitable[httpx.Response] | httpx.Response]


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------
def reply_text(**overrides: object) -> str:
    data = invoice_data(**overrides)
    del data["model"], data["page_count"]
    return json.dumps(data)


def anchorless_reply(document_number: str) -> str:
    data = invoice_data(document_number=ef(document_number), total=ef(6764))
    data["vendor"]["name"] = ef("ACME")
    for item in data["line_items"]:
        item["anchor"] = None
    del data["model"], data["page_count"]
    return json.dumps(data)


def message(
    text: str | None = None,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-5-5",
    blocks: Sequence[dict[str, Any]] = (),
    **extra: Any,
) -> dict[str, Any]:
    content = [*blocks]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1200, "output_tokens": 300},
        **extra,
    }


def ok(payload: object, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=payload, headers={"request-id": "req_ok", **headers})


def api_error(status: int, error_type: str, msg: str = "boom", **headers: str) -> httpx.Response:
    body = {"type": "error", "error": {"type": error_type, "message": msg}, "request_id": "req_x"}
    return httpx.Response(status, json=body, headers=headers)


class Upstream:
    """Scripted MockTransport handler: replays steps in order, then repeats the last."""

    def __init__(self, *steps: httpx.Response | Exception | Responder) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps[min(len(self.requests), len(self.steps)) - 1]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        result = step(request)
        return await result if isinstance(result, Awaitable) else result

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def make_client(
    upstream: Upstream, *, sleep: Any = None, **kwargs: Any
) -> tuple[VisionClient, SleepRecorder]:
    recorder = SleepRecorder()
    client = VisionClient(
        api_key=SecretStr(API_KEY),
        transport=httpx.MockTransport(upstream),
        sleep=sleep or recorder,
        rng=random.Random(1234),  # noqa: S311 - deterministic jitter for assertions
        **kwargs,
    )
    return client, recorder


def png_page(page_number: int) -> RasterizedPage:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 60), "white").save(buffer, format="PNG")
    metadata = PageMetadata(
        page_number=page_number, width_px=40, height_px=60, dpi=200, color_mode=ColorMode.RGB
    )
    return RasterizedPage(metadata=metadata, png=buffer.getvalue())


BUILDER = VisionPayloadBuilder()


def vision_request(
    model: VisionModelId = VisionModelId.SONNET_5_5, *, max_retries: int = 2, pages: int = 2
) -> VisionRequest:
    config = ExtractionModelConfig(model=model, max_retries=max_retries)
    return BUILDER.build([png_page(n) for n in range(1, pages + 1)], config, page_count=pages)


# ===========================================================================
# Vision client - happy path and request shape
# ===========================================================================
async def test_successful_extraction_request_shape() -> None:
    upstream = Upstream(ok(message(reply_text())))
    client, sleeps = make_client(upstream)
    async with client:
        response = await client.extract(vision_request())
    (request,) = upstream.requests
    assert request.method == "POST"
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == API_KEY
    assert request.headers["anthropic-version"] == API_VERSION == "2026-10-21"
    assert request.headers["anthropic-beta"] == FALLBACK_BETA
    assert request.headers["content-type"] == "application/json"
    body = upstream.bodies[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(body)
    assert response.attempts == 1
    assert response.request_id == "req_ok"
    assert response.usage["input_tokens"] == 1200
    assert json.loads(response.text)["total"]["value"] == 6764
    assert sleeps.delays == []


async def test_timeouts_are_bounded_per_phase() -> None:
    upstream = Upstream(ok(message("{}")))
    client, _ = make_client(upstream)
    request = vision_request()
    await client.extract(request)
    timeout = upstream.requests[0].extensions["timeout"]
    assert timeout == {"connect": 10.0, "read": 120.0, "write": 60.0, "pool": 10.0}


@pytest.mark.parametrize(
    ("model", "fallback_enabled", "expect_fallback"),
    [
        (VisionModelId.OPUS_5_5, True, True),
        (VisionModelId.SONNET_5_5, False, False),
        (VisionModelId.HAIKU_5_5, True, False),  # no server-side fallback on this tier
    ],
)
async def test_server_side_fallback_opt_in(
    model: VisionModelId, fallback_enabled: bool, expect_fallback: bool
) -> None:
    upstream = Upstream(ok(message("{}", model=model.value)))
    client, _ = make_client(upstream, server_side_fallback=fallback_enabled)
    await client.extract(vision_request(model))
    assert ("fallbacks" in upstream.bodies[0]) is expect_fallback
    assert ("anthropic-beta" in upstream.requests[0].headers) is expect_fallback


async def test_thinking_and_fallback_blocks_are_skipped() -> None:
    blocks: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {
            "type": "fallback",
            "from": {"model": "claude-sonnet-5-5"},
            "to": {"model": "claude-sonnet-5"},
        },
    ]
    upstream = Upstream(ok(message('{"a": 1}', blocks=blocks, model="claude-sonnet-5")))
    client, _ = make_client(upstream)
    response = await client.extract(vision_request())
    assert response.text == '{"a": 1}'
    assert response.fallback_used
    assert response.model == "claude-sonnet-5"


# ===========================================================================
# Vision client - retries and backoff
# ===========================================================================
async def test_rate_limit_honours_retry_after() -> None:
    upstream = Upstream(
        api_error(429, "rate_limit_error", **{"retry-after": "7"}), ok(message("{}"))
    )
    client, sleeps = make_client(upstream)
    response = await client.extract(vision_request())
    assert response.attempts == 2
    assert sleeps.delays == [pytest.approx(7.0)]


async def test_retry_after_is_capped() -> None:
    upstream = Upstream(
        api_error(429, "rate_limit_error", **{"retry-after": "3600"}), ok(message("{}"))
    )
    client, sleeps = make_client(upstream, retry=RetryPolicy(max_retry_after=45))
    await client.extract(vision_request())
    assert sleeps.delays == [45]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(500, "api_error"), (503, "api_error"), (529, "overloaded_error"), (502, "api_error")],
)
async def test_transient_server_errors_are_retried(status: int, error_type: str) -> None:
    upstream = Upstream(api_error(status, error_type), ok(message("{}")))
    client, sleeps = make_client(upstream)
    assert (await client.extract(vision_request())).attempts == 2
    assert len(sleeps.delays) == 1


async def test_bounded_retries_with_decorrelated_jitter_then_failure() -> None:
    upstream = Upstream(api_error(529, "overloaded_error", "Overloaded"))
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0)
    client, sleeps = make_client(upstream, retry=policy)
    with pytest.raises(UpstreamOverloadedError) as caught:
        await client.extract(vision_request(max_retries=5))
    assert len(upstream.requests) == 6  # 1 attempt + 5 retries
    assert len(sleeps.delays) == 5
    previous = policy.base_delay
    for delay in sleeps.delays:
        assert (
            policy.base_delay
            <= delay
            <= min(policy.max_delay, max(policy.base_delay, previous * 3))
        )
        previous = delay
    assert caught.value.upstream_status == 529
    assert caught.value.error_type == "overloaded_error"
    assert caught.value.request_id == "req_x"


async def test_backoff_grows_until_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
    rng = random.Random(7)  # noqa: S311
    delay, delays = policy.base_delay, []
    for _ in range(40):
        delay = policy.next_delay(delay, rng)
        delays.append(delay)
    assert max(delays) <= 30.0
    assert min(delays) >= 1.0
    assert max(delays[10:]) > 10.0  # it does grow, not hover at the base


async def test_zero_retries_means_one_attempt() -> None:
    upstream = Upstream(api_error(500, "api_error"))
    client, sleeps = make_client(upstream)
    with pytest.raises(UpstreamError):
        await client.extract(vision_request(max_retries=0))
    assert len(upstream.requests) == 1
    assert sleeps.delays == []


@pytest.mark.parametrize(
    ("status", "error_type", "error_cls"),
    [
        (400, "invalid_request_error", UpstreamRejectedError),
        (401, "authentication_error", UpstreamRejectedError),
        (403, "permission_error", UpstreamRejectedError),
        (404, "not_found_error", UpstreamRejectedError),
        (413, "request_too_large", UpstreamRejectedError),
    ],
)
async def test_client_errors_are_not_retried(
    status: int, error_type: str, error_cls: type[UpstreamError]
) -> None:
    upstream = Upstream(api_error(status, error_type, f"bad {error_type}"))
    client, sleeps = make_client(upstream)
    with pytest.raises(error_cls) as caught:
        await client.extract(vision_request())
    assert len(upstream.requests) == 1
    assert sleeps.delays == []
    problem = caught.value.to_problem()
    assert problem.status == 502
    assert API_KEY not in (problem.detail or "")


async def test_should_retry_header_overrides_status() -> None:
    no = Upstream(api_error(500, "api_error", **{"x-should-retry": "false"}))
    client, _ = make_client(no)
    with pytest.raises(UpstreamError):
        await client.extract(vision_request())
    assert len(no.requests) == 1

    yes = Upstream(
        api_error(400, "invalid_request_error", **{"x-should-retry": "true"}), ok(message("{}"))
    )
    client, _ = make_client(yes)
    assert (await client.extract(vision_request())).attempts == 2


@pytest.mark.parametrize(
    "drop",
    [
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.ReadError("connection reset"),
    ],
)
async def test_connection_drops_are_retried(drop: Exception) -> None:
    upstream = Upstream(drop, drop, ok(message("{}")))
    client, sleeps = make_client(upstream)
    assert (await client.extract(vision_request())).attempts == 3
    assert len(sleeps.delays) == 2


async def test_persistent_connection_failure() -> None:
    upstream = Upstream(httpx.ConnectError("refused"))
    client, _ = make_client(upstream)
    with pytest.raises(UpstreamUnavailableError, match="ConnectError on attempt 3"):
        await client.extract(vision_request())


async def test_persistent_read_timeout() -> None:
    upstream = Upstream(httpx.ReadTimeout("slow"))
    client, _ = make_client(upstream)
    with pytest.raises(UpstreamTimeoutError) as caught:
        await client.extract(vision_request(max_retries=1))
    assert len(upstream.requests) == 2
    assert caught.value.to_problem().status == 504


async def test_non_json_error_page_is_retried_and_mapped() -> None:
    gateway = httpx.Response(502, text="<html>Bad Gateway</html>")
    upstream = Upstream(gateway)
    client, _ = make_client(upstream)
    with pytest.raises(UpstreamError, match="HTTP 502") as caught:
        await client.extract(vision_request(max_retries=1))
    assert len(upstream.requests) == 2
    assert type(caught.value) is UpstreamError


# ===========================================================================
# Vision client - malformed and unusable responses
# ===========================================================================
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="{truncated", headers={"request-id": "req_bad"}),
        ok({"type": "error"}),
        ok(["not", "an", "object"]),
        ok(message(None)),  # no text block
        ok({**message("{}"), "content": "oops"}),
    ],
)
async def test_malformed_success_bodies(response: httpx.Response) -> None:
    upstream = Upstream(response)
    client, _ = make_client(upstream)
    with pytest.raises(UpstreamProtocolError):
        await client.extract(vision_request())
    assert len(upstream.requests) == 1


async def test_refusal_is_not_retried() -> None:
    refusal = message(
        None, stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}
    )
    upstream = Upstream(ok(refusal))
    client, _ = make_client(upstream)
    with pytest.raises(VisionRefusalError, match="cyber") as caught:
        await client.extract(vision_request())
    assert caught.value.to_problem().status == 422


async def test_truncated_output_is_reported() -> None:
    upstream = Upstream(ok(message('{"partial":', stop_reason="max_tokens")))
    client, _ = make_client(upstream)
    with pytest.raises(ExtractionParseError, match="max_tokens"):
        await client.extract(vision_request())


@pytest.mark.parametrize(
    ("response", "cls", "problem_status"),
    [
        (api_error(429, "rate_limit_error"), UpstreamRateLimitedError, 503),
        (api_error(529, "overloaded_error"), UpstreamOverloadedError, 503),
        (api_error(500, "api_error"), UpstreamError, 502),
        (api_error(400, "invalid_request_error"), UpstreamRejectedError, 502),
        (api_error(504, "timeout_error"), UpstreamTimeoutError, 504),
        (httpx.Response(429, text="slow down"), UpstreamRateLimitedError, 503),
        (httpx.Response(529), UpstreamOverloadedError, 503),
        (httpx.Response(408), UpstreamTimeoutError, 504),
        (httpx.Response(418, json={"error": "teapot"}), UpstreamRejectedError, 502),
        (httpx.Response(500, json={"type": "error", "error": {"type": 7}}), UpstreamError, 502),
    ],
)
def test_error_envelope_mapping(
    response: httpx.Response, cls: type[UpstreamError], problem_status: int
) -> None:
    error = error_from_response(response)
    assert type(error) is cls
    problem = error.to_problem()
    assert problem.status == problem_status
    assert problem.type.startswith(PROBLEM_TYPE_PREFIX + "upstream-")


def test_error_message_is_truncated_and_request_id_kept() -> None:
    response = api_error(400, "invalid_request_error", "x" * 1000, **{"request-id": "req_hdr"})
    error = error_from_response(response)
    assert len(str(error)) == 300
    assert error.request_id == "req_hdr"
    assert error_from_response(api_error(400, "invalid_request_error")).request_id == "req_x"


def test_parse_message_directly() -> None:
    response = parse_message(message("{}"), request_id=None, attempts=3)
    assert isinstance(response, VisionResponse)
    assert response.attempts == 3
    assert parse_message(message("{}", usage=None), request_id=None, attempts=1).usage == {}


# ===========================================================================
# Vision client - retry-after parsing, policy and lifecycle
# ===========================================================================
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("2", 2.0),
        ("0.25", 0.25),
        ("-4", 0.0),
        (format_datetime(NOW + timedelta(seconds=30), usegmt=True), 30.0),
        (format_datetime(NOW - timedelta(seconds=30), usegmt=True), 0.0),
        ("Thu, 29 Oct 2026 12:00:10", 10.0),  # no zone: read as UTC
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_retry_after(header: str | None, expected: float | None) -> None:
    assert parse_retry_after(header, now=NOW) == expected


def test_retry_after_relative_to_wall_clock() -> None:
    future = format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)
    assert 3500 < (parse_retry_after(future) or 0) <= 3600


@pytest.mark.parametrize(
    "kwargs", [{"base_delay": 0}, {"base_delay": 5, "max_delay": 1}, {"max_retry_after": -1}]
)
def test_retry_policy_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"require|max_retry_after"):
        RetryPolicy(**kwargs)


def test_client_construction_guards() -> None:
    with pytest.raises(ValueError, match="empty"):
        VisionClient(api_key=SecretStr(""))
    with pytest.raises(ValueError, match="MVP_ANTHROPIC_API_KEY"):
        VisionClient.from_settings(VisionSettings(anthropic_api_key=None))


async def test_from_settings_uses_configured_endpoint() -> None:
    upstream = Upstream(ok(message("{}")))
    settings = VisionSettings(
        anthropic_api_key=SecretStr(API_KEY),
        anthropic_base_url="https://gateway.example.invalid/",  # type: ignore[arg-type]
        vision_server_side_fallback=False,
    )
    async with VisionClient.from_settings(
        settings, transport=httpx.MockTransport(upstream)
    ) as client:
        await client.extract(vision_request())
    assert str(upstream.requests[0].url) == "https://gateway.example.invalid/v1/messages"
    assert "fallbacks" not in upstream.bodies[0]


# ===========================================================================
# Vision client - cancellation
# ===========================================================================
async def test_cancellation_during_request_propagates() -> None:
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    upstream = Upstream(hang)
    client, sleeps = make_client(upstream)
    task = asyncio.create_task(client.extract(vision_request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(upstream.requests) == 1
    assert sleeps.delays == []


async def test_cancellation_during_backoff_stops_retrying() -> None:
    sleeping = asyncio.Event()

    async def blocking_sleep(delay: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    upstream = Upstream(api_error(529, "overloaded_error"))
    client, _ = make_client(upstream, sleep=blocking_sleep)
    task = asyncio.create_task(client.extract(vision_request(max_retries=5)))
    await sleeping.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(upstream.requests) == 1


async def test_real_sleep_is_used_by_default() -> None:
    upstream = Upstream(api_error(503, "api_error"), ok(message("{}")))
    client = VisionClient(
        api_key=SecretStr(API_KEY),
        transport=httpx.MockTransport(upstream),
        retry=RetryPolicy(base_delay=0.001, max_delay=0.002),
    )
    async with client:
        assert (await client.extract(vision_request())).attempts == 2


# ===========================================================================
# Pipeline worker
# ===========================================================================
class RecordingJobs(JobManager):
    def __init__(self) -> None:
        super().__init__()
        self.history: dict[UUID, list[JobStatus]] = {}

    def _advance(self, job_id: UUID, target: JobStatus, **changes: Any) -> JobStatusResponse:
        snapshot = super()._advance(job_id, target, **changes)
        self.history.setdefault(job_id, []).append(target)
        return snapshot


def descriptor() -> DocumentDescriptor:
    return parse(DocumentDescriptor, descriptor_data())


def new_job(jobs: JobManager, **options: Any) -> UUID:
    return jobs.submit(descriptor(), IngestionOptions(**options)).job.job_id


def worker_with(
    upstream: Upstream | None = None,
    *,
    extractor: Any = None,
    page_count: int = 3,
    **kwargs: Any,
) -> tuple[PipelineWorker, RecordingJobs]:
    jobs = RecordingJobs()
    if extractor is None:
        assert upstream is not None
        extractor, _ = make_client(upstream)
    worker = PipelineWorker(
        jobs=jobs,
        rasterizer=FakeRasterizer(page_count=page_count, page_size_inches=(1.0, 1.5)),
        extractor=extractor,
        **kwargs,
    )
    return worker, jobs


DONE = [JobStatus.RASTERIZING, JobStatus.EXTRACTING, JobStatus.RECONCILING, JobStatus.SUCCEEDED]


async def test_end_to_end_success() -> None:
    upstream = Upstream(ok(message(reply_text())))
    worker, jobs = worker_with(upstream)
    job_id = new_job(jobs)
    final = await worker.process(job_id, b"%PDF-fake")
    assert final.status is JobStatus.SUCCEEDED
    assert jobs.history[job_id] == DONE
    assert [p.page_number for p in final.pages] == [1, 2, 3]
    assert final.extraction is not None
    assert final.extraction.page_count == 3
    assert final.extraction.model is VisionModelId.SONNET_5_5
    assert final.reconciliation is not None
    assert final.reconciliation.status is ReconciliationStatus.PASSED
    content = upstream.bodies[0]["messages"][0]["content"]
    assert [b["type"] for b in content].count("image") == 3


async def test_background_submission() -> None:
    worker, jobs = worker_with(Upstream(ok(message(reply_text()))))
    job_id = new_job(jobs)
    task = worker.submit(job_id, b"%PDF")
    assert worker.active_jobs == {job_id}
    assert (await task).status is JobStatus.SUCCEEDED
    assert worker.active_jobs == frozenset()


async def test_failed_arithmetic_is_a_successful_job() -> None:
    worker, jobs = worker_with(Upstream(ok(message(reply_text(total=ef(9999))))))
    final = await worker.process(new_job(jobs), b"%PDF")
    assert final.status is JobStatus.SUCCEEDED
    assert final.reconciliation is not None
    assert final.reconciliation.status is ReconciliationStatus.FAILED


@pytest.mark.parametrize(
    ("upstream", "slug", "status", "last_state"),
    [
        (
            Upstream(api_error(529, "overloaded_error")),
            "upstream-overloaded",
            503,
            JobStatus.EXTRACTING,
        ),
        (
            Upstream(api_error(400, "invalid_request_error")),
            "upstream-rejected",
            502,
            JobStatus.EXTRACTING,
        ),
        (Upstream(ok(message("not json at all"))), "extraction-invalid", 502, JobStatus.EXTRACTING),
        (
            Upstream(ok(message(reply_text(total=ef(-1))))),
            "extraction-invalid",
            502,
            JobStatus.EXTRACTING,
        ),
        (
            Upstream(httpx.Response(200, text="<html>")),
            "upstream-protocol",
            502,
            JobStatus.EXTRACTING,
        ),
    ],
)
async def test_upstream_and_parse_failures_fail_the_job(
    upstream: Upstream, slug: str, status: int, last_state: JobStatus
) -> None:
    worker, jobs = worker_with(upstream)
    job_id = new_job(jobs)
    final = await worker.process(job_id, b"%PDF")
    assert final.status is JobStatus.FAILED
    assert jobs.history[job_id][-2:] == [last_state, JobStatus.FAILED]
    assert final.error is not None
    assert final.error.type == PROBLEM_TYPE_PREFIX + slug
    assert final.error.status == status
    assert final.error.instance == f"/v1/jobs/{job_id}"


async def test_rasterization_failure_fails_the_job() -> None:
    worker, jobs = worker_with(Upstream(ok(message(reply_text()))))
    job_id = new_job(jobs)
    final = await worker.process(job_id, CORRUPT_MARKER + b"...")
    assert jobs.history[job_id] == [JobStatus.RASTERIZING, JobStatus.FAILED]
    assert final.error is not None
    assert final.error.status == 415


async def test_reconciliation_overflow_fails_the_job() -> None:
    item = invoice_data()["line_items"][0] | {
        "quantity": "999999999999",
        "unit_price_cents": 10**15,
        "amount_cents": 10**15,
        "anchor": None,
    }
    worker, jobs = worker_with(Upstream(ok(message(reply_text(line_items=[item])))))
    job_id = new_job(jobs)
    final = await worker.process(job_id, b"%PDF")
    assert jobs.history[job_id][-2:] == [JobStatus.RECONCILING, JobStatus.FAILED]
    assert final.error is not None
    assert final.error.type.endswith("reconciliation-overflow")


class StubExtractor:
    def __init__(self, behaviour: Callable[[VisionRequest], Awaitable[VisionResponse]]) -> None:
        self.behaviour = behaviour
        self.calls = 0

    async def extract(self, request: VisionRequest) -> VisionResponse:
        self.calls += 1
        return await self.behaviour(request)


async def test_unexpected_error_fails_without_leaking_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def explode(request: VisionRequest) -> VisionResponse:
        raise RuntimeError("secret internal state")

    worker, jobs = worker_with(extractor=StubExtractor(explode))
    final = await worker.process(new_job(jobs), b"%PDF")
    assert final.status is JobStatus.FAILED
    assert final.error is not None
    assert final.error.status == 500
    assert final.error.detail is None
    assert "secret internal state" in caplog.text  # logged for operators, not returned


async def test_job_deadline_fails_the_job() -> None:
    async def hang(request: VisionRequest) -> VisionResponse:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    worker, jobs = worker_with(extractor=StubExtractor(hang), job_timeout_seconds=0.05)
    final = await worker.process(new_job(jobs), b"%PDF")
    assert final.status is JobStatus.FAILED
    assert final.error is not None
    assert final.error.type.endswith("job-timeout")
    assert final.error.status == 504


async def test_cancellation_mid_job_marks_it_failed_and_propagates() -> None:
    entered = asyncio.Event()

    async def hang(request: VisionRequest) -> VisionResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    worker, jobs = worker_with(extractor=StubExtractor(hang))
    job_id = new_job(jobs)
    task = worker.submit(job_id, b"%PDF")
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    final = jobs.get(job_id)
    assert final.status is JobStatus.FAILED
    assert final.error is not None
    assert final.error.type.endswith("job-cancelled")
    assert jobs.history[job_id][-2:] == [JobStatus.EXTRACTING, JobStatus.FAILED]


async def test_cancellation_while_queued_for_a_slot() -> None:
    entered = asyncio.Event()

    async def hang(request: VisionRequest) -> VisionResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    worker, jobs = worker_with(extractor=StubExtractor(hang), max_concurrency=1)
    first, second = new_job(jobs), new_job(jobs)
    first_task = worker.submit(first, b"%PDF")
    second_task = worker.submit(second, b"%PDF")
    await entered.wait()
    assert jobs.get(second).status is JobStatus.QUEUED  # waiting for the only slot
    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task
    assert jobs.history[second] == [JobStatus.FAILED]
    await worker.shutdown()
    assert first_task.cancelled()
    assert jobs.get(first).status is JobStatus.FAILED


async def test_shutdown_leaves_no_job_running() -> None:
    async def slow(request: VisionRequest) -> VisionResponse:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    worker, jobs = worker_with(extractor=StubExtractor(slow), max_concurrency=2)
    ids = [new_job(jobs) for _ in range(5)]
    for job_id in ids:
        worker.submit(job_id, b"%PDF")
    await asyncio.sleep(0.05)
    await worker.shutdown()
    assert all(jobs.get(job_id).status is JobStatus.FAILED for job_id in ids)
    assert worker.active_jobs == frozenset()


async def test_concurrency_is_bounded_and_jobs_stay_isolated() -> None:
    in_flight = 0
    peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        images = [
            b for b in json.loads(request.content)["messages"][0]["content"] if b["type"] == "image"
        ]
        return ok(message(anchorless_reply(f"DOC-{len(images)}")))

    worker, jobs = worker_with(Upstream(respond), max_concurrency=3, page_count=5)
    plan = {
        new_job(jobs, rasterization=RasterizationConfig(max_pages=n % 5 + 1)): n % 5 + 1
        for n in range(12)
    }
    tasks = [worker.submit(job_id, b"%PDF") for job_id in plan]
    finals = await asyncio.gather(*tasks)
    assert peak == 3
    assert all(f.status is JobStatus.SUCCEEDED for f in finals)
    for final in finals:
        expected_pages = plan[final.job_id]
        assert len(final.pages) == expected_pages
        assert final.extraction is not None
        assert final.extraction.document_number.value == f"DOC-{expected_pages}"
        assert jobs.history[final.job_id] == DONE


async def test_same_job_cannot_be_processed_twice() -> None:
    worker, jobs = worker_with(Upstream(ok(message(reply_text()))))
    job_id = new_job(jobs)
    results = await asyncio.gather(
        worker.process(job_id, b"%PDF"), worker.process(job_id, b"%PDF"), return_exceptions=True
    )
    assert sum(isinstance(r, InvalidTransitionError) for r in results) == 1
    assert jobs.get(job_id).status is JobStatus.SUCCEEDED
    assert jobs.history[job_id] == DONE

    other = new_job(jobs)
    task = worker.submit(other, b"%PDF")
    with pytest.raises(InvalidTransitionError, match="already scheduled"):
        worker.submit(other, b"%PDF")
    await task


async def test_job_options_override_default_model() -> None:
    upstream = Upstream(ok(message(reply_text())))
    worker, jobs = worker_with(
        upstream, default_extraction=ExtractionModelConfig(model=VisionModelId.OPUS_5_5)
    )
    await worker.process(new_job(jobs), b"%PDF")
    override = ExtractionModelConfig(model=VisionModelId.HAIKU_5_5)
    final = await worker.process(new_job(jobs, extraction=override), b"%PDF")
    assert [b["model"] for b in upstream.bodies] == ["claude-opus-5-5", "claude-haiku-5-5"]
    assert final.extraction is not None
    assert final.extraction.model is VisionModelId.HAIKU_5_5


async def test_worker_from_settings_and_guards() -> None:
    settings = VisionSettings(
        vision_model=VisionModelId.HAIKU_5_5,
        worker_max_concurrency=7,
        worker_job_timeout_seconds=30,
    )

    async def unused(request: VisionRequest) -> VisionResponse:
        raise AssertionError("not called")

    extractor = StubExtractor(unused)
    worker = PipelineWorker.from_settings(
        settings, jobs=JobManager(), rasterizer=FakeRasterizer(), extractor=extractor
    )
    assert worker.max_concurrency == 7
    assert worker.job_timeout_seconds == 30
    assert worker.default_extraction.model is VisionModelId.HAIKU_5_5
    with pytest.raises(ValueError, match="max_concurrency"):
        PipelineWorker(
            jobs=JobManager(), rasterizer=FakeRasterizer(), extractor=extractor, max_concurrency=0
        )
    with pytest.raises(ValueError, match="job_timeout_seconds"):
        PipelineWorker(
            jobs=JobManager(),
            rasterizer=FakeRasterizer(),
            extractor=extractor,
            job_timeout_seconds=0,
        )


async def test_pipeline_error_base_is_handled() -> None:
    async def fail(request: VisionRequest) -> VisionResponse:
        raise PipelineError("plain")

    worker, jobs = worker_with(extractor=StubExtractor(fail))
    final = await worker.process(new_job(jobs), b"%PDF")
    assert final.error is not None
    assert final.error.status == 500
    assert final.error.detail == "plain"
