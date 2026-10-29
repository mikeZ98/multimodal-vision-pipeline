"""Phase 5 verification: FastAPI routes, uploads, problem documents and lifecycle.

Hermetic: requests go through ``httpx.ASGITransport`` (no listening socket), the
lifespan is driven in-process, upstream calls hit ``httpx.MockTransport`` and pages
come from the fake rasterizer.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image
from pydantic import SecretStr
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from multimodal_vision_pipeline.api import Services, create_app
from multimodal_vision_pipeline.api.errors import handle_http_error, json_pointer
from multimodal_vision_pipeline.api.middleware import BodySizeLimitMiddleware
from multimodal_vision_pipeline.api.uploads import (
    MULTIPART_OVERHEAD_BYTES,
    resolve_mime_type,
    safe_filename,
    sniff_mime_type,
)
from multimodal_vision_pipeline.core import FakeRasterizer, JobManager, UnsupportedDocumentError
from multimodal_vision_pipeline.core.errors import PROBLEM_TYPE_PREFIX
from multimodal_vision_pipeline.core.vision_builder import VisionRequest
from multimodal_vision_pipeline.schemas import (
    DocumentMimeType,
    JobStatus,
    ReconciliationStatus,
    VisionSettings,
)
from multimodal_vision_pipeline.tools import VisionClient, VisionResponse
from tests.factories import messages_api_body, model_reply

PROBLEM = "application/problem+json"
PDF = b"%PDF-1.7\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def image_bytes(fmt: str, frames: int = 1) -> bytes:
    images = [Image.new("RGB", (32, 48), "white") for _ in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format=fmt, save_all=frames > 1, append_images=images[1:])
    return buffer.getvalue()


DOCUMENTS = {
    "application/pdf": PDF,
    "image/tiff": image_bytes("TIFF", frames=2),
    "image/png": image_bytes("PNG"),
    "image/jpeg": image_bytes("JPEG"),
}


class Upstream:
    def __init__(self, handler: Callable[[httpx.Request], Awaitable[httpx.Response]] | None = None):
        self.calls = 0
        self.handler = handler

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.handler is not None:
            return await self.handler(request)
        return httpx.Response(200, json=messages_api_body(model_reply()))


class GatedExtractor:
    """Blocks every extraction until ``release`` is set; counts calls."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.calls = 0

    async def extract(self, request: VisionRequest) -> VisionResponse:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return VisionResponse(
            text=model_reply(),
            model="claude-sonnet-5-5",
            stop_reason="end_turn",
            request_id=None,
            attempts=1,
        )


def settings(**overrides: Any) -> VisionSettings:
    base: dict[str, Any] = {
        "anthropic_api_key": SecretStr("sk-test"),
        "worker_shutdown_grace_seconds": 5,
    }
    return VisionSettings(**(base | overrides))


def build_app(
    upstream: Upstream | None = None,
    *,
    extractor: Any = None,
    jobs: JobManager | None = None,
    **overrides: Any,
) -> FastAPI:
    if extractor is None:
        transport = httpx.MockTransport(upstream or Upstream())
        extractor = VisionClient(api_key=SecretStr("sk-test"), transport=transport)
    return create_app(
        settings(**overrides),
        jobs=jobs,
        rasterizer=FakeRasterizer(page_count=2, page_size_inches=(1.0, 1.5)),
        extractor=extractor,
    )


@asynccontextmanager
async def running(
    app: FastAPI, *, raise_app_exceptions: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def services(app: FastAPI) -> Services:
    value = app.state.services
    assert isinstance(value, Services)
    return value


async def upload(
    client: httpx.AsyncClient,
    data: bytes = PDF,
    content_type: str = "application/pdf",
    *,
    filename: str = "invoice.pdf",
    options: dict[str, Any] | str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    form: dict[str, str] = {}
    if options is not None:
        form["options"] = options if isinstance(options, str) else json.dumps(options)
    return await client.post(
        "/api/v1/documents",
        files={"file": (filename, data, content_type)},
        data=form,
        headers=headers,
    )


async def wait_terminal(client: httpx.AsyncClient, location: str) -> dict[str, Any]:
    for _ in range(500):
        response = await client.get(location)
        body: dict[str, Any] = response.json()
        if JobStatus(body["status"]).is_terminal:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"job at {location} never finished")


def assert_problem(
    response: httpx.Response, status: int, *, slug: str | None = None
) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM
    body: dict[str, Any] = response.json()
    assert body["status"] == status
    assert body["title"]
    if slug is not None:
        assert body["type"] == PROBLEM_TYPE_PREFIX + slug
    return body


# ===========================================================================
# Health
# ===========================================================================
async def test_liveness_and_readiness() -> None:
    async with running(build_app(worker_max_concurrency=3, worker_max_queued_jobs=9)) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        ready = await client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "worker": {
                "accepting": True,
                "active_jobs": 0,
                "max_concurrency": 3,
                "max_queued_jobs": 9,
            },
            "stored_jobs": 0,
            "max_jobs": 10_000,
        }


async def test_not_ready_before_startup() -> None:
    app = build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/healthz")).status_code == 200
        assert_problem(await client.get("/readyz"), 503, slug="not-ready")
        assert_problem(await upload(client), 503, slug="not-ready")


# ===========================================================================
# Ingestion flow
# ===========================================================================
@pytest.mark.parametrize("mime", list(DOCUMENTS))
async def test_ingest_and_track_to_completion(mime: str) -> None:
    upstream = Upstream()
    async with running(build_app(upstream)) as client:
        response = await upload(client, DOCUMENTS[mime], mime, filename=f"doc.{mime.split('/')[1]}")
        assert response.status_code == 202
        accepted = response.json()
        job_id = UUID(accepted["job_id"])
        assert response.headers["location"] == accepted["status_url"] == f"/api/v1/jobs/{job_id}"
        assert accepted["status"] == "queued"

        final = await wait_terminal(client, response.headers["location"])
    assert final["status"] == "succeeded"
    assert final["document"]["mime_type"] == mime
    assert final["document"]["size_bytes"] == len(DOCUMENTS[mime])
    expected_pages = 2 if DocumentMimeType(mime).is_multipage else 1
    assert [p["page_number"] for p in final["pages"]] == list(range(1, expected_pages + 1))
    assert final["extraction"]["total"]["value"] == 6764
    assert final["reconciliation"]["status"] == ReconciliationStatus.PASSED
    assert upstream.calls == 1


async def test_job_snapshot_headers_while_running() -> None:
    gate = GatedExtractor()
    async with running(build_app(extractor=gate)) as client:
        location = (await upload(client)).headers["location"]
        await gate.entered.wait()
        running_job = await client.get(location)
        assert running_job.status_code == 200
        assert running_job.json()["status"] == "extracting"
        assert running_job.headers["retry-after"] == "1"
        assert running_job.headers["cache-control"] == "no-store"
        gate.release.set()
        final = await wait_terminal(client, location)
        done = await client.get(location)
    assert final["status"] == "succeeded"
    assert "retry-after" not in done.headers


async def test_generic_content_type_is_sniffed() -> None:
    async with running(build_app()) as client:
        response = await upload(
            client, image_bytes("PNG"), "application/octet-stream", filename="scan"
        )
        assert response.status_code == 202
        final = await wait_terminal(client, response.headers["location"])
    assert final["document"]["mime_type"] == "image/png"


async def test_filename_is_sanitised() -> None:
    async with running(build_app()) as client:
        response = await upload(client, filename="C:\\fakepath\\..\\invoice.pdf")
        final = await wait_terminal(client, response.headers["location"])
    assert final["document"]["filename"] == "invoice.pdf"


async def test_options_are_applied() -> None:
    upstream = Upstream()
    options = {"rasterization": {"max_pages": 1}, "extraction": {"model": "claude-haiku-5-5"}}
    async with running(build_app(upstream)) as client:
        response = await upload(client, options=options)
        final = await wait_terminal(client, response.headers["location"])
    assert len(final["pages"]) == 1
    assert final["extraction"]["model"] == "claude-haiku-5-5"


async def test_upstream_failure_is_reported_on_the_job() -> None:
    async def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
        )

    async with running(build_app(Upstream(reject))) as client:
        response = await upload(client)
        final = await wait_terminal(client, response.headers["location"])
    assert final["status"] == "failed"
    assert final["error"]["type"] == PROBLEM_TYPE_PREFIX + "upstream-rejected"
    assert final["error"]["instance"] == response.headers["location"]


# ===========================================================================
# Idempotency
# ===========================================================================
async def test_idempotent_resubmission_returns_existing_job() -> None:
    upstream = Upstream()
    async with running(build_app(upstream)) as client:
        first = await upload(client, headers={"Idempotency-Key": "upload-42"})
        await wait_terminal(client, first.headers["location"])
        second = await upload(client, headers={"Idempotency-Key": "upload-42"})
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.headers["location"] == first.headers["location"]
    assert second.json()["status"] == "succeeded"
    assert upstream.calls == 1


async def test_idempotency_key_in_options_is_honoured() -> None:
    async with running(build_app()) as client:
        first = await upload(client, options={"idempotency_key": "opt-1"})
        second = await upload(
            client, options={"idempotency_key": "opt-1"}, headers={"Idempotency-Key": "opt-1"}
        )
    assert (first.status_code, second.status_code) == (202, 200)


async def test_reused_key_with_different_document_conflicts() -> None:
    async with running(build_app()) as client:
        await upload(client, headers={"Idempotency-Key": "k-1"})
        conflict = await upload(
            client, image_bytes("PNG"), "image/png", headers={"Idempotency-Key": "k-1"}
        )
    assert_problem(conflict, 409, slug="idempotency-conflict")


async def test_replay_does_not_need_capacity() -> None:
    gate = GatedExtractor()
    async with running(
        build_app(extractor=gate, worker_max_concurrency=1, worker_max_queued_jobs=1)
    ) as client:
        first = await upload(client, headers={"Idempotency-Key": "busy"})
        replay = await upload(client, headers={"Idempotency-Key": "busy"})
        fresh = await upload(client)
        gate.release.set()
    assert (first.status_code, replay.status_code) == (202, 200)
    assert_problem(fresh, 503, slug="job-capacity")


@pytest.mark.parametrize(
    ("headers", "options", "pointer"),
    [
        ({"Idempotency-Key": "has spaces"}, None, "/headers/Idempotency-Key"),
        ({"Idempotency-Key": "a"}, {"idempotency_key": "b"}, "/options/idempotency_key"),
    ],
)
async def test_bad_idempotency_keys(
    headers: dict[str, str], options: dict[str, Any] | None, pointer: str
) -> None:
    async with running(build_app()) as client:
        body = assert_problem(
            await upload(client, headers=headers, options=options), 422, slug="validation"
        )
    assert [e["pointer"] for e in body["errors"]] == [pointer]


# ===========================================================================
# Size limits
# ===========================================================================
LIMIT = 2048


async def test_file_over_limit_is_413() -> None:
    app = build_app(max_upload_bytes=LIMIT)
    async with running(app) as client:
        body = assert_problem(
            await upload(client, PDF + b"0" * LIMIT), 413, slug="document-too-large"
        )
        assert str(LIMIT) in body["detail"]
    assert len(services(app).jobs) == 0


async def test_file_at_limit_is_accepted() -> None:
    document = PDF + b"0" * (LIMIT - len(PDF))
    async with running(build_app(max_upload_bytes=LIMIT)) as client:
        assert (await upload(client, document)).status_code == 202


async def test_declared_body_over_limit_rejected_before_reading() -> None:
    app = build_app(max_upload_bytes=LIMIT)
    async with running(app) as client:
        body = assert_problem(
            await upload(client, PDF + b"0" * (LIMIT + MULTIPART_OVERHEAD_BYTES)), 413
        )
        assert len(services(app).jobs) == 0
    assert body["title"] == "Content Too Large"


async def test_streamed_body_over_limit_is_cut_off() -> None:
    boundary = "x-boundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    chunks_sent = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal chunks_sent
        yield head + PDF
        for _ in range(200):  # ~13 MB if nothing stopped it
            chunks_sent += 1
            yield b"0" * 65_536

    app = build_app(max_upload_bytes=LIMIT)
    async with running(app) as client:
        response = await client.post(
            "/api/v1/documents",
            content=body(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
    assert "content-length" not in response.request.headers
    assert_problem(response, 413)
    assert chunks_sent < 5


async def test_options_part_is_bounded() -> None:
    async with running(build_app()) as client:
        response = await upload(client, options="x" * (70 * 1024))
    assert_problem(response, 400)


async def test_invalid_content_length_header() -> None:
    sent: list[MutableMapping[str, Any]] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("must not be reached")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    middleware = BodySizeLimitMiddleware(app, max_body_bytes=10)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/x",
        "headers": [(b"content-length", b"-1")],
    }
    await middleware(scope, receive, send)
    assert sent[0]["status"] == 400
    with pytest.raises(ValueError, match="max_body_bytes"):
        BodySizeLimitMiddleware(app, max_body_bytes=0)


async def test_middleware_forwards_disconnects() -> None:
    received: list[dict[str, Any]] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        received.append(await receive())

    async def disconnect() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}
    await BodySizeLimitMiddleware(app, max_body_bytes=10)(scope, disconnect, None)  # type: ignore[arg-type]
    assert received == [{"type": "http.disconnect"}]


async def test_middleware_passes_through_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    await BodySizeLimitMiddleware(app, max_body_bytes=10)({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan"]


# ===========================================================================
# Media types
# ===========================================================================
@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (b"GIF89a....", "image/gif"),
        (b"PK\x03\x04", "application/zip"),
        (PDF, "image/png"),  # declared type contradicts the content
        (image_bytes("PNG"), "application/pdf"),
        (b"plain text", "application/octet-stream"),  # nothing recognisable
    ],
)
async def test_unsupported_or_mislabelled_media_is_415(data: bytes, content_type: str) -> None:
    async with running(build_app()) as client:
        assert_problem(await upload(client, data, content_type), 415, slug="unsupported-document")


@pytest.mark.parametrize(
    ("declared", "data", "expected"),
    [
        ("image/jpg", DOCUMENTS["image/jpeg"], DocumentMimeType.JPEG),
        ("IMAGE/TIF; charset=binary", DOCUMENTS["image/tiff"], DocumentMimeType.TIFF),
        (None, PDF, DocumentMimeType.PDF),
        ("", b"MM\x00*rest", DocumentMimeType.TIFF),
    ],
)
def test_mime_resolution(declared: str | None, data: bytes, expected: DocumentMimeType) -> None:
    assert resolve_mime_type(declared, data[:16]) is expected


def test_sniffing_and_filenames() -> None:
    assert sniff_mime_type(b"\x89PNG\r\n\x1a\n...") is DocumentMimeType.PNG
    assert sniff_mime_type(b"hello") is None
    with pytest.raises(UnsupportedDocumentError):
        resolve_mime_type("text/plain", b"hello")
    assert safe_filename(None) == "upload"
    assert safe_filename("../../") == "upload"
    assert safe_filename("a/b/c.pdf") == "c.pdf"
    assert safe_filename("\x00\x1fbad\x7f.pdf") == "bad.pdf"
    assert len(safe_filename("x" * 400)) == 255


# ===========================================================================
# Validation and problem formatting
# ===========================================================================
async def test_missing_file_part() -> None:
    async with running(build_app()) as client:
        response = await client.post("/api/v1/documents", data={"options": "{}"})
    body = assert_problem(response, 422, slug="validation")
    assert body["errors"] == [{"pointer": "/file", "detail": "field required"}]
    assert body["instance"] == "/api/v1/documents"


async def test_empty_file() -> None:
    async with running(build_app()) as client:
        body = assert_problem(await upload(client, b""), 422)
    assert body["errors"][0]["pointer"] == "/file"


async def test_options_must_be_text() -> None:
    async with running(build_app()) as client:
        response = await client.post(
            "/api/v1/documents",
            files={
                "file": ("a.pdf", PDF, "application/pdf"),
                "options": ("o.json", b"{}", "application/json"),
            },
        )
    body = assert_problem(response, 422)
    assert body["errors"] == [{"pointer": "/options", "detail": "expected a JSON string"}]


async def test_unexpected_extra_fields_are_rejected() -> None:
    async with running(build_app()) as client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("a.pdf", PDF, "application/pdf")},
            data={"options": "{}", "extra": "1"},
        )
    assert_problem(response, 400)


async def test_options_validation_errors_are_pointed() -> None:
    options = {"rasterization": {"dpi": 1000, "color_mode": "CMYK"}, "surprise": True}
    async with running(build_app()) as client:
        body = assert_problem(await upload(client, options=options), 422, slug="validation")
    pointers = {e["pointer"] for e in body["errors"]}
    assert pointers == {
        "/options/rasterization/dpi",
        "/options/rasterization/color_mode",
        "/options/surprise",
    }
    assert all(e["detail"] for e in body["errors"])


async def test_options_must_be_json() -> None:
    async with running(build_app()) as client:
        body = assert_problem(await upload(client, options="{not json"), 422)
    assert body["errors"][0]["pointer"] == "/options"


async def test_malformed_job_id_is_a_validation_problem() -> None:
    async with running(build_app()) as client:
        body = assert_problem(await client.get("/api/v1/jobs/not-a-uuid"), 422, slug="validation")
    assert body["errors"][0]["pointer"] == "/path/job_id"
    assert body["instance"] == "/api/v1/jobs/not-a-uuid"


async def test_unknown_job_is_404() -> None:
    job_id = uuid4()
    async with running(build_app()) as client:
        body = assert_problem(await client.get(f"/api/v1/jobs/{job_id}"), 404, slug="job-not-found")
    assert body["instance"] == f"/api/v1/jobs/{job_id}"


async def test_framework_errors_become_problems() -> None:
    async with running(build_app()) as client:
        missing = assert_problem(await client.get("/nope"), 404)
        wrong_method = await client.delete("/api/v1/documents")
    assert missing["type"] == "about:blank"
    assert missing["title"] == "Not Found"
    assert_problem(wrong_method, 405)
    assert wrong_method.headers["allow"] == "POST"


async def test_unexpected_errors_are_opaque_500s() -> None:
    class BrokenJobs(JobManager):
        def get(self, job_id: UUID) -> Any:
            raise RuntimeError("database password is hunter2")

    async with running(build_app(jobs=BrokenJobs()), raise_app_exceptions=False) as client:
        response = await client.get(f"/api/v1/jobs/{uuid4()}")
    body = assert_problem(response, 500, slug="internal")
    assert "hunter2" not in response.text
    assert "detail" not in body


async def test_http_error_handler_normalises_odd_statuses() -> None:
    request = Request(
        {"type": "http", "method": "GET", "path": "/x", "headers": [], "query_string": b""}
    )
    response = await handle_http_error(request, StarletteHTTPException(304))
    assert response.status_code == 500
    teapot = await handle_http_error(request, StarletteHTTPException(418, detail="short and stout"))
    assert json.loads(bytes(teapot.body))["detail"] == "short and stout"


@pytest.mark.parametrize(
    ("location", "pointer"),
    [
        (("options", "rasterization", "dpi"), "/options/rasterization/dpi"),
        (("a/b", "c~d", 0), "/a~1b/c~0d/0"),
        ((), ""),
        (("x" * 200, "y" * 100), "/" + "x" * 200),
    ],
)
def test_json_pointer(location: tuple[str | int, ...], pointer: str) -> None:
    assert json_pointer(location) == pointer


async def test_openapi_documents_routes() -> None:
    async with running(build_app()) as client:
        spec = (await client.get("/openapi.json")).json()
    ingest = spec["paths"]["/api/v1/documents"]["post"]
    assert "multipart/form-data" in ingest["requestBody"]["content"]
    assert {"202", "409", "413", "415", "422"} <= set(ingest["responses"])
    assert "application/problem+json" in ingest["responses"]["415"]["content"]
    assert "/api/v1/jobs/{job_id}" in spec["paths"]


# ===========================================================================
# Capacity and lifecycle
# ===========================================================================
async def test_capacity_exhaustion_is_503_and_not_ready() -> None:
    gate = GatedExtractor()
    app = build_app(extractor=gate, worker_max_concurrency=1, worker_max_queued_jobs=1)
    async with running(app) as client:
        assert (await upload(client)).status_code == 202
        busy = await upload(client)
        readiness = await client.get("/readyz")
        gate.release.set()
    assert_problem(busy, 503, slug="job-capacity")
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_ready"
    assert len(services(app).jobs) == 1  # the rejected upload created no job


async def test_shutdown_drains_in_flight_jobs() -> None:
    gate = GatedExtractor()
    app = build_app(extractor=gate, worker_shutdown_grace_seconds=5)
    transport = httpx.ASGITransport(app=app)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        location = (await upload(client)).headers["location"]
        await gate.entered.wait()

        stopping = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0.02)
        assert not stopping.done()  # waiting for the in-flight job
        assert (await client.get("/readyz")).status_code == 503
        assert_problem(await upload(client), 503, slug="job-capacity")

        gate.release.set()
        await stopping
        final = (await client.get(location)).json()
    assert final["status"] == "succeeded"


async def test_shutdown_cancels_jobs_after_grace_period() -> None:
    gate = GatedExtractor()  # never released
    app = build_app(extractor=gate, worker_shutdown_grace_seconds=0.05)
    async with running(app) as client:
        location = (await upload(client)).headers["location"]
        await gate.entered.wait()
    final = services(app).jobs.get(UUID(location.rsplit("/", 1)[1]))
    assert final.status is JobStatus.FAILED
    assert final.error is not None
    assert final.error.type == PROBLEM_TYPE_PREFIX + "job-cancelled"


async def test_lifespan_owns_and_closes_its_vision_client() -> None:
    app = create_app(settings(), rasterizer=FakeRasterizer())
    async with running(app):
        client = services(app).extractor
        assert isinstance(client, VisionClient)
    assert client._http.is_closed


async def test_lifespan_requires_api_key_without_injected_client() -> None:
    app = create_app(settings(anthropic_api_key=None))
    with pytest.raises(ValueError, match="MVP_ANTHROPIC_API_KEY"):
        async with app.router.lifespan_context(app):
            pass


async def test_injected_empty_job_store_is_used() -> None:
    jobs = JobManager(max_jobs=5)
    app = build_app(jobs=jobs)
    async with running(app) as client:
        await upload(client)
        assert (await client.get("/readyz")).json()["max_jobs"] == 5
    assert services(app).jobs is jobs
    assert len(jobs) == 1
