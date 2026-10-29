"""Document ingestion and job tracking routes."""

from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from multimodal_vision_pipeline.api.dependencies import JobsDep, SettingsDep, WorkerDep
from multimodal_vision_pipeline.api.errors import InvalidRequestError, invalid_params
from multimodal_vision_pipeline.api.uploads import OPTIONS_MAX_BYTES, read_upload
from multimodal_vision_pipeline.schemas import (
    API_PREFIX,
    PROBLEM_JSON_MEDIA_TYPE,
    DocumentDescriptor,
    IngestionAccepted,
    IngestionOptions,
    InvalidParam,
    JobStatusResponse,
    ProblemDetails,
    job_path,
)

router = APIRouter(prefix=API_PREFIX, tags=["documents"])

POLL_AFTER_SECONDS: Final = 1
IDEMPOTENCY_HEADER: Final = "Idempotency-Key"


def _problem(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {PROBLEM_JSON_MEDIA_TYPE: {"schema": ProblemDetails.model_json_schema()}},
    }


_INGEST_REQUEST_BODY: Final = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "PDF, TIFF, PNG or JPEG document.",
                        },
                        "options": {
                            "type": "string",
                            "description": "JSON-encoded IngestionOptions.",
                        },
                    },
                }
            }
        },
    }
}


def _parse_options(raw: UploadFile | str | None, header_key: str | None) -> IngestionOptions:
    if isinstance(raw, UploadFile):
        raise InvalidRequestError(
            "options must be a text field",
            errors=(InvalidParam(pointer="/options", detail="expected a JSON string"),),
        )
    try:
        options = IngestionOptions() if raw is None else IngestionOptions.model_validate_json(raw)
    except ValidationError as exc:
        raise InvalidRequestError(
            "invalid ingestion options", errors=invalid_params(exc.errors(), prefix=("options",))
        ) from None
    if header_key is None:
        return options
    if options.idempotency_key not in {None, header_key}:
        raise InvalidRequestError(
            f"{IDEMPOTENCY_HEADER} header and options.idempotency_key differ",
            errors=(
                InvalidParam(pointer="/options/idempotency_key", detail="conflicts with header"),
            ),
        )
    try:
        return IngestionOptions(**{**dict(options), "idempotency_key": header_key})
    except ValidationError:
        raise InvalidRequestError(
            f"invalid {IDEMPOTENCY_HEADER} header",
            errors=(
                InvalidParam(
                    pointer=f"/headers/{IDEMPOTENCY_HEADER}",
                    detail="1-128 characters: letters, digits and . _ : - (not first)",
                ),
            ),
        ) from None


@router.post(
    "/documents",
    status_code=202,
    response_model=IngestionAccepted,
    summary="Upload a document for asynchronous extraction",
    responses={
        200: {"model": IngestionAccepted, "description": "Idempotent replay of an existing job"},
        409: _problem("Idempotency key already bound to a different request"),
        413: _problem("Document or request body too large"),
        415: _problem("Unsupported or mislabelled document type"),
        422: _problem("Invalid file or options"),
        503: _problem("No capacity to accept the job"),
    },
    openapi_extra=_INGEST_REQUEST_BODY,
)
async def ingest_document(
    request: Request,
    settings: SettingsDep,
    jobs: JobsDep,
    worker: WorkerDep,
    idempotency_key: Annotated[str | None, Header(alias=IDEMPOTENCY_HEADER)] = None,
) -> Response:
    # Two file slots so a mistakenly file-typed ``options`` part gets a precise 422.
    async with request.form(max_files=2, max_fields=1, max_part_size=OPTIONS_MAX_BYTES) as form:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise InvalidRequestError(
                "a 'file' part is required",
                errors=(InvalidParam(pointer="/file", detail="field required"),),
            )
        options = _parse_options(form.get("options"), idempotency_key)
        received = await read_upload(upload, max_bytes=settings.max_upload_bytes)
    document = DocumentDescriptor(
        filename=received.filename,
        mime_type=received.mime_type,
        size_bytes=len(received.data),
        sha256=received.sha256,
    )
    # No awaits from here on: the capacity check, job creation and scheduling are atomic
    # with respect to other requests on this event loop.
    key = options.idempotency_key
    if key is None or jobs.find_by_idempotency_key(key) is None:
        worker.ensure_capacity()
    job, created = jobs.submit(document, options)
    if created:
        worker.submit(job.job_id, received.data)
    location = job_path(job.job_id)
    body = IngestionAccepted(
        job_id=job.job_id, status=job.status, status_url=location, created_at=job.created_at
    )
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=202 if created else 200,
        headers={"Location": location},
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Job state, page metadata, extraction and reconciliation",
    responses={404: _problem("No such job"), 422: _problem("Malformed job id")},
)
async def get_job(job_id: UUID, jobs: JobsDep) -> Response:
    snapshot = jobs.get(job_id)
    headers = {"Cache-Control": "no-store"}
    if not snapshot.status.is_terminal:
        headers["Retry-After"] = str(POLL_AFTER_SECONDS)
    return Response(snapshot.model_dump_json(), media_type="application/json", headers=headers)
