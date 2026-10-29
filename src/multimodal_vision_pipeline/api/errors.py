"""RFC 9457 problem responses and the exception handlers that produce them."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from http import HTTPStatus
from typing import Any, ClassVar, Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from multimodal_vision_pipeline.core.errors import PipelineError
from multimodal_vision_pipeline.schemas import PROBLEM_JSON_MEDIA_TYPE, InvalidParam, ProblemDetails

logger = logging.getLogger(__name__)

MAX_POINTER_CHARS: Final = 256
MAX_INVALID_PARAMS: Final = 100
_MAX_PARAM_DETAIL_CHARS: Final = 500


class ProblemResponse(JSONResponse):
    """``application/problem+json`` body for a ``ProblemDetails`` document."""

    media_type = PROBLEM_JSON_MEDIA_TYPE

    def __init__(self, problem: ProblemDetails, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(
            problem.model_dump(mode="json", exclude_none=True),
            status_code=problem.status,
            headers=headers,
        )


class InvalidRequestError(PipelineError):
    """The request is well-formed HTTP but its content is invalid; lists each field."""

    status: ClassVar[int] = 422
    title: ClassVar[str] = "Request validation failed"
    slug: ClassVar[str] = "validation"

    def __init__(self, message: str = "", *, errors: Sequence[InvalidParam] = ()) -> None:
        super().__init__(message)
        self.errors = tuple(errors[:MAX_INVALID_PARAMS])

    def to_problem(self, *, instance: str | None = None) -> ProblemDetails:
        base = super().to_problem(instance=instance)
        return ProblemDetails(**{**dict(base), "errors": self.errors})


class ServiceNotReadyError(PipelineError):
    status: ClassVar[int] = 503
    title: ClassVar[str] = "Service is not ready"
    slug: ClassVar[str] = "not-ready"


def json_pointer(location: Iterable[str | int]) -> str:
    """RFC 6901 pointer for a validation location, shortened to fit the schema bound."""
    segments = [str(part).replace("~", "~0").replace("/", "~1") for part in location]
    while segments and len("".join("/" + s for s in segments)) > MAX_POINTER_CHARS:
        segments.pop()
    return "".join("/" + s for s in segments)


def invalid_params(
    errors: Sequence[Mapping[str, Any]], *, prefix: Sequence[str | int] = ()
) -> tuple[InvalidParam, ...]:
    """Convert Pydantic/FastAPI error dicts into RFC 9457 ``errors`` entries."""
    params: list[InvalidParam] = []
    for error in errors[:MAX_INVALID_PARAMS]:
        detail = str(error.get("msg") or "invalid value")[:_MAX_PARAM_DETAIL_CHARS]
        location = (*prefix, *error.get("loc", ()))
        params.append(InvalidParam(pointer=json_pointer(location), detail=detail))
    return tuple(params)


def _instance(request: Request) -> str:
    return request.url.path


async def handle_pipeline_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, PipelineError):  # pragma: no cover - registered for PipelineError
        raise exc
    if exc.status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        logger.warning(
            "%s %s -> %d %s: %s", request.method, request.url.path, exc.status, exc.slug, exc
        )
    return ProblemResponse(exc.to_problem(instance=_instance(request)))


async def handle_validation_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    errors = invalid_params(exc.errors())
    error = InvalidRequestError(f"{len(exc.errors())} invalid request field(s)", errors=errors)
    return ProblemResponse(error.to_problem(instance=_instance(request)))


async def handle_http_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    status = exc.status_code if exc.status_code >= HTTPStatus.BAD_REQUEST else 500
    title = HTTPStatus(status).phrase
    detail = exc.detail if exc.detail and exc.detail != title else None
    problem = ProblemDetails(title=title, status=status, detail=detail, instance=_instance(request))
    return ProblemResponse(problem, headers=exc.headers)


async def handle_unexpected_error(request: Request, exc: Exception) -> Response:
    logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return ProblemResponse(PipelineError().to_problem(instance=_instance(request)))


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(PipelineError, handle_pipeline_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
