"""Domain errors. Each maps to one RFC 9457 problem type and HTTP status."""

from typing import ClassVar

from multimodal_vision_pipeline.schemas import ProblemDetails

PROBLEM_TYPE_PREFIX = "urn:multimodal-vision-pipeline:problem:"


class PipelineError(Exception):
    """Base for every expected failure; carries its HTTP status and problem slug."""

    status: ClassVar[int] = 500
    title: ClassVar[str] = "Internal processing error"
    slug: ClassVar[str] = "internal"

    def to_problem(self, *, instance: str | None = None) -> ProblemDetails:
        return ProblemDetails(
            type=PROBLEM_TYPE_PREFIX + self.slug,
            title=self.title,
            status=self.status,
            detail=str(self) or None,
            instance=instance,
        )


# --- rasterization ---------------------------------------------------------
class RasterizationError(PipelineError):
    status = 422
    title = "Document could not be rasterized"
    slug = "rasterization-failed"


class UnsupportedDocumentError(RasterizationError):
    status = 415
    title = "Unsupported document"
    slug = "unsupported-document"


class DocumentTooLargeError(RasterizationError):
    status = 413
    title = "Document too large"
    slug = "document-too-large"


class PageSelectionError(RasterizationError):
    title = "Page selection matches no page"
    slug = "page-selection"


# --- vision payload / extraction ------------------------------------------
class PayloadTooLargeError(PipelineError):
    status = 422
    title = "Pages exceed vision request limits"
    slug = "vision-payload-too-large"


class ExtractionParseError(PipelineError):
    status = 502
    title = "Vision model returned an invalid extraction"
    slug = "extraction-invalid"


class VisionRefusalError(PipelineError):
    status = 422
    title = "Vision model declined to process the document"
    slug = "vision-refusal"


# --- upstream vision API ---------------------------------------------------
class UpstreamError(PipelineError):
    """The vision API failed. Carries the upstream envelope for logs, not for clients."""

    status = 502
    title = "Vision service error"
    slug = "upstream-error"

    def __init__(
        self,
        message: str = "",
        *,
        error_type: str | None = None,
        upstream_status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.upstream_status = upstream_status
        self.request_id = request_id


class UpstreamRateLimitedError(UpstreamError):
    status = 503
    title = "Vision service rate limit exceeded"
    slug = "upstream-rate-limited"


class UpstreamOverloadedError(UpstreamError):
    status = 503
    title = "Vision service overloaded"
    slug = "upstream-overloaded"


class UpstreamTimeoutError(UpstreamError):
    status = 504
    title = "Vision service timed out"
    slug = "upstream-timeout"


class UpstreamUnavailableError(UpstreamError):
    title = "Vision service unreachable"
    slug = "upstream-unavailable"


class UpstreamRejectedError(UpstreamError):
    title = "Vision service rejected the request"
    slug = "upstream-rejected"


class UpstreamProtocolError(UpstreamError):
    title = "Vision service returned a malformed response"
    slug = "upstream-protocol"


# --- reconciliation --------------------------------------------------------
class ReconciliationOverflowError(PipelineError):
    status = 422
    title = "Extracted amounts exceed the supported range"
    slug = "reconciliation-overflow"


# --- jobs ------------------------------------------------------------------
class JobNotFoundError(PipelineError):
    status = 404
    title = "Job not found"
    slug = "job-not-found"


class InvalidTransitionError(PipelineError):
    status = 409
    title = "Invalid job state transition"
    slug = "invalid-transition"


class IdempotencyConflictError(PipelineError):
    status = 409
    title = "Idempotency key reused with a different request"
    slug = "idempotency-conflict"


class JobCapacityError(PipelineError):
    status = 503
    title = "Job store is full"
    slug = "job-capacity"


class JobTimeoutError(PipelineError):
    status = 504
    title = "Job exceeded its processing deadline"
    slug = "job-timeout"


class JobCancelledError(PipelineError):
    status = 503
    title = "Job was cancelled"
    slug = "job-cancelled"
