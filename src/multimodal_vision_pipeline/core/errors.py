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
