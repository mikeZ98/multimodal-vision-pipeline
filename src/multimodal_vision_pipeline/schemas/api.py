"""HTTP wire models: ingestion envelopes, async job tracking and RFC 9457 problems."""

from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from multimodal_vision_pipeline.schemas._base import IdempotencyKey, PipelineModel, UtcDatetime
from multimodal_vision_pipeline.schemas.document import (
    DocumentDescriptor,
    PageMetadata,
    RasterizationConfig,
)
from multimodal_vision_pipeline.schemas.extraction import (
    DocumentKind,
    ExtractionModelConfig,
    ExtractionResult,
)
from multimodal_vision_pipeline.schemas.reconciliation import (
    JsonPointer,
    ReconciliationPolicy,
    ReconciliationReport,
)

PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"

UriReference = Annotated[str, StringConstraints(min_length=1, max_length=2048, pattern=r"^\S+$")]
"""RFC 3986 URI reference (absolute or relative); whitespace is never valid."""


class JobStatus(StrEnum):
    """Job lifecycle: ``queued → rasterizing → extracting → reconciling → succeeded | failed``."""

    QUEUED = "queued"
    RASTERIZING = "rasterizing"
    EXTRACTING = "extracting"
    RECONCILING = "reconciling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED}


class IngestionOptions(PipelineModel):
    """Processing options sent alongside the uploaded file (multipart JSON part).

    ``extraction=None`` uses the service's configured model settings.
    """

    document_kind: DocumentKind | None = None
    rasterization: RasterizationConfig = RasterizationConfig()
    extraction: ExtractionModelConfig | None = None
    reconciliation: ReconciliationPolicy = ReconciliationPolicy()
    idempotency_key: IdempotencyKey | None = None


class IngestionAccepted(PipelineModel):
    """``202 Accepted`` body: the job was created (or replayed for an idempotency key)."""

    job_id: UUID
    status: JobStatus
    status_url: UriReference
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _check_status_url(self) -> Self:
        if not self.status_url.rstrip("/").endswith(str(self.job_id)):
            raise ValueError("status_url must address the accepted job")
        return self


class InvalidParam(PipelineModel):
    """RFC 9457 extension member: one offending request field."""

    pointer: JsonPointer
    detail: str = Field(min_length=1, max_length=500)


class ProblemDetails(PipelineModel):
    """RFC 9457 problem document, served as ``application/problem+json``."""

    type: UriReference = "about:blank"
    title: str = Field(min_length=1, max_length=200)
    status: int = Field(ge=400, le=599)
    detail: str | None = Field(default=None, max_length=2000)
    instance: UriReference | None = None
    errors: tuple[InvalidParam, ...] = Field(default=(), max_length=100)


class JobStatusResponse(PipelineModel):
    """Observable state of an ingestion job.

    Succeeded jobs carry extraction and reconciliation; failed jobs carry an error;
    running jobs carry neither an error nor a reconciliation report.
    """

    job_id: UUID
    status: JobStatus
    document: DocumentDescriptor
    created_at: UtcDatetime
    updated_at: UtcDatetime
    pages: tuple[PageMetadata, ...] = Field(default=(), max_length=500)
    extraction: ExtractionResult | None = None
    reconciliation: ReconciliationReport | None = None
    error: ProblemDetails | None = None

    @model_validator(mode="after")
    def _check_lifecycle(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        numbers = [p.page_number for p in self.pages]
        if numbers != sorted(set(numbers)):
            raise ValueError("pages must be unique and in ascending page order")
        if self.extraction is not None and self.pages:
            rendered = set(numbers)
            if max(rendered) > self.extraction.page_count:
                raise ValueError("rasterized pages exceed the extraction page_count")
            for anchor in self.extraction.anchors():
                if anchor.page_number not in rendered:
                    raise ValueError(f"anchor references page {anchor.page_number}, not rasterized")
        match self.status:
            case JobStatus.SUCCEEDED:
                if self.extraction is None or self.reconciliation is None or self.error:
                    raise ValueError(
                        "succeeded job requires extraction and reconciliation, no error"
                    )
            case JobStatus.FAILED:
                if self.error is None:
                    raise ValueError("failed job requires an error")
            case _:
                if self.error is not None or self.reconciliation is not None:
                    raise ValueError(f"{self.status} job must not carry an error or reconciliation")
        return self
