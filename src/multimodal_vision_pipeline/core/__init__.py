"""Processing engines: rasterization, vision payloads, reconciliation and job state."""

from multimodal_vision_pipeline.core.errors import (
    DocumentTooLargeError,
    ExtractionParseError,
    IdempotencyConflictError,
    InvalidTransitionError,
    JobCapacityError,
    JobNotFoundError,
    PageSelectionError,
    PayloadTooLargeError,
    PipelineError,
    RasterizationError,
    ReconciliationOverflowError,
    UnsupportedDocumentError,
)
from multimodal_vision_pipeline.core.job_manager import TRANSITIONS, JobManager, Submission
from multimodal_vision_pipeline.core.rasterizer import (
    DocumentRasterizer,
    FakeRasterizer,
    PageRasterizer,
    RasterizedPage,
    Rasterizer,
    plan_pages,
)
from multimodal_vision_pipeline.core.reconciler import reconcile
from multimodal_vision_pipeline.core.vision_builder import (
    EncodedImage,
    ImageLimits,
    VisionExtraction,
    VisionPayloadBuilder,
    VisionRequest,
    parse_extraction,
    structured_output_schema,
)

__all__ = [
    "TRANSITIONS",
    "DocumentRasterizer",
    "DocumentTooLargeError",
    "EncodedImage",
    "ExtractionParseError",
    "FakeRasterizer",
    "IdempotencyConflictError",
    "ImageLimits",
    "InvalidTransitionError",
    "JobCapacityError",
    "JobManager",
    "JobNotFoundError",
    "PageRasterizer",
    "PageSelectionError",
    "PayloadTooLargeError",
    "PipelineError",
    "RasterizationError",
    "RasterizedPage",
    "Rasterizer",
    "ReconciliationOverflowError",
    "Submission",
    "UnsupportedDocumentError",
    "VisionExtraction",
    "VisionPayloadBuilder",
    "VisionRequest",
    "parse_extraction",
    "plan_pages",
    "reconcile",
    "structured_output_schema",
]
