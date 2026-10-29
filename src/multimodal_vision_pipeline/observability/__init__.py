"""Structured logging, correlation context, redaction and Prometheus metrics."""

from multimodal_vision_pipeline.observability.context import (
    CORRELATION_HEADER,
    REQUEST_ID_HEADER,
    bind,
    correlation_id_var,
    current_context,
)
from multimodal_vision_pipeline.observability.logging import (
    ContextFilter,
    JsonFormatter,
    configure_logging,
)
from multimodal_vision_pipeline.observability.metrics import PipelineMetrics
from multimodal_vision_pipeline.observability.middleware import CorrelationMiddleware
from multimodal_vision_pipeline.observability.redaction import redact_text, sanitize

__all__ = [
    "CORRELATION_HEADER",
    "REQUEST_ID_HEADER",
    "ContextFilter",
    "CorrelationMiddleware",
    "JsonFormatter",
    "PipelineMetrics",
    "bind",
    "configure_logging",
    "correlation_id_var",
    "current_context",
    "redact_text",
    "sanitize",
]
