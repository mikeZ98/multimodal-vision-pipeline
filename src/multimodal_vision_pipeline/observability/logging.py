"""Structured JSON logs: one object per line, correlation context attached, redacted."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final, TextIO

from multimodal_vision_pipeline.observability.context import current_context
from multimodal_vision_pipeline.observability.redaction import (
    REDACTED,
    SENSITIVE_KEYS,
    redact_text,
    sanitize,
)

CONTEXT_FIELDS: Final = ("correlation_id", "request_id", "job_id", "stage")
_RESERVED: Final = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def rfc3339(timestamp: float) -> str:
    """UTC RFC 3339 timestamp with millisecond precision, e.g. ``2026-10-29T12:00:00.123Z``."""
    moment = datetime.fromtimestamp(timestamp, UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ContextFilter(logging.Filter):
    """Copies the correlation context onto each record at emit time.

    Runs in the emitting task's context, so the values are those of the code that
    logged, even when a handler formats later.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in current_context().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True


class JsonFormatter(logging.Formatter):
    """Renders ``timestamp, level, logger, message`` plus context and sanitized extras."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": rfc3339(record.created),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }
        context = current_context()
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, None) or context.get(name)
            if value is not None:
                entry[name] = value
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in entry:
                entry[key] = REDACTED if key.lower() in SENSITIVE_KEYS else sanitize(value)
        if record.exc_info and record.exc_info[1] is not None:
            error = record.exc_info[1]
            entry["error_type"] = type(error).__name__
            entry["traceback"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(entry, ensure_ascii=False, default=str, separators=(",", ":"))


_HANDLER_NAME: Final = "mvp-json"


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> logging.Handler:
    """Install the JSON handler on the root logger, replacing a previous one. Idempotent."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
            existing.close()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    root.setLevel(level)
    return handler
