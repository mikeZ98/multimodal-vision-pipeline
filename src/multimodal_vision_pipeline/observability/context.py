"""Correlation context carried across ``await`` points and into spawned tasks.

``asyncio.create_task`` copies the current context, so a job scheduled while handling
a request keeps that request's correlation id for its whole lifetime.
"""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Final

CORRELATION_HEADER: Final = "X-Correlation-ID"
REQUEST_ID_HEADER: Final = "X-Request-ID"

_SAFE_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
stage_var: ContextVar[str | None] = ContextVar("stage", default=None)

_VARS: Final[dict[str, ContextVar[str | None]]] = {
    "correlation_id": correlation_id_var,
    "request_id": request_id_var,
    "job_id": job_id_var,
    "stage": stage_var,
}


def new_id() -> str:
    return uuid.uuid4().hex


def accept_id(raw: str | None) -> str | None:
    """Return ``raw`` if it is a safe identifier (no whitespace or control characters)."""
    return raw if raw is not None and _SAFE_ID.fullmatch(raw) else None


def current_context() -> dict[str, str]:
    """The bound context values that are set, by field name."""
    return {name: value for name, var in _VARS.items() if (value := var.get()) is not None}


@contextmanager
def bind(**values: str | None) -> Iterator[None]:
    """Set context fields for the ``with`` block and restore the previous values after."""
    unknown = set(values) - set(_VARS)
    if unknown:
        raise ValueError(f"unknown context fields: {sorted(unknown)}")
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (_VARS[name], _VARS[name].set(value)) for name, value in values.items()
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
