"""Correlation + access-log + HTTP metrics middleware (pure ASGI)."""

import logging
import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from multimodal_vision_pipeline.observability.context import (
    CORRELATION_HEADER,
    REQUEST_ID_HEADER,
    accept_id,
    bind,
    new_id,
)
from multimodal_vision_pipeline.observability.metrics import UNMATCHED_ROUTE, PipelineMetrics

access_logger = logging.getLogger("multimodal_vision_pipeline.access")


def route_template(scope: Scope) -> str:
    """The matched route's path template (``/api/v1/jobs/{job_id}``), never the raw path."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE


class CorrelationMiddleware:
    """Binds ``X-Correlation-ID`` / ``X-Request-ID`` for the request and echoes them.

    A valid inbound request id is kept, otherwise one is generated; the correlation
    id defaults to the request id. Invalid inbound values (whitespace, control
    characters, oversize) are replaced, never propagated into logs or headers.
    """

    def __init__(self, app: ASGIApp, *, metrics: PipelineMetrics | None = None) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = accept_id(headers.get(REQUEST_ID_HEADER)) or new_id()
        correlation_id = accept_id(headers.get(CORRELATION_HEADER)) or request_id
        status_code = 500
        start = time.perf_counter()

        async def send_with_ids(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers[CORRELATION_HEADER] = correlation_id
                response_headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        with bind(correlation_id=correlation_id, request_id=request_id):
            try:
                await self.app(scope, receive, send_with_ids)
            finally:
                elapsed = time.perf_counter() - start
                route = route_template(scope)
                if self.metrics is not None:
                    self.metrics.http_request(
                        method=scope["method"],
                        route=route,
                        status_code=status_code,
                        seconds=elapsed,
                    )
                access_logger.info(
                    "%s %s -> %d",
                    scope["method"],
                    route,
                    status_code,
                    extra={"status_code": status_code, "latency_ms": round(elapsed * 1000, 2)},
                )
