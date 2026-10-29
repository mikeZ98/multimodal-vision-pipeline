"""ASGI middleware bounding request bodies before any parser buffers them."""

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from multimodal_vision_pipeline.api.errors import ProblemResponse
from multimodal_vision_pipeline.schemas import ProblemDetails


class BodySizeLimitMiddleware:
    """Rejects bodies larger than ``max_body_bytes`` with ``413``.

    A declared ``Content-Length`` is checked up front, without reading the body.
    Chunked bodies are counted as they stream in; crossing the limit raises an
    ``HTTPException(413)`` from ``receive``, which aborts whichever parser is reading
    and is rendered as a problem document by the exception handlers.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            rejection = self._check_declared(declared, scope["path"])
            if rejection is not None:
                await rejection(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(413, detail=self._too_large_detail())
            return message

        await self.app(scope, limited_receive, send)

    def _too_large_detail(self) -> str:
        return f"request body exceeds {self.max_body_bytes} bytes"

    def _check_declared(self, declared: str, path: str) -> ProblemResponse | None:
        if not declared.isdigit():
            return ProblemResponse(
                ProblemDetails(
                    title="Bad Request", status=400, detail="invalid Content-Length", instance=path
                )
            )
        if int(declared) > self.max_body_bytes:
            return ProblemResponse(
                ProblemDetails(
                    title="Content Too Large",
                    status=413,
                    detail=self._too_large_detail(),
                    instance=path,
                )
            )
        return None
