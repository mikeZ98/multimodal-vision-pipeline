"""Resilient async client for the Messages API, over raw ``httpx``.

Retries 408/409/429/5xx/529 and transport failures with decorrelated-jitter
backoff, honouring ``retry-after`` and ``x-should-retry``. Cancellation is never
swallowed: an ``asyncio.CancelledError`` aborts the in-flight request or the
backoff sleep and propagates. Upstream error envelopes become ``UpstreamError``
subclasses that map to RFC 9457 problems without leaking upstream details.
"""

import asyncio
import json
import logging
import random
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final, Self

import httpx
from pydantic import SecretStr

from multimodal_vision_pipeline.core.errors import (
    ExtractionParseError,
    PipelineError,
    UpstreamError,
    UpstreamOverloadedError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    VisionRefusalError,
)
from multimodal_vision_pipeline.core.vision_builder import VisionRequest
from multimodal_vision_pipeline.schemas import VisionModelId, VisionSettings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: Final = "https://api.anthropic.com"
MESSAGES_PATH: Final = "/v1/messages"
API_VERSION: Final = "2026-10-21"
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
FALLBACK_MODELS: Final = frozenset({VisionModelId.OPUS_5_5, VisionModelId.SONNET_5_5})
"""Models with server-side refusal fallback; the Haiku tier has none."""

RETRYABLE_STATUSES: Final = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
CONNECT_TIMEOUT_SECONDS: Final = 10.0
WRITE_TIMEOUT_SECONDS: Final = 60.0
POOL_TIMEOUT_SECONDS: Final = 10.0
_MAX_DETAIL_CHARS: Final = 300

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Decorrelated jitter: ``delay = min(cap, uniform(base, previous * 3))``.

    A server-provided ``retry-after`` raises the delay to at least that value, still
    capped by ``max_retry_after`` so one response cannot stall a job indefinitely.
    """

    base_delay: float = 0.5
    max_delay: float = 20.0
    max_retry_after: float = 60.0

    def __post_init__(self) -> None:
        if not 0 < self.base_delay <= self.max_delay:
            raise ValueError("require 0 < base_delay <= max_delay")
        if self.max_retry_after < 0:
            raise ValueError("max_retry_after must be >= 0")

    def next_delay(self, previous: float, rng: random.Random) -> float:
        upper = max(self.base_delay, previous * 3)
        return min(self.max_delay, rng.uniform(self.base_delay, upper))

    def apply_retry_after(self, delay: float, retry_after: float | None) -> float:
        if retry_after is None:
            return delay
        return min(max(delay, retry_after), max(self.max_retry_after, delay))


@dataclass(frozen=True, slots=True)
class VisionResponse:
    """The extraction text plus what is worth logging about how it was produced."""

    text: str = field(repr=False)
    model: str
    stop_reason: str
    request_id: str | None
    attempts: int
    fallback_used: bool = False
    usage: Mapping[str, Any] = field(default_factory=dict)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds from a ``retry-after`` header (delta-seconds or HTTP-date)."""
    if value is None or not value.strip():
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1] + "…"


_ERROR_TYPES: Final[Mapping[str, type[UpstreamError]]] = {
    "rate_limit_error": UpstreamRateLimitedError,
    "overloaded_error": UpstreamOverloadedError,
    "api_error": UpstreamError,
    "timeout_error": UpstreamTimeoutError,
    "invalid_request_error": UpstreamRejectedError,
    "authentication_error": UpstreamRejectedError,
    "permission_error": UpstreamRejectedError,
    "billing_error": UpstreamRejectedError,
    "not_found_error": UpstreamRejectedError,
    "request_too_large": UpstreamRejectedError,
}
_STATUS_FALLBACK: Final[Mapping[int, type[UpstreamError]]] = {
    408: UpstreamTimeoutError,
    429: UpstreamRateLimitedError,
    504: UpstreamTimeoutError,
    529: UpstreamOverloadedError,
}


def error_from_response(response: httpx.Response) -> UpstreamError:
    """Map an error response (Anthropic envelope or not) to an ``UpstreamError``."""
    error_type: str | None = None
    message = f"HTTP {response.status_code}"
    request_id = response.headers.get("request-id")
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        envelope = payload.get("error")
        if isinstance(envelope, dict):
            error_type = envelope.get("type") if isinstance(envelope.get("type"), str) else None
            if isinstance(envelope.get("message"), str):
                message = f"{error_type or 'error'}: {envelope['message']}"
        if isinstance(payload.get("request_id"), str):
            request_id = request_id or payload["request_id"]
    if error_type is not None and error_type in _ERROR_TYPES:
        cls = _ERROR_TYPES[error_type]
    elif response.status_code in _STATUS_FALLBACK:
        cls = _STATUS_FALLBACK[response.status_code]
    elif 400 <= response.status_code < 500:  # noqa: PLR2004 - HTTP client error class
        cls = UpstreamRejectedError
    else:
        cls = UpstreamError
    return cls(
        _truncate(message),
        error_type=error_type,
        upstream_status=response.status_code,
        request_id=request_id,
    )


def _should_retry(response: httpx.Response) -> bool:
    hint = response.headers.get("x-should-retry")
    if hint == "true":
        return True
    if hint == "false":
        return False
    return response.status_code in RETRYABLE_STATUSES


def parse_message(payload: object, *, request_id: str | None, attempts: int) -> VisionResponse:
    """Extract the structured-output text from a successful Messages API body."""
    if not isinstance(payload, dict) or payload.get("type") != "message":
        raise UpstreamProtocolError("response is not a message object", request_id=request_id)
    stop_reason = payload.get("stop_reason")
    content = payload.get("content")
    if not isinstance(content, list):
        raise UpstreamProtocolError("message has no content list", request_id=request_id)
    if stop_reason == "refusal":
        details = payload.get("stop_details")
        category = details.get("category") if isinstance(details, dict) else None
        raise VisionRefusalError(f"refusal category: {category or 'unspecified'}")
    if stop_reason == "max_tokens":
        raise ExtractionParseError("output truncated at max_tokens; raise the token limit")
    texts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if not texts:
        raise UpstreamProtocolError("message contains no text block", request_id=request_id)
    usage = payload.get("usage")
    return VisionResponse(
        text=texts[0],
        model=str(payload.get("model", "")),
        stop_reason=str(stop_reason),
        request_id=request_id,
        attempts=attempts,
        fallback_used=any(isinstance(b, dict) and b.get("type") == "fallback" for b in content),
        usage=usage if isinstance(usage, dict) else {},
    )


class VisionClient:
    """Sends ``VisionRequest`` bodies to the Messages API with bounded retries."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str = DEFAULT_BASE_URL,
        retry: RetryPolicy | None = None,
        server_side_fallback: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("api_key is empty")
        self._api_key = api_key
        self._retry = retry or RetryPolicy()
        self._server_side_fallback = server_side_fallback
        self._sleep = sleep
        self._rng = rng or secrets.SystemRandom()
        self._http = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            headers={"anthropic-version": API_VERSION, "content-type": "application/json"},
        )

    @classmethod
    def from_settings(cls, settings: VisionSettings, **kwargs: Any) -> Self:
        if settings.anthropic_api_key is None:
            raise ValueError("MVP_ANTHROPIC_API_KEY is not set")
        return cls(
            api_key=settings.anthropic_api_key,
            base_url=str(settings.anthropic_base_url).rstrip("/"),
            server_side_fallback=settings.vision_server_side_fallback,
            **kwargs,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def request_body(self, request: VisionRequest) -> dict[str, Any]:
        body = request.body()
        if self._uses_fallback(request):
            body["fallbacks"] = "default"
        return body

    def _uses_fallback(self, request: VisionRequest) -> bool:
        return self._server_side_fallback and request.config.model in FALLBACK_MODELS

    def _headers(self, request: VisionRequest) -> dict[str, str]:
        headers = {"x-api-key": self._api_key.get_secret_value()}
        if self._uses_fallback(request):
            headers["anthropic-beta"] = FALLBACK_BETA
        return headers

    @staticmethod
    def _timeout(request: VisionRequest) -> httpx.Timeout:
        return httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=request.config.request_timeout_seconds,
            write=WRITE_TIMEOUT_SECONDS,
            pool=POOL_TIMEOUT_SECONDS,
        )

    async def extract(self, request: VisionRequest) -> VisionResponse:
        """POST the request, retrying transient failures up to ``config.max_retries`` times."""
        body = json.dumps(self.request_body(request), separators=(",", ":")).encode()
        headers = self._headers(request)
        timeout = self._timeout(request)
        max_attempts = request.config.max_retries + 1
        delay = self._retry.base_delay
        for attempt in range(1, max_attempts + 1):
            failure: PipelineError
            retry_after: float | None = None
            try:
                response = await self._http.post(
                    MESSAGES_PATH, content=body, headers=headers, timeout=timeout
                )
            except httpx.TimeoutException as exc:
                failure = UpstreamTimeoutError(f"{type(exc).__name__} on attempt {attempt}")
                retryable = True
            except httpx.TransportError as exc:
                failure = UpstreamUnavailableError(f"{type(exc).__name__} on attempt {attempt}")
                retryable = True
            else:
                request_id = response.headers.get("request-id")
                if response.is_success:
                    try:
                        payload = response.json()
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise UpstreamProtocolError(
                            "response body is not JSON", request_id=request_id
                        ) from exc
                    return parse_message(payload, request_id=request_id, attempts=attempt)
                failure = error_from_response(response)
                retryable = _should_retry(response)
                retry_after = parse_retry_after(response.headers.get("retry-after"))
            if not retryable or attempt == max_attempts:
                raise failure
            delay = self._retry.apply_retry_after(
                self._retry.next_delay(delay, self._rng), retry_after
            )
            logger.warning(
                "vision request attempt %d/%d failed (%s); retrying in %.2fs",
                attempt,
                max_attempts,
                failure.slug,
                delay,
            )
            await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
