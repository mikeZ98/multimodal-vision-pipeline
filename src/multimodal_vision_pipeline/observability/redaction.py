"""Sanitizer applied to everything that reaches a log line.

Removes credentials, encoded document payloads and financial / personal
identifiers. Structured values are redacted by key name as well as by content.
"""

import re
from collections.abc import Mapping
from typing import Final

REDACTED: Final = "[REDACTED]"
MAX_STRING_CHARS: Final = 4000
MAX_DEPTH: Final = 6
MAX_ITEMS: Final = 50

SENSITIVE_KEYS: Final = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "x-api-key",
        "authorization",
        "password",
        "secret",
        "token",
        "data",
        "png",
        "document",
        "content",
        "image",
        "text",
        "body",
    }
)
"""Keys whose values are dropped wholesale: credentials and document/model payloads."""

_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Credentials first, so later patterns never see a partially redacted key.
    (re.compile(r"(?i)\b(x-api-key|authorization)\s*[:=]\s*(bearer\s+)?\S+"), r"\1: " + REDACTED),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{8,}=*"), "Bearer " + REDACTED),
    # Encoded payloads: data URIs and any long base64 run (page images, PDFs).
    (re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+"), "[REDACTED_DATA_URI]"),
    (re.compile(r"[A-Za-z0-9+/]{120,}={0,2}"), "[REDACTED_BASE64]"),
    # Financial and personal identifiers.
    (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), "[REDACTED_IBAN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
)


_PAN: Final = re.compile(
    r"\b(?:\d{13,19}|\d{4}(?:[ -]\d{4}){3}(?:[ -]\d{1,4})?|\d{4}[ -]\d{6}[ -]\d{5})\b"
)
"""Card-number layouts: contiguous, 4-4-4-4(-n) and Amex 4-6-5. Confirmed by Luhn."""


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2  # noqa: PLR2004 - Luhn doubling
        total += value
    return total % 10 == 0


def _redact_pan(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_PAN]" if luhn_valid(digits) else match.group(0)


def redact_text(text: str) -> str:
    """Scrub secrets and identifiers from free text and bound its length."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _PAN.sub(_redact_pan, text)
    if len(text) > MAX_STRING_CHARS:
        text = text[:MAX_STRING_CHARS] + f"…[truncated {len(text) - MAX_STRING_CHARS} chars]"
    return text


def sanitize(value: object, *, _depth: int = 0) -> object:  # noqa: PLR0911 - one exit per kind
    """JSON-safe, redacted copy of ``value`` for structured log fields."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[REDACTED_BYTES len={len(value)}]"
    if isinstance(value, str):
        return redact_text(value)
    if _depth >= MAX_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_ITEMS:
                out["…"] = f"{len(value) - MAX_ITEMS} more"
                break
            name = str(key)
            out[name] = (
                REDACTED if name.lower() in SENSITIVE_KEYS else sanitize(item, _depth=_depth + 1)
            )
        return out
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        cleaned = [sanitize(item, _depth=_depth + 1) for item in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            cleaned.append(f"… {len(items) - MAX_ITEMS} more")
        return cleaned
    return redact_text(str(value))
