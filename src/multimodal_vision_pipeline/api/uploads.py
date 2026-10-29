"""Bounded upload reading, content sniffing and filename sanitising."""

import hashlib
from dataclasses import dataclass, field
from typing import Final

from starlette.datastructures import UploadFile

from multimodal_vision_pipeline.api.errors import InvalidRequestError
from multimodal_vision_pipeline.core.errors import DocumentTooLargeError, UnsupportedDocumentError
from multimodal_vision_pipeline.schemas import DocumentMimeType, InvalidParam

READ_CHUNK_BYTES: Final = 64 * 1024
OPTIONS_MAX_BYTES: Final = 64 * 1024
MULTIPART_OVERHEAD_BYTES: Final = OPTIONS_MAX_BYTES + 16 * 1024
"""Headroom above the file limit for the options part, part headers and boundaries."""
MAX_FILENAME_CHARS: Final = 255
DEFAULT_FILENAME: Final = "upload"

_SIGNATURES: Final = (
    (b"%PDF-", DocumentMimeType.PDF),
    (b"\x89PNG\r\n\x1a\n", DocumentMimeType.PNG),
    (b"\xff\xd8\xff", DocumentMimeType.JPEG),
    (b"II*\x00", DocumentMimeType.TIFF),
    (b"MM\x00*", DocumentMimeType.TIFF),
)
_ALIASES: Final = {
    "image/jpg": DocumentMimeType.JPEG,
    "image/pjpeg": DocumentMimeType.JPEG,
    "image/tif": DocumentMimeType.TIFF,
    "application/x-pdf": DocumentMimeType.PDF,
}
_GENERIC_TYPES: Final = frozenset({"", "application/octet-stream"})
_SNIFF_BYTES: Final = 16


def sniff_mime_type(head: bytes) -> DocumentMimeType | None:
    """Supported type identified by the file's magic bytes, if any."""
    return next((mime for magic, mime in _SIGNATURES if head.startswith(magic)), None)


def resolve_mime_type(declared: str | None, head: bytes) -> DocumentMimeType:
    """Trust the content, not the label.

    A generic or missing declared type is replaced by the sniffed type. A specific
    declared type must be supported and must match the content.
    """
    sniffed = sniff_mime_type(head)
    base = (declared or "").split(";", 1)[0].strip().lower()
    if base in _GENERIC_TYPES:
        if sniffed is None:
            raise UnsupportedDocumentError("could not detect a supported document type")
        return sniffed
    try:
        declared_type = _ALIASES.get(base) or DocumentMimeType(base)
    except ValueError:
        supported = ", ".join(m.value for m in DocumentMimeType)
        raise UnsupportedDocumentError(
            f"unsupported media type {base!r}; use {supported}"
        ) from None
    if sniffed is not declared_type:
        raise UnsupportedDocumentError(f"content does not match the declared type {base}")
    return declared_type


def safe_filename(raw: str | None) -> str:
    """Basename without control characters, bounded in length; never empty."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:MAX_FILENAME_CHARS]
    return name or DEFAULT_FILENAME


@dataclass(frozen=True, slots=True)
class ReceivedDocument:
    data: bytes = field(repr=False)
    sha256: str
    filename: str
    mime_type: DocumentMimeType


async def read_upload(upload: UploadFile, *, max_bytes: int) -> ReceivedDocument:
    """Read an upload in chunks, refusing it as soon as it exceeds ``max_bytes``."""
    digest = hashlib.sha256()
    buffer = bytearray()
    while chunk := await upload.read(READ_CHUNK_BYTES):
        if len(buffer) + len(chunk) > max_bytes:
            raise DocumentTooLargeError(f"document exceeds {max_bytes} bytes")
        buffer += chunk
        digest.update(chunk)
    if not buffer:
        raise InvalidRequestError(
            "the uploaded file is empty",
            errors=(InvalidParam(pointer="/file", detail="must not be empty"),),
        )
    mime_type = resolve_mime_type(upload.content_type, bytes(buffer[:_SNIFF_BYTES]))
    return ReceivedDocument(
        data=bytes(buffer),
        sha256=digest.hexdigest(),
        filename=safe_filename(upload.filename),
        mime_type=mime_type,
    )
