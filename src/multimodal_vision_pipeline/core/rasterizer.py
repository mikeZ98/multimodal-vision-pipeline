"""Document rasterization: PDF/TIFF/PNG/JPEG bytes in, a bounded stream of PNG pages out.

Memory is bounded by construction: pages are planned first (page count only), then
rendered ``chunk_size`` pages at a time in a worker thread, encoded to PNG and the
decoded bitmaps released before the next chunk starts. A 500+ page PDF never has
more than one chunk of bitmaps alive.
"""

import asyncio
import io
import math
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, cast

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from multimodal_vision_pipeline.core.errors import (
    DocumentTooLargeError,
    PageSelectionError,
    RasterizationError,
    UnsupportedDocumentError,
)
from multimodal_vision_pipeline.schemas import (
    MAX_DPI,
    MAX_UPLOAD_BYTES,
    MIN_DPI,
    BoundingBox,
    ColorMode,
    DocumentMimeType,
    PageMetadata,
    RasterizationConfig,
)
from multimodal_vision_pipeline.schemas.document import MAX_PIXELS_PER_SIDE

DEFAULT_CHUNK_SIZE: Final = 4
PDF_POINTS_PER_INCH: Final = 72
POPPLER_TIMEOUT_SECONDS: Final = 120
_MIN_SOURCE_DPI: Final = 36
"""Embedded DPI below this is treated as missing (aspect-only JFIF headers etc.)."""
_INK_THRESHOLD: Final = 16
"""Inverted luminance above this counts as content when locating the content box."""
_PILLOW_FORMATS: Final = {
    DocumentMimeType.TIFF: "TIFF",
    DocumentMimeType.PNG: "PNG",
    DocumentMimeType.JPEG: "JPEG",
}


@dataclass(frozen=True, slots=True)
class RasterizedPage:
    """One rendered page as lossless PNG bytes plus its metadata."""

    metadata: PageMetadata
    png: bytes = field(repr=False)
    media_type: str = "image/png"


class Rasterizer(Protocol):
    """What the pipeline needs from a rasterizer; implementations may not shell out in tests."""

    async def count_pages(self, document: bytes, mime_type: DocumentMimeType) -> int: ...

    def rasterize(
        self, document: bytes, mime_type: DocumentMimeType, config: RasterizationConfig
    ) -> AsyncIterator[RasterizedPage]: ...


def plan_pages(page_count: int, config: RasterizationConfig) -> tuple[int, ...]:
    """Pages to render, in order.

    Without an explicit selection the first ``max_pages`` pages are taken. Explicit
    ranges are clipped to the document; a selection that matches nothing is an error.
    """
    if page_count < 1:
        raise RasterizationError("document has no pages")
    if config.pages is None:
        return tuple(range(1, min(page_count, config.max_pages) + 1))
    selected = tuple(
        page for r in config.pages for page in range(r.start, min(r.end, page_count) + 1)
    )
    if not selected:
        raise PageSelectionError(f"selection matches no page; the document has {page_count}")
    return selected


def contiguous_chunks(pages: Sequence[int], chunk_size: int) -> list[tuple[int, int]]:
    """Split ascending page numbers into inclusive ``(first, last)`` runs of ≤ ``chunk_size``."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunks: list[tuple[int, int]] = []
    for page in pages:
        if chunks:
            first, last = chunks[-1]
            if page == last + 1 and page - first < chunk_size:
                chunks[-1] = (first, page)
                continue
        chunks.append((page, page))
    return chunks


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparent pixels onto white so they do not turn black."""
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image


def to_color_mode(image: Image.Image, mode: ColorMode) -> Image.Image:
    """Convert any Pillow image to ``RGB`` or ``L``, flattening alpha onto white."""
    flat = _flatten_alpha(image)
    return flat if flat.mode == mode.value else flat.convert(mode.value)


def content_bbox(image: Image.Image) -> BoundingBox | None:
    """Normalized box around non-background pixels, or ``None`` for a blank page."""
    luminance = image.convert("L")
    mask = ImageOps.invert(luminance).point([255 if v > _INK_THRESHOLD else 0 for v in range(256)])
    box = mask.getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    width, height = image.size
    return BoundingBox(x0=left / width, y0=top / height, x1=right / width, y1=bottom / height)


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def scale_to_dpi(image: Image.Image, target_dpi: int) -> tuple[Image.Image, int]:
    """Downsample a raster image from its embedded DPI to at most ``target_dpi``.

    Never upsamples: that adds no detail, only memory (a 72-DPI phone photo would
    grow ~8x in pixels at 200 DPI). Returns the image and its effective DPI, clamped
    to the schema range. Images without a usable embedded DPI are taken as-is and
    reported at ``target_dpi``.
    """
    raw = image.info.get("dpi")
    if not (isinstance(raw, tuple) and len(raw) == 2):  # noqa: PLR2004 - (x, y) pair
        return image, target_dpi
    x_dpi, y_dpi = (float(v) for v in raw)
    if x_dpi < _MIN_SOURCE_DPI or y_dpi < _MIN_SOURCE_DPI:
        return image, target_dpi
    x_scale, y_scale = min(1.0, target_dpi / x_dpi), min(1.0, target_dpi / y_dpi)
    effective = round(min(x_dpi * x_scale, y_dpi * y_scale))
    effective = max(MIN_DPI, min(MAX_DPI, effective))
    if x_scale == 1.0 and y_scale == 1.0:
        return image, effective
    width = max(1, round(image.width * x_scale))
    height = max(1, round(image.height * y_scale))
    return image.resize((width, height), Image.Resampling.LANCZOS), effective


class PageRasterizer(ABC):
    """Shared streaming, chunking, bounds checking and encoding.

    Subclasses only count pages and render an inclusive page range to Pillow
    images (paired with the DPI they were actually rendered at). Both run in a
    worker thread, so they may block.
    """

    def __init__(
        self, *, chunk_size: int = DEFAULT_CHUNK_SIZE, max_document_bytes: int = MAX_UPLOAD_BYTES
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size
        self.max_document_bytes = max_document_bytes

    @abstractmethod
    def _count_pages_sync(self, document: bytes, mime_type: DocumentMimeType) -> int: ...

    @abstractmethod
    def _render_sync(
        self,
        document: bytes,
        mime_type: DocumentMimeType,
        *,
        first_page: int,
        last_page: int,
        dpi: int,
        color_mode: ColorMode,
    ) -> list[tuple[Image.Image, int]]: ...

    def _check_document(self, document: bytes) -> None:
        if not document:
            raise RasterizationError("document is empty")
        if len(document) > self.max_document_bytes:
            raise DocumentTooLargeError(
                f"document is {len(document)} bytes; limit is {self.max_document_bytes}"
            )

    async def count_pages(self, document: bytes, mime_type: DocumentMimeType) -> int:
        self._check_document(document)
        return await asyncio.to_thread(self._count_pages_sync, document, mime_type)

    async def rasterize(
        self, document: bytes, mime_type: DocumentMimeType, config: RasterizationConfig
    ) -> AsyncIterator[RasterizedPage]:
        """Yield rendered pages in page order, one chunk of bitmaps in memory at a time."""
        plan = plan_pages(await self.count_pages(document, mime_type), config)
        for first, last in contiguous_chunks(plan, self.chunk_size):
            pages = await asyncio.to_thread(
                self._render_chunk, document, mime_type, first, last, config
            )
            for page in pages:
                yield page

    def _render_chunk(
        self,
        document: bytes,
        mime_type: DocumentMimeType,
        first: int,
        last: int,
        config: RasterizationConfig,
    ) -> list[RasterizedPage]:
        rendered = self._render_sync(
            document,
            mime_type,
            first_page=first,
            last_page=last,
            dpi=config.dpi,
            color_mode=config.color_mode,
        )
        try:
            if len(rendered) != last - first + 1:
                raise RasterizationError(
                    f"renderer returned {len(rendered)} page(s) for pages {first}-{last}"
                )
            return [
                self._finalize(image, page_number, dpi, config.color_mode)
                for page_number, (image, dpi) in zip(range(first, last + 1), rendered, strict=True)
            ]
        finally:
            for image, _ in rendered:
                image.close()

    @staticmethod
    def _finalize(
        image: Image.Image, page_number: int, dpi: int, color_mode: ColorMode
    ) -> RasterizedPage:
        if max(image.size) > MAX_PIXELS_PER_SIDE:
            raise DocumentTooLargeError(
                f"page {page_number} renders to {image.width}x{image.height}px; "
                f"limit is {MAX_PIXELS_PER_SIDE}px per side"
            )
        converted = to_color_mode(image, color_mode)
        try:
            metadata = PageMetadata(
                page_number=page_number,
                width_px=converted.width,
                height_px=converted.height,
                dpi=dpi,
                color_mode=color_mode,
                content_bbox=content_bbox(converted),
            )
            return RasterizedPage(metadata=metadata, png=encode_png(converted))
        finally:
            if converted is not image:
                converted.close()


# ---------------------------------------------------------------------------
# Production implementation
# ---------------------------------------------------------------------------
_PAGE_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*x\s*([\d.]+)\s*pts")


def _open_raster(document: bytes, mime_type: DocumentMimeType) -> Image.Image:
    """Open with only the decoder matching the declared type; bombs become errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            return Image.open(io.BytesIO(document), formats=[_PILLOW_FORMATS[mime_type]])
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise DocumentTooLargeError(str(exc)) from exc
        except UnidentifiedImageError as exc:
            raise UnsupportedDocumentError(f"content is not a valid {mime_type}") from exc
        except OSError as exc:  # header parsed but truncated or malformed
            raise UnsupportedDocumentError(f"corrupt {mime_type} data: {exc}") from exc


class DocumentRasterizer(PageRasterizer):
    """PDF via poppler (``pdf2image``); TIFF/PNG/JPEG via Pillow, in memory."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_document_bytes: int = MAX_UPLOAD_BYTES,
        poppler_timeout_seconds: int = POPPLER_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_document_bytes=max_document_bytes)
        self.poppler_timeout_seconds = poppler_timeout_seconds

    def _count_pages_sync(self, document: bytes, mime_type: DocumentMimeType) -> int:
        if mime_type is DocumentMimeType.PDF:
            return int(str(self._pdf_info(document)["Pages"]))
        with _open_raster(document, mime_type) as image:
            return int(getattr(image, "n_frames", 1)) if mime_type.is_multipage else 1

    def _render_sync(
        self,
        document: bytes,
        mime_type: DocumentMimeType,
        *,
        first_page: int,
        last_page: int,
        dpi: int,
        color_mode: ColorMode,
    ) -> list[tuple[Image.Image, int]]:
        if mime_type is DocumentMimeType.PDF:
            return self._render_pdf(document, first_page, last_page, dpi, color_mode)
        return self._render_raster(document, mime_type, first_page, last_page, dpi)

    # -- PDF ---------------------------------------------------------------
    def _pdf_info(self, document: bytes) -> dict[str, object]:
        import pdf2image  # noqa: PLC0415 - optional at import time, required for PDFs
        from pdf2image import exceptions as pdf_errors  # noqa: PLC0415

        try:
            info = pdf2image.pdfinfo_from_bytes(document, timeout=self.poppler_timeout_seconds)
        except (pdf_errors.PDFPageCountError, pdf_errors.PDFSyntaxError) as exc:
            raise UnsupportedDocumentError(f"content is not a readable PDF: {exc}") from exc
        except pdf_errors.PDFPopplerTimeoutError as exc:
            raise RasterizationError("PDF inspection timed out") from exc
        return cast("dict[str, object]", info)

    def _safe_pdf_dpi(self, document: bytes, dpi: int) -> int:
        """Lower the DPI so the first page's long side fits ``MAX_PIXELS_PER_SIDE``."""
        match = _PAGE_SIZE_RE.match(str(self._pdf_info(document).get("Page size", "")))
        if match is None:
            return dpi
        long_side_pts = max(float(match.group(1)), float(match.group(2)))
        if long_side_pts <= 0:
            return dpi
        ceiling = math.floor(MAX_PIXELS_PER_SIDE * PDF_POINTS_PER_INCH / long_side_pts)
        if ceiling < MIN_DPI:
            raise DocumentTooLargeError(
                f"page is {long_side_pts:.0f}pt long; cannot render at >= {MIN_DPI} DPI "
                f"within {MAX_PIXELS_PER_SIDE}px"
            )
        return min(dpi, ceiling, MAX_DPI)

    def _render_pdf(
        self, document: bytes, first_page: int, last_page: int, dpi: int, color_mode: ColorMode
    ) -> list[tuple[Image.Image, int]]:
        import pdf2image  # noqa: PLC0415
        from pdf2image import exceptions as pdf_errors  # noqa: PLC0415

        effective_dpi = self._safe_pdf_dpi(document, dpi)
        try:
            images = pdf2image.convert_from_bytes(
                document,
                dpi=effective_dpi,
                first_page=first_page,
                last_page=last_page,
                grayscale=color_mode is ColorMode.GRAYSCALE,
                thread_count=1,
                timeout=self.poppler_timeout_seconds,
            )
        except pdf_errors.PDFPopplerTimeoutError as exc:
            raise RasterizationError(f"rendering pages {first_page}-{last_page} timed out") from exc
        except (pdf_errors.PDFPageCountError, pdf_errors.PDFSyntaxError) as exc:
            raise UnsupportedDocumentError(f"content is not a readable PDF: {exc}") from exc
        return [(image, effective_dpi) for image in images]

    # -- TIFF / PNG / JPEG ---------------------------------------------------
    @staticmethod
    def _render_raster(
        document: bytes, mime_type: DocumentMimeType, first_page: int, last_page: int, dpi: int
    ) -> list[tuple[Image.Image, int]]:
        rendered: list[tuple[Image.Image, int]] = []
        try:
            with _open_raster(document, mime_type) as source:
                for page_number in range(first_page, last_page + 1):
                    try:
                        source.seek(page_number - 1)
                        frame = ImageOps.exif_transpose(source)  # always a detached copy
                        scaled, effective_dpi = scale_to_dpi(frame, dpi)
                    except EOFError as exc:
                        raise RasterizationError(f"page {page_number} does not exist") from exc
                    except OSError as exc:
                        raise UnsupportedDocumentError(f"corrupt {mime_type} data: {exc}") from exc
                    if scaled is not frame:
                        frame.close()
                    rendered.append((scaled, effective_dpi))
        except BaseException:
            for image, _ in rendered:
                image.close()
            raise
        return rendered


# ---------------------------------------------------------------------------
# Hermetic fake
# ---------------------------------------------------------------------------
A4_INCHES: Final = (8.27, 11.69)
CORRUPT_MARKER: Final = b"%CORRUPT"
"""Documents starting with this are rejected by the fake, to exercise error paths."""


class FakeRasterizer(PageRasterizer):
    """Synthesizes real Pillow pages for any input bytes; no codecs or binaries involved.

    Page ``n`` is white with a black block whose position depends on ``n``, so the
    detected content box is deterministic. ``render_calls`` records each chunk.
    """

    def __init__(
        self,
        *,
        page_count: int = 3,
        page_size_inches: tuple[float, float] = A4_INCHES,
        fail_on_page: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_document_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_document_bytes=max_document_bytes)
        self.page_count = page_count
        self.page_size_inches = page_size_inches
        self.fail_on_page = fail_on_page
        self.render_calls: list[tuple[int, int, int]] = []

    def _count_pages_sync(self, document: bytes, mime_type: DocumentMimeType) -> int:
        if document.startswith(CORRUPT_MARKER):
            raise UnsupportedDocumentError(f"content is not a valid {mime_type}")
        return self.page_count if mime_type.is_multipage else 1

    def _render_sync(
        self,
        document: bytes,  # noqa: ARG002 - pages are synthesized, input is ignored
        mime_type: DocumentMimeType,  # noqa: ARG002
        *,
        first_page: int,
        last_page: int,
        dpi: int,
        color_mode: ColorMode,
    ) -> list[tuple[Image.Image, int]]:
        self.render_calls.append((first_page, last_page, dpi))
        if self.fail_on_page is not None and first_page <= self.fail_on_page <= last_page:
            raise RasterizationError(f"simulated failure on page {self.fail_on_page}")
        width = max(1, round(self.page_size_inches[0] * dpi))
        height = max(1, round(self.page_size_inches[1] * dpi))
        pages: list[tuple[Image.Image, int]] = []
        for page_number in range(first_page, last_page + 1):
            image = Image.new(color_mode.value, (width, height), "white")
            offset = (page_number % 4) / 8
            box = (
                round(width * (0.1 + offset)),
                round(height * 0.1),
                round(width * (0.4 + offset)),
                round(height * 0.3),
            )
            ImageDraw.Draw(image).rectangle(box, fill="black")
            pages.append((image, dpi))
        return pages
