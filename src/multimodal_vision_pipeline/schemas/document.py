"""Document ingestion and rasterization contracts.

Pure data validation: nothing here touches poppler, Pillow or the filesystem.
"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from multimodal_vision_pipeline.schemas._base import NormalizedCoord, PipelineModel

MIN_DPI = 72
MAX_DPI = 600
DEFAULT_DPI = 200
MAX_PAGE_LIMIT = 500
DEFAULT_MAX_PAGES = 50
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PIXELS_PER_SIDE = 30_000
"""A 50-inch side at 600 DPI; anything larger is a decompression-bomb candidate."""

Dpi = Annotated[int, Field(ge=MIN_DPI, le=MAX_DPI)]
PageNumber = Annotated[int, Field(ge=1, le=MAX_PAGE_LIMIT * 100)]
"""1-based page index into the source document."""

Filename = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255, pattern=r"^[^/\\\x00-\x1f\x7f]+$"),
]
"""Bare file name: no path separators or control characters."""

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DocumentMimeType(StrEnum):
    """Upload formats the pipeline accepts."""

    PDF = "application/pdf"
    TIFF = "image/tiff"
    PNG = "image/png"
    JPEG = "image/jpeg"

    @property
    def is_multipage(self) -> bool:
        """Whether the format can carry more than one page and must be rasterized per page."""
        return self in {DocumentMimeType.PDF, DocumentMimeType.TIFF}


class ColorMode(StrEnum):
    """Pillow image modes produced by rasterization."""

    RGB = "RGB"
    GRAYSCALE = "L"


class PageRange(PipelineModel):
    """Inclusive, 1-based page range ``[start, end]``."""

    start: PageNumber
    end: PageNumber

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.start > self.end:
            raise ValueError(f"page range start {self.start} exceeds end {self.end}")
        return self

    @property
    def page_count(self) -> int:
        return self.end - self.start + 1

    def __contains__(self, page_number: object) -> bool:
        return isinstance(page_number, int) and self.start <= page_number <= self.end


class RasterizationConfig(PipelineModel):
    """How a document is turned into page images.

    ``pages=None`` selects every page up to ``max_pages``. Explicit ranges must be
    ascending and non-overlapping, and may not select more than ``max_pages`` pages.
    """

    dpi: Dpi = DEFAULT_DPI
    color_mode: ColorMode = ColorMode.RGB
    pages: tuple[PageRange, ...] | None = Field(default=None, min_length=1, max_length=64)
    max_pages: int = Field(default=DEFAULT_MAX_PAGES, ge=1, le=MAX_PAGE_LIMIT)

    @model_validator(mode="after")
    def _check_pages(self) -> Self:
        if self.pages is None:
            return self
        for previous, current in zip(self.pages, self.pages[1:], strict=False):
            if current.start <= previous.end:
                raise ValueError(
                    f"page ranges must be ascending and non-overlapping: "
                    f"{previous.start}-{previous.end} then {current.start}-{current.end}"
                )
        selected = sum(r.page_count for r in self.pages)
        if selected > self.max_pages:
            raise ValueError(f"page selection covers {selected} pages, max_pages={self.max_pages}")
        return self

    def selects(self, page_number: int) -> bool:
        """Whether ``page_number`` falls inside the explicit selection (or no selection is set)."""
        if self.pages is None:
            return 1 <= page_number <= self.max_pages
        return any(page_number in r for r in self.pages)


class BoundingBox(PipelineModel):
    """Axis-aligned box in normalized page coordinates, ``(x0, y0)`` top-left."""

    x0: NormalizedCoord
    y0: NormalizedCoord
    x1: NormalizedCoord
    y1: NormalizedCoord

    @model_validator(mode="after")
    def _check_extent(self) -> Self:
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("bounding box must have x0 < x1 and y0 < y1")
        return self

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


class PageMetadata(PipelineModel):
    """A rasterized page: its pixel resolution and optional detected content region."""

    page_number: PageNumber
    width_px: int = Field(ge=1, le=MAX_PIXELS_PER_SIDE)
    height_px: int = Field(ge=1, le=MAX_PIXELS_PER_SIDE)
    dpi: Dpi
    color_mode: ColorMode
    content_bbox: BoundingBox | None = None


class DocumentDescriptor(PipelineModel):
    """Validated facts about an uploaded document, captured before rasterization."""

    filename: Filename
    mime_type: DocumentMimeType
    size_bytes: int = Field(ge=1, le=MAX_UPLOAD_BYTES)
    sha256: Sha256Hex
    page_count: int | None = Field(default=None, ge=1, le=MAX_PAGE_LIMIT * 100)

    @model_validator(mode="after")
    def _check_single_page_formats(self) -> Self:
        if not self.mime_type.is_multipage and self.page_count not in {None, 1}:
            raise ValueError(f"{self.mime_type} documents have exactly one page")
        return self
