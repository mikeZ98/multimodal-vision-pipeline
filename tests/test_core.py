"""Phase 3 verification: rasterization, reconciliation, vision payloads and job state.

Hermetic: images are synthesized in memory, poppler is replaced by patched
``pdf2image`` functions, and any attempt to start a subprocess fails the test.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import subprocess
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pdf2image
import pytest
from pdf2image import exceptions as pdf_errors
from PIL import Image
from pydantic import ValidationError

from multimodal_vision_pipeline.core import (
    TRANSITIONS,
    DocumentRasterizer,
    DocumentTooLargeError,
    ExtractionParseError,
    FakeRasterizer,
    IdempotencyConflictError,
    InvalidTransitionError,
    JobCapacityError,
    JobManager,
    JobNotFoundError,
    PageSelectionError,
    PayloadTooLargeError,
    PipelineError,
    RasterizationError,
    RasterizedPage,
    ReconciliationOverflowError,
    UnsupportedDocumentError,
    VisionExtraction,
    VisionPayloadBuilder,
    parse_extraction,
    plan_pages,
    reconcile,
)
from multimodal_vision_pipeline.core import rasterizer as rasterizer_module
from multimodal_vision_pipeline.core import vision_builder as builder_module
from multimodal_vision_pipeline.core.errors import PROBLEM_TYPE_PREFIX
from multimodal_vision_pipeline.core.rasterizer import (
    CORRUPT_MARKER,
    PageRasterizer,
    content_bbox,
    contiguous_chunks,
    to_color_mode,
)
from multimodal_vision_pipeline.core.vision_builder import (
    HIGH_RES_LIMITS,
    MODEL_IMAGE_LIMITS,
    STANDARD_LIMITS,
    encode_page,
    fit_dimensions,
)
from multimodal_vision_pipeline.schemas import (
    ALL_RULES,
    MAX_CENTS,
    ColorMode,
    DocumentDescriptor,
    DocumentKind,
    DocumentMimeType,
    Effort,
    ExtractionModelConfig,
    ExtractionResult,
    IngestionOptions,
    JobStatus,
    PageMetadata,
    PageRange,
    ProblemDetails,
    RasterizationConfig,
    ReconciliationPolicy,
    ReconciliationRule,
    ReconciliationStatus,
    RoundingMode,
    VisionModelId,
)
from tests.factories import descriptor_data, ef, invoice_data, missing, parse

SMALL_PAGE = (1.0, 1.5)
"""Inches; keeps synthetic pages tiny so high-DPI tests stay fast."""


@pytest.fixture(autouse=True)
def _forbid_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"subprocess spawned in a hermetic test: {args!r}")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)


async def collect(
    rasterizer: PageRasterizer,
    document: bytes = b"%PDF-fake",
    mime: DocumentMimeType = DocumentMimeType.PDF,
    config: RasterizationConfig | None = None,
) -> list[RasterizedPage]:
    return [p async for p in rasterizer.rasterize(document, mime, config or RasterizationConfig())]


def decode(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png))
    image.load()
    return image


def raster_bytes(
    fmt: str,
    size: tuple[int, int] = (300, 400),
    *,
    mode: str = "RGB",
    color: float | tuple[float, ...] | str = "white",
    frames: int = 1,
    **save: Any,
) -> bytes:
    images = [Image.new(mode, size, color) for _ in range(frames)]
    buffer = io.BytesIO()
    if frames > 1:
        images[0].save(buffer, format=fmt, save_all=True, append_images=images[1:], **save)
    else:
        images[0].save(buffer, format=fmt, **save)
    return buffer.getvalue()


# ===========================================================================
# Rasterization - planning and chunking
# ===========================================================================
def test_plan_without_selection_truncates_to_max_pages() -> None:
    assert plan_pages(600, RasterizationConfig(max_pages=50)) == tuple(range(1, 51))
    assert plan_pages(3, RasterizationConfig()) == (1, 2, 3)


def test_plan_clips_selection_to_document() -> None:
    config = RasterizationConfig(pages=(PageRange(start=2, end=3), PageRange(start=5, end=9)))
    assert plan_pages(6, config) == (2, 3, 5, 6)


def test_plan_selection_outside_document_is_an_error() -> None:
    config = RasterizationConfig(pages=(PageRange(start=10, end=12),))
    with pytest.raises(PageSelectionError, match="document has 4"):
        plan_pages(4, config)


def test_plan_requires_pages() -> None:
    with pytest.raises(RasterizationError, match="no pages"):
        plan_pages(0, RasterizationConfig())


@pytest.mark.parametrize(
    ("pages", "size", "expected"),
    [
        ([1, 2, 3, 4, 5], 2, [(1, 2), (3, 4), (5, 5)]),
        ([1, 2, 5, 6, 7], 4, [(1, 2), (5, 7)]),
        ([3], 4, [(3, 3)]),
        ([], 4, []),
        ([1, 2, 3], 1, [(1, 1), (2, 2), (3, 3)]),
    ],
)
def test_contiguous_chunks(pages: list[int], size: int, expected: list[tuple[int, int]]) -> None:
    assert contiguous_chunks(pages, size) == expected


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        contiguous_chunks([1], 0)
    with pytest.raises(ValueError, match="chunk_size"):
        FakeRasterizer(chunk_size=0)


# ===========================================================================
# Rasterization - streaming behaviour via the fake
# ===========================================================================
async def test_fake_streams_pages_in_chunks() -> None:
    fake = FakeRasterizer(page_count=5, chunk_size=2, page_size_inches=SMALL_PAGE)
    pages = await collect(fake, config=RasterizationConfig(dpi=100))
    assert [p.metadata.page_number for p in pages] == [1, 2, 3, 4, 5]
    assert fake.render_calls == [(1, 2, 100), (3, 4, 100), (5, 5, 100)]


async def test_streaming_is_lazy_one_chunk_at_a_time() -> None:
    fake = FakeRasterizer(page_count=40, chunk_size=4, page_size_inches=SMALL_PAGE)
    stream = fake.rasterize(b"x", DocumentMimeType.PDF, RasterizationConfig())
    first = await anext(stream)
    assert first.metadata.page_number == 1
    assert fake.render_calls == [(1, 4, 200)]  # nothing beyond the first chunk was rendered
    await stream.aclose()  # type: ignore[attr-defined]


async def test_huge_document_is_bounded_by_max_pages_and_chunks() -> None:
    fake = FakeRasterizer(page_count=1200, chunk_size=4, page_size_inches=(0.5, 0.5))
    pages = await collect(fake, config=RasterizationConfig(dpi=72, max_pages=500))
    assert len(pages) == 500
    assert len(fake.render_calls) == 125
    assert all(last - first < 4 for first, last, _ in fake.render_calls)


@pytest.mark.parametrize(("dpi", "size"), [(72, (72, 108)), (200, (200, 300)), (600, (600, 900))])
async def test_dpi_controls_pixel_dimensions(dpi: int, size: tuple[int, int]) -> None:
    fake = FakeRasterizer(page_count=1, page_size_inches=SMALL_PAGE)
    (page,) = await collect(fake, config=RasterizationConfig(dpi=dpi))
    assert (page.metadata.width_px, page.metadata.height_px) == size
    assert page.metadata.dpi == dpi
    assert decode(page.png).size == size


@pytest.mark.parametrize("mode", [ColorMode.RGB, ColorMode.GRAYSCALE])
async def test_color_mode_is_applied(mode: ColorMode) -> None:
    fake = FakeRasterizer(page_count=1, page_size_inches=SMALL_PAGE)
    (page,) = await collect(fake, config=RasterizationConfig(color_mode=mode))
    assert page.metadata.color_mode is mode
    assert decode(page.png).mode == mode.value
    assert page.media_type == "image/png"


async def test_page_selection_renders_only_selected_pages() -> None:
    fake = FakeRasterizer(page_count=10, chunk_size=4, page_size_inches=SMALL_PAGE)
    config = RasterizationConfig(pages=(PageRange(start=2, end=3), PageRange(start=8, end=20)))
    pages = await collect(fake, config=config)
    assert [p.metadata.page_number for p in pages] == [2, 3, 8, 9, 10]
    assert [(a, b) for a, b, _ in fake.render_calls] == [(2, 3), (8, 10)]


async def test_content_bbox_tracks_synthetic_content() -> None:
    fake = FakeRasterizer(page_count=2, page_size_inches=(4.0, 4.0))
    first, second = await collect(fake, config=RasterizationConfig(dpi=100))
    assert first.metadata.content_bbox is not None
    assert second.metadata.content_bbox is not None
    assert first.metadata.content_bbox.x0 == pytest.approx(0.225, abs=0.01)
    assert second.metadata.content_bbox.x0 == pytest.approx(0.35, abs=0.01)


def test_content_bbox_of_blank_page_is_none() -> None:
    assert content_bbox(Image.new("RGB", (50, 50), "white")) is None
    assert content_bbox(Image.new("L", (50, 50), 250)) is None  # near-white noise ignored


async def test_failure_mid_document_surfaces_after_earlier_pages() -> None:
    fake = FakeRasterizer(page_count=6, chunk_size=2, fail_on_page=4, page_size_inches=SMALL_PAGE)
    seen: list[int] = []

    async def drain() -> None:
        async for page in fake.rasterize(b"x", DocumentMimeType.PDF, RasterizationConfig()):
            seen.append(page.metadata.page_number)  # noqa: PERF401 - partial progress matters

    with pytest.raises(RasterizationError, match="page 4"):
        await drain()
    assert seen == [1, 2]


@pytest.mark.parametrize(
    ("document", "error"),
    [(b"", RasterizationError), (b"x" * 101, DocumentTooLargeError)],
)
async def test_document_size_bounds(document: bytes, error: type[Exception]) -> None:
    fake = FakeRasterizer(max_document_bytes=100)
    with pytest.raises(error):
        await collect(fake, document)


async def test_corrupt_document_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError):
        await collect(FakeRasterizer(), CORRUPT_MARKER + b"...")


async def test_single_page_formats_count_one_page() -> None:
    fake = FakeRasterizer(page_count=9, page_size_inches=SMALL_PAGE)
    assert await fake.count_pages(b"x", DocumentMimeType.PNG) == 1
    assert len(await collect(fake, mime=DocumentMimeType.JPEG)) == 1


async def test_oversized_render_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rasterizer_module, "MAX_PIXELS_PER_SIDE", 100)
    fake = FakeRasterizer(page_count=1, page_size_inches=SMALL_PAGE)
    with pytest.raises(DocumentTooLargeError, match="limit is 100px"):
        await collect(fake, config=RasterizationConfig(dpi=72))


async def test_renderer_page_count_mismatch_detected() -> None:
    class Lossy(FakeRasterizer):
        def _render_sync(self, *args: Any, **kwargs: Any) -> list[tuple[Image.Image, int]]:
            return super()._render_sync(*args, **kwargs)[:-1]

    with pytest.raises(RasterizationError, match="returned 1 page"):
        await collect(Lossy(page_count=2, page_size_inches=SMALL_PAGE))


def test_alpha_is_flattened_onto_white() -> None:
    transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_color_mode(transparent, ColorMode.RGB).getpixel((0, 0)) == (255, 255, 255)
    palette = Image.new("P", (4, 4), 0)
    palette.info["transparency"] = 0
    assert to_color_mode(palette, ColorMode.GRAYSCALE).getpixel((0, 0)) == 255
    assert (
        to_color_mode(Image.new("LA", (2, 2), (0, 255)), ColorMode.GRAYSCALE).getpixel((0, 0)) == 0
    )


# ===========================================================================
# Rasterization - production Pillow path (real codecs, in memory)
# ===========================================================================
async def test_multipage_tiff_downsampled_to_target_dpi() -> None:
    document = raster_bytes("TIFF", (300, 450), frames=3, dpi=(300, 300))
    rasterizer = DocumentRasterizer(chunk_size=2)
    assert await rasterizer.count_pages(document, DocumentMimeType.TIFF) == 3
    pages = await collect(rasterizer, document, DocumentMimeType.TIFF, RasterizationConfig(dpi=200))
    assert [p.metadata.page_number for p in pages] == [1, 2, 3]
    assert {(p.metadata.width_px, p.metadata.height_px, p.metadata.dpi) for p in pages} == {
        (200, 300, 200)
    }


@pytest.mark.parametrize(("source_dpi", "reported"), [(100, 100), (50, 72), (72, 72)])
async def test_low_dpi_raster_is_never_upsampled(source_dpi: int, reported: int) -> None:
    document = raster_bytes("TIFF", (300, 400), dpi=(source_dpi, source_dpi))
    (page,) = await collect(
        DocumentRasterizer(), document, DocumentMimeType.TIFF, RasterizationConfig(dpi=600)
    )
    assert (page.metadata.width_px, page.metadata.height_px) == (300, 400)
    assert page.metadata.dpi == reported


async def test_anisotropic_dpi_scales_each_axis() -> None:
    document = raster_bytes("TIFF", (400, 400), dpi=(400, 100))
    (page,) = await collect(
        DocumentRasterizer(), document, DocumentMimeType.TIFF, RasterizationConfig(dpi=200)
    )
    assert (page.metadata.width_px, page.metadata.height_px) == (200, 400)
    assert page.metadata.dpi == 100


async def test_raster_without_dpi_keeps_native_size() -> None:
    document = raster_bytes("PNG", (321, 123))
    (page,) = await collect(
        DocumentRasterizer(), document, DocumentMimeType.PNG, RasterizationConfig(dpi=600)
    )
    assert (page.metadata.width_px, page.metadata.height_px) == (321, 123)


async def test_jpeg_dpi_downscaled_and_grayscaled() -> None:
    document = raster_bytes("JPEG", (600, 300), color="red", dpi=(300, 300))
    config = RasterizationConfig(dpi=150, color_mode=ColorMode.GRAYSCALE)
    (page,) = await collect(DocumentRasterizer(), document, DocumentMimeType.JPEG, config)
    assert (page.metadata.width_px, page.metadata.height_px) == (300, 150)
    assert decode(page.png).mode == "L"


async def test_jpeg_exif_orientation_applied() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise for display
    document = raster_bytes("JPEG", (200, 100), exif=exif, dpi=(200, 200))
    (page,) = await collect(DocumentRasterizer(), document, DocumentMimeType.JPEG)
    assert (page.metadata.width_px, page.metadata.height_px) == (100, 200)


async def test_transparent_png_has_white_background() -> None:
    document = raster_bytes("PNG", (10, 10), mode="RGBA", color=(0, 0, 0, 0))
    (page,) = await collect(DocumentRasterizer(), document, DocumentMimeType.PNG)
    assert decode(page.png).getpixel((5, 5)) == (255, 255, 255)
    assert page.metadata.content_bbox is None


@pytest.mark.parametrize(
    ("document", "mime"),
    [
        (raster_bytes("PNG"), DocumentMimeType.JPEG),  # declared type does not match content
        (raster_bytes("JPEG"), DocumentMimeType.TIFF),
        (b"definitely not an image", DocumentMimeType.PNG),
    ],
)
async def test_content_must_match_declared_type(document: bytes, mime: DocumentMimeType) -> None:
    with pytest.raises(UnsupportedDocumentError):
        await DocumentRasterizer().count_pages(document, mime)


@pytest.mark.parametrize("keep", [300, -200])  # header cut vs. scan data cut
async def test_truncated_image_rejected(keep: int) -> None:
    rng = random.Random(7)  # noqa: S311 - noisy pixels make the scan data large
    noisy = Image.frombytes("RGB", (256, 256), rng.randbytes(256 * 256 * 3))
    buffer = io.BytesIO()
    noisy.save(buffer, format="JPEG", quality=95)
    with pytest.raises(UnsupportedDocumentError, match="corrupt"):
        await collect(DocumentRasterizer(), buffer.getvalue()[:keep], DocumentMimeType.JPEG)


def test_missing_tiff_frame_rejected() -> None:
    document = raster_bytes("TIFF", (20, 20), frames=2)
    with pytest.raises(RasterizationError, match="page 3 does not exist"):
        DocumentRasterizer()._render_sync(
            document,
            DocumentMimeType.TIFF,
            first_page=2,
            last_page=3,
            dpi=200,
            color_mode=ColorMode.RGB,
        )


async def test_decompression_bomb_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    with pytest.raises(DocumentTooLargeError):
        await DocumentRasterizer().count_pages(
            raster_bytes("PNG", (100, 100)), DocumentMimeType.PNG
        )


# ===========================================================================
# Rasterization - production PDF path with pdf2image patched (no poppler)
# ===========================================================================
class FakePoppler:
    def __init__(self, *, pages: int = 7, page_size: str | None = "612 x 792 pts (letter)") -> None:
        self.pages = pages
        self.page_size = page_size
        self.info_error: Exception | None = None
        self.convert_error: Exception | None = None
        self.convert_calls: list[dict[str, Any]] = []

    def pdfinfo_from_bytes(self, document: bytes, **kwargs: Any) -> dict[str, object]:
        if self.info_error:
            raise self.info_error
        info: dict[str, object] = {"Pages": self.pages}
        if self.page_size is not None:
            info["Page size"] = self.page_size
        return info

    def convert_from_bytes(self, document: bytes, **kwargs: Any) -> list[Image.Image]:
        if self.convert_error:
            raise self.convert_error
        self.convert_calls.append(kwargs)
        mode = "L" if kwargs["grayscale"] else "RGB"
        count = kwargs["last_page"] - kwargs["first_page"] + 1
        return [Image.new(mode, (60, 80), "white") for _ in range(count)]


@pytest.fixture
def poppler(monkeypatch: pytest.MonkeyPatch) -> FakePoppler:
    fake = FakePoppler()
    monkeypatch.setattr(pdf2image, "pdfinfo_from_bytes", fake.pdfinfo_from_bytes)
    monkeypatch.setattr(pdf2image, "convert_from_bytes", fake.convert_from_bytes)
    return fake


async def test_pdf_rendered_in_bounded_chunks(poppler: FakePoppler) -> None:
    rasterizer = DocumentRasterizer(chunk_size=3, poppler_timeout_seconds=30)
    config = RasterizationConfig(dpi=300, color_mode=ColorMode.GRAYSCALE)
    pages = await collect(rasterizer, config=config)
    assert [p.metadata.page_number for p in pages] == list(range(1, 8))
    assert [(c["first_page"], c["last_page"]) for c in poppler.convert_calls] == [
        (1, 3),
        (4, 6),
        (7, 7),
    ]
    call = poppler.convert_calls[0]
    assert call["dpi"] == 300
    assert call["grayscale"] is True
    assert call["thread_count"] == 1
    assert call["timeout"] == 30
    assert all(p.metadata.dpi == 300 for p in pages)


@pytest.mark.parametrize(
    ("page_size", "expected_dpi"),
    [
        ("20000 x 20000 pts", 108),  # 30000px * 72 / 20000pt
        ("612 x 792 pts (letter)", 600),
        (None, 600),
        ("0 x 0 pts", 600),
    ],
)
async def test_pdf_dpi_clamped_for_huge_pages(
    poppler: FakePoppler, page_size: str | None, expected_dpi: int
) -> None:
    poppler.pages = 1
    poppler.page_size = page_size
    (page,) = await collect(DocumentRasterizer(), config=RasterizationConfig(dpi=600))
    assert poppler.convert_calls[0]["dpi"] == expected_dpi
    assert page.metadata.dpi == expected_dpi


async def test_pdf_page_too_large_for_minimum_dpi(poppler: FakePoppler) -> None:
    poppler.page_size = "100000 x 10 pts"
    with pytest.raises(DocumentTooLargeError, match="cannot render"):
        await collect(DocumentRasterizer())


@pytest.mark.parametrize(
    ("stage", "error", "expected"),
    [
        ("info", pdf_errors.PDFSyntaxError("bad xref"), UnsupportedDocumentError),
        ("info", pdf_errors.PDFPageCountError("no pages"), UnsupportedDocumentError),
        ("info", pdf_errors.PDFPopplerTimeoutError("slow"), RasterizationError),
        ("convert", pdf_errors.PDFPopplerTimeoutError("slow"), RasterizationError),
        ("convert", pdf_errors.PDFSyntaxError("bad stream"), UnsupportedDocumentError),
    ],
)
async def test_pdf_errors_translated(
    poppler: FakePoppler, stage: str, error: Exception, expected: type[PipelineError]
) -> None:
    if stage == "info":
        poppler.info_error = error
    else:
        poppler.convert_error = error
    with pytest.raises(expected):
        await collect(DocumentRasterizer())


# ===========================================================================
# Reconciliation engine
# ===========================================================================
def invoice(**overrides: object) -> ExtractionResult:
    return parse(ExtractionResult, invoice_data(**overrides))


def line(quantity: str, unit: int, amount: int, rate: str | None = "23") -> dict[str, Any]:
    return {
        "description": "item",
        "quantity": quantity,
        "unit_price_cents": unit,
        "amount_cents": amount,
        "tax_rate": rate,
        "confidence": 0.9,
        "anchor": None,
    }


def tax_line(rate: str, base: int, tax: int) -> dict[str, Any]:
    return {"rate": rate, "taxable_base_cents": base, "tax_cents": tax, "confidence": 0.9}


def test_consistent_invoice_passes_every_rule() -> None:
    report = reconcile(invoice(), ReconciliationPolicy())
    assert report.status is ReconciliationStatus.PASSED
    assert report.evaluated_rules == ALL_RULES
    assert report.skipped_rules == frozenset()
    assert report.discrepancies == ()


def test_line_item_mismatch_reports_location_and_delta() -> None:
    result = invoice(line_items=[line("2", 1250, 2500), line("3", 1999, 5000)])
    report = reconcile(result, ReconciliationPolicy())
    (d, *rest) = report.discrepancies
    assert report.status is ReconciliationStatus.FAILED
    assert d.rule is ReconciliationRule.LINE_ITEM_AMOUNT
    assert (d.expected_cents, d.actual_cents, d.delta_cents) == (5997, 5000, -997)
    assert d.location == "/line_items/1/amount_cents"
    assert "3 x 1999 = 5997" in d.message
    assert [x.rule for x in rest] == [ReconciliationRule.LINE_ITEMS_SUBTOTAL]


@pytest.mark.parametrize(
    ("rounding", "tolerance", "status"),
    [
        # 1.5 x 1999 = 2998.5 cents; the invoice prints 2999.
        (RoundingMode.HALF_UP, 0, ReconciliationStatus.PASSED),
        (RoundingMode.HALF_EVEN, 0, ReconciliationStatus.FAILED),  # bankers' -> 2998
        (RoundingMode.HALF_EVEN, 1, ReconciliationStatus.PASSED),
    ],
)
def test_bankers_vs_half_up_rounding(
    rounding: RoundingMode, tolerance: int, status: ReconciliationStatus
) -> None:
    policy = ReconciliationPolicy(
        rounding=rounding,
        tolerance_cents=tolerance,
        rules=frozenset({ReconciliationRule.LINE_ITEM_AMOUNT}),
    )
    report = reconcile(invoice(), policy)
    assert report.status is status
    if status is ReconciliationStatus.FAILED:
        assert report.discrepancies[0].delta_cents == 1


@pytest.mark.parametrize(("printed", "passes"), [(6764, True), (6765, True), (6766, False)])
def test_tolerance_boundary_is_inclusive(printed: int, passes: bool) -> None:
    policy = ReconciliationPolicy(
        tolerance_cents=1, rules=frozenset({ReconciliationRule.GRAND_TOTAL})
    )
    report = reconcile(invoice(total=ef(printed)), policy)
    assert (report.status is ReconciliationStatus.PASSED) is passes


def test_zero_tolerance_demands_exact_cents() -> None:
    policy = ReconciliationPolicy(
        tolerance_cents=0, rules=frozenset({ReconciliationRule.GRAND_TOTAL})
    )
    assert reconcile(invoice(), policy).status is ReconciliationStatus.PASSED
    (d,) = reconcile(invoice(total=ef(6765)), policy).discrepancies
    assert d.delta_cents == 1


@pytest.mark.parametrize("tolerance", [-1, -100])
def test_negative_tolerance_is_rejected_by_policy(tolerance: int) -> None:
    with pytest.raises(ValidationError):
        ReconciliationPolicy(tolerance_cents=tolerance)


def test_fractional_quantities_are_exact() -> None:
    # 0.1 + 0.2 style inputs: float math would yield 300.00000000000006 cents.
    result = invoice(
        line_items=[line("0.3", 1000, 300), line("0.1", 3, 0)],
        subtotal=ef(300),
        tax_breakdown=[tax_line("23", 300, 69)],
        tax_total=ef(69),
        total=ef(369),
    )
    report = reconcile(result, ReconciliationPolicy(tolerance_cents=0))
    assert report.status is ReconciliationStatus.PASSED  # 0.1 x 3 = 0.3 -> 0 cents


def test_tax_rate_and_tax_total_mismatches() -> None:
    result = invoice(
        tax_breakdown=[tax_line("23", 5000, 1150), tax_line("8", 499, 50)],
        tax_total=ef(1210),
    )
    report = reconcile(result, ReconciliationPolicy())
    by_rule = {d.rule: d for d in report.discrepancies}
    rate = by_rule[ReconciliationRule.TAX_RATE_APPLICATION]
    assert (rate.expected_cents, rate.actual_cents, rate.location) == (
        40,
        50,
        "/tax_breakdown/1/tax_cents",
    )
    total = by_rule[ReconciliationRule.TAX_BREAKDOWN_TOTAL]
    assert (total.expected_cents, total.actual_cents) == (1200, 1210)


def test_grand_total_applies_discount() -> None:
    result = invoice(discount=ef(764), total=ef(6000))
    assert reconcile(result, ReconciliationPolicy()).status is ReconciliationStatus.PASSED


def test_missing_discount_counts_as_zero() -> None:
    report = reconcile(invoice(discount=missing()), ReconciliationPolicy())
    assert ReconciliationRule.GRAND_TOTAL in report.evaluated_rules
    assert report.status is ReconciliationStatus.PASSED


def test_illegible_values_skip_dependent_rules() -> None:
    result = invoice(
        discount=ef(None, status="illegible", confidence=0.2),
        tax_total=ef(None, status="illegible", confidence=0.2),
    )
    report = reconcile(result, ReconciliationPolicy())
    assert report.skipped_rules == {
        ReconciliationRule.GRAND_TOTAL,
        ReconciliationRule.TAX_BREAKDOWN_TOTAL,
    }
    assert report.status is ReconciliationStatus.PASSED


def test_nothing_evaluable_is_skipped() -> None:
    result = invoice(
        status="partial",
        line_items=[],
        tax_breakdown=[],
        total=missing(),
    )
    report = reconcile(result, ReconciliationPolicy())
    assert report.status is ReconciliationStatus.SKIPPED
    assert report.evaluated_rules == frozenset()


def test_policy_rule_subset_is_respected() -> None:
    rules = frozenset({ReconciliationRule.LINE_ITEMS_SUBTOTAL})
    report = reconcile(invoice(total=ef(1)), ReconciliationPolicy(rules=rules))
    assert report.evaluated_rules == rules
    assert report.status is ReconciliationStatus.PASSED  # the broken total is not checked


def test_discrepancies_are_deterministically_ordered() -> None:
    result = invoice(
        line_items=[line("1", 100, 90), line("1", 100, 80)],
        subtotal=ef(1),
        total=ef(1),
    )
    report = reconcile(result, ReconciliationPolicy())
    assert [(d.rule, d.location) for d in report.discrepancies] == [
        (ReconciliationRule.LINE_ITEM_AMOUNT, "/line_items/0/amount_cents"),
        (ReconciliationRule.LINE_ITEM_AMOUNT, "/line_items/1/amount_cents"),
        (ReconciliationRule.LINE_ITEMS_SUBTOTAL, "/subtotal/value"),
        (ReconciliationRule.GRAND_TOTAL, "/total/value"),
    ]
    assert reconcile(result, ReconciliationPolicy()) == report


def test_overflowing_expected_amount_raises() -> None:
    result = invoice(line_items=[line("999999999999", MAX_CENTS, MAX_CENTS)])
    with pytest.raises(ReconciliationOverflowError, match="/line_items/0/amount_cents"):
        reconcile(result, ReconciliationPolicy())


def test_overflowing_subtotal_sum_raises() -> None:
    result = invoice(line_items=[line("1", MAX_CENTS, MAX_CENTS)] * 2, subtotal=ef(MAX_CENTS))
    rules = frozenset({ReconciliationRule.LINE_ITEMS_SUBTOTAL})
    with pytest.raises(ReconciliationOverflowError):
        reconcile(result, ReconciliationPolicy(rules=rules))


# ===========================================================================
# Vision payload builder
# ===========================================================================
def png_page(page_number: int, size: tuple[int, int], *, noise: bool = False) -> RasterizedPage:
    if noise:
        rng = random.Random(page_number)  # noqa: S311 - test fixture data, not crypto
        image = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
    else:
        image = Image.new("RGB", size, "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    metadata = PageMetadata(
        page_number=page_number,
        width_px=size[0],
        height_px=size[1],
        dpi=200,
        color_mode=ColorMode.RGB,
    )
    return RasterizedPage(metadata=metadata, png=buffer.getvalue())


@pytest.mark.parametrize(
    ("size", "limit_edge", "limit_pixels", "expected"),
    [
        ((1000, 800), 2576, 3_750_000, (1000, 800)),  # never upscale
        ((5000, 2500), 2576, 10**9, (2576, 1288)),  # long edge bound
        ((2481, 3508), 2576, 3_750_000, (1628, 2302)),  # pixel bound (A4 @ 300 DPI)
        ((10_000, 10), 2576, 3_750_000, (2576, 2)),
        ((10, 10_000), 1568, 1_150_000, (1, 1568)),
    ],
)
def test_fit_dimensions(
    size: tuple[int, int], limit_edge: int, limit_pixels: int, expected: tuple[int, int]
) -> None:
    width, height = fit_dimensions(*size, max_long_edge=limit_edge, max_pixels=limit_pixels)
    assert (width, height) == expected
    assert max(width, height) <= limit_edge
    assert width * height <= limit_pixels


@pytest.mark.parametrize("size", [(2481, 3508), (1654, 2339), (3000, 1000)])
def test_fit_dimensions_preserves_aspect_ratio(size: tuple[int, int]) -> None:
    width, height = fit_dimensions(*size, max_long_edge=1568, max_pixels=1_150_000)
    assert abs(width / height - size[0] / size[1]) < 2 / min(width, height)


def test_fit_dimensions_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_dimensions(0, 10, max_long_edge=100, max_pixels=100)


def test_encode_page_scales_and_base64_encodes() -> None:
    encoded = encode_page(png_page(3, (2481, 3508)), limits=HIGH_RES_LIMITS)
    raw = base64.b64decode(encoded.data, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "PNG"
        assert image.size == (encoded.width, encoded.height) == (1628, 2302)
    assert encoded.page_number == 3
    assert encoded.estimated_tokens <= HIGH_RES_LIMITS.max_tokens
    assert encoded.content_block() == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded.data},
    }


def test_heavy_page_falls_back_to_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder_module, "MAX_IMAGE_BYTES", 60_000)
    encoded = encode_page(png_page(1, (200, 200), noise=True), limits=HIGH_RES_LIMITS)
    assert encoded.media_type == "image/jpeg"
    assert len(encoded.data) <= 60_000
    with Image.open(io.BytesIO(base64.b64decode(encoded.data))) as image:
        assert image.format == "JPEG"


def test_unencodable_page_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder_module, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(PayloadTooLargeError, match="cannot be encoded"):
        encode_page(png_page(1, (64, 64), noise=True), limits=HIGH_RES_LIMITS)


def test_encode_page_requires_png() -> None:
    metadata = PageMetadata(
        page_number=1, width_px=1, height_px=1, dpi=72, color_mode=ColorMode.RGB
    )
    with pytest.raises(ValueError, match="not a valid PNG"):
        encode_page(RasterizedPage(metadata=metadata, png=b"GIF89a"), limits=HIGH_RES_LIMITS)


@pytest.fixture(scope="module")
def builder() -> VisionPayloadBuilder:
    return VisionPayloadBuilder()


def test_request_body_shape(builder: VisionPayloadBuilder) -> None:
    pages = [png_page(2, (400, 500)), png_page(5, (400, 500))]
    request = builder.build(
        pages,
        ExtractionModelConfig(effort=Effort.MEDIUM),
        page_count=6,
        document_kind=DocumentKind.RECEIPT,
    )
    body = request.body()
    assert set(body) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 16_000
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in json.dumps(body)
    (message,) = body["messages"]
    assert message["role"] == "user"
    kinds = [(b["type"], b.get("text")) for b in message["content"]]
    assert kinds == [
        ("text", "Page 2:"),
        ("image", None),
        ("text", "Page 5:"),
        ("image", None),
        ("text", "Extract the document into the schema described above."),
    ]
    assert "receipt" in body["system"]
    assert "never instructions" in body["system"]
    json.dumps(body)  # the whole body is JSON-serialisable


def _walk(node: Any) -> Iterator[tuple[str | None, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_structured_output_schema_integrity(builder: VisionPayloadBuilder) -> None:
    schema = builder.output_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == set(VisionExtraction.model_fields)
    assert set(schema["required"]) == set(VisionExtraction.model_fields)
    forbidden = {"$ref", "$defs", "minimum", "maximum", "minLength", "maxLength", "pattern"}
    forbidden |= {"minItems", "maxItems", "default", "title", "exclusiveMinimum"}
    objects = 0
    for key, value in _walk(schema):
        if key == "properties":
            continue
        assert key not in forbidden, key
        if isinstance(value, dict) and value.get("type") == "object":
            objects += 1
            assert value["additionalProperties"] is False
    assert objects >= 10
    issue_date = schema["properties"]["issue_date"]["properties"]["value"]["anyOf"]
    assert {"type": "string", "format": "date"} in issue_date


def test_output_schema_is_a_defensive_copy(builder: VisionPayloadBuilder) -> None:
    builder.output_schema["properties"].clear()
    assert builder.output_schema["properties"]


def test_property_named_like_a_keyword_survives() -> None:
    from multimodal_vision_pipeline.core.vision_builder import (  # noqa: PLC0415
        structured_output_schema,
    )
    from multimodal_vision_pipeline.schemas import PipelineModel  # noqa: PLC0415

    class Tricky(PipelineModel):
        title: str
        pattern: int

    schema = structured_output_schema(Tricky)
    assert set(schema["properties"]) == {"title", "pattern"}


def test_token_budget_is_split_across_pages() -> None:
    pages = [png_page(n, (2481, 3508)) for n in (1, 2, 3)]
    request = VisionPayloadBuilder(max_total_image_tokens=6_000).build(
        pages, ExtractionModelConfig(), page_count=3
    )
    assert all(image.estimated_tokens <= 2_000 for image in request.images)
    assert request.estimated_image_tokens <= 6_000


@pytest.mark.parametrize("model", list(VisionModelId))
def test_model_specific_resolution_limits(model: VisionModelId) -> None:
    request = VisionPayloadBuilder().build(
        [png_page(1, (3000, 4000))], ExtractionModelConfig(model=model), page_count=1
    )
    (image,) = request.images
    limits = MODEL_IMAGE_LIMITS[model]
    assert max(image.width, image.height) <= limits.max_long_edge
    assert image.width * image.height <= limits.max_pixels
    assert limits in {HIGH_RES_LIMITS, STANDARD_LIMITS}


@pytest.mark.parametrize(
    ("pages", "error", "message"),
    [
        ([], PayloadTooLargeError, "no pages"),
        ([png_page(n, (8, 8)) for n in range(1, 102)], PayloadTooLargeError, "101 pages"),
        ([png_page(2, (8, 8)), png_page(1, (8, 8))], ValueError, "ascending"),
        ([png_page(1, (8, 8)), png_page(1, (8, 8))], ValueError, "unique"),
        ([png_page(102, (8, 8))], ValueError, "within page_count"),
    ],
)
def test_build_rejects_bad_page_sets(
    builder: VisionPayloadBuilder,
    pages: list[RasterizedPage],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        builder.build(pages, ExtractionModelConfig(), page_count=101)


def test_budget_too_small_for_page_count() -> None:
    with pytest.raises(PayloadTooLargeError, match="token image budget"):
        VisionPayloadBuilder(max_total_image_tokens=2).build(
            [png_page(n, (8, 8)) for n in (1, 2, 3)], ExtractionModelConfig(), page_count=3
        )
    with pytest.raises(ValueError, match="max_total_image_tokens"):
        VisionPayloadBuilder(max_total_image_tokens=0)


def test_total_request_size_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder_module, "MAX_REQUEST_IMAGE_BYTES", 100)
    with pytest.raises(PayloadTooLargeError, match="encoded pages total"):
        VisionPayloadBuilder().build([png_page(1, (64, 64))], ExtractionModelConfig(), page_count=1)


def _reply(**overrides: object) -> str:
    data = invoice_data(**overrides)
    del data["model"], data["page_count"]
    return json.dumps(data)


@pytest.fixture
def two_page_request(builder: VisionPayloadBuilder) -> Any:
    pages = [png_page(1, (40, 60)), png_page(2, (40, 60))]
    return builder.build(pages, ExtractionModelConfig(model=VisionModelId.HAIKU_5_5), page_count=4)


def test_parse_valid_reply(two_page_request: Any) -> None:
    result = parse_extraction(_reply(), two_page_request)
    assert result.model is VisionModelId.HAIKU_5_5
    assert result.page_count == 4
    assert result.total.value == 6764
    assert reconcile(result, ReconciliationPolicy()).status is ReconciliationStatus.PASSED


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        _reply(total=ef(-5)),
        _reply(currency=ef("dollars")),
        json.dumps(invoice_data()),  # service-owned fields are not accepted from the model
        _reply(line_items=[{"description": "x"}]),
    ],
)
def test_parse_rejects_invalid_replies(two_page_request: Any, raw: str) -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction(raw, two_page_request)


def test_parse_rejects_anchor_on_unsent_page(builder: VisionPayloadBuilder) -> None:
    request = builder.build([png_page(1, (40, 60))], ExtractionModelConfig(), page_count=4)
    with pytest.raises(ExtractionParseError, match="page 2, not sent"):
        parse_extraction(_reply(), request)


# ===========================================================================
# Job state machine
# ===========================================================================
class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


T0 = datetime.fromisoformat("2026-10-29T12:00:00+00:00")


def document(sha: str = "a" * 64) -> DocumentDescriptor:
    return parse(DocumentDescriptor, descriptor_data(sha256=sha))


def pages_meta(*numbers: int) -> tuple[PageMetadata, ...]:
    return tuple(
        PageMetadata(page_number=n, width_px=10, height_px=10, dpi=200, color_mode=ColorMode.RGB)
        for n in numbers
    )


def problem() -> ProblemDetails:
    return RasterizationError("boom").to_problem()


def drive_to(manager: JobManager, job_id: UUID, status: JobStatus) -> None:
    steps: list[Callable[[], object]] = [
        lambda: manager.begin_rasterizing(job_id),
        lambda: manager.begin_extracting(job_id, pages_meta(1, 2)),
        lambda: manager.begin_reconciling(job_id, invoice()),
    ]
    order = [JobStatus.QUEUED, JobStatus.RASTERIZING, JobStatus.EXTRACTING, JobStatus.RECONCILING]
    for step in steps[: order.index(status)]:
        step()


def test_happy_path_lifecycle() -> None:
    clock = Clock(T0)
    manager = JobManager(clock=clock)
    job, created = manager.submit(document(), IngestionOptions())
    assert created
    assert job.status is JobStatus.QUEUED
    assert job.created_at == job.updated_at == T0
    clock.tick(1)
    assert manager.begin_rasterizing(job.job_id).status is JobStatus.RASTERIZING
    clock.tick(1)
    extracting = manager.begin_extracting(job.job_id, pages_meta(1, 2))
    assert [p.page_number for p in extracting.pages] == [1, 2]
    clock.tick(1)
    reconciling = manager.begin_reconciling(job.job_id, invoice())
    assert reconciling.extraction is not None
    clock.tick(1)
    report = reconcile(reconciling.extraction, ReconciliationPolicy())
    done = manager.succeed(job.job_id, report)
    assert done.status is JobStatus.SUCCEEDED
    assert done.updated_at == T0 + timedelta(seconds=4)
    assert manager.get(job.job_id) == done


def test_updated_at_never_moves_backwards() -> None:
    clock = Clock(T0)
    manager = JobManager(clock=clock)
    job = manager.submit(document(), IngestionOptions()).job
    clock.tick(-60)
    assert manager.begin_rasterizing(job.job_id).updated_at == T0


def _all_pairs() -> list[tuple[JobStatus, JobStatus]]:
    return [(a, b) for a in JobStatus for b in JobStatus if b not in TRANSITIONS[a]]


def test_transition_table_is_complete() -> None:
    assert set(TRANSITIONS) == set(JobStatus)
    for status in JobStatus:
        assert (JobStatus.FAILED in TRANSITIONS[status]) is (not status.is_terminal)


@pytest.mark.parametrize(("current", "target"), _all_pairs())
def test_illegal_transitions_rejected_without_side_effects(
    current: JobStatus, target: JobStatus
) -> None:
    manager = JobManager()
    job_id = manager.submit(document(), IngestionOptions()).job.job_id
    if current is JobStatus.SUCCEEDED:
        drive_to(manager, job_id, JobStatus.RECONCILING)
        manager.succeed(job_id, reconcile(invoice(), ReconciliationPolicy()))
    elif current is JobStatus.FAILED:
        manager.fail(job_id, problem())
    else:
        drive_to(manager, job_id, current)
    before = manager.get(job_id)
    actions: dict[JobStatus, Callable[[], object]] = {
        JobStatus.QUEUED: lambda: manager._advance(job_id, JobStatus.QUEUED),
        JobStatus.RASTERIZING: lambda: manager.begin_rasterizing(job_id),
        JobStatus.EXTRACTING: lambda: manager.begin_extracting(job_id, pages_meta(1, 2)),
        JobStatus.RECONCILING: lambda: manager.begin_reconciling(job_id, invoice()),
        JobStatus.SUCCEEDED: lambda: manager.succeed(
            job_id, reconcile(invoice(), ReconciliationPolicy())
        ),
        JobStatus.FAILED: lambda: manager.fail(job_id, problem()),
    }
    with pytest.raises(InvalidTransitionError, match=f"{current} -> {target}"):
        actions[target]()
    assert manager.get(job_id) == before


@pytest.mark.parametrize("status", [s for s in JobStatus if not s.is_terminal])
def test_any_running_state_can_fail(status: JobStatus) -> None:
    manager = JobManager()
    job_id = manager.submit(document(), IngestionOptions()).job.job_id
    drive_to(manager, job_id, status)
    failed = manager.fail(job_id, problem())
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None
    assert failed.error.type == PROBLEM_TYPE_PREFIX + "rasterization-failed"


def test_invalid_payload_rejected_atomically() -> None:
    manager = JobManager()
    job_id = manager.submit(document(), IngestionOptions()).job.job_id
    drive_to(manager, job_id, JobStatus.EXTRACTING)
    stray = invoice(
        page_count=9,
        total=ef(6764, anchor_={"page_number": 9, "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1}}),
    )
    before = manager.get(job_id)
    with pytest.raises(ValidationError, match="page 9, not rasterized"):
        manager.begin_reconciling(job_id, stray)
    assert manager.get(job_id) == before


def test_unknown_job() -> None:
    manager = JobManager()
    with pytest.raises(JobNotFoundError):
        manager.get(uuid4())
    with pytest.raises(JobNotFoundError):
        manager.begin_rasterizing(uuid4())
    with pytest.raises(JobNotFoundError):
        manager.options(uuid4())


def test_options_are_retained() -> None:
    manager = JobManager()
    options = IngestionOptions(rasterization=RasterizationConfig(dpi=300))
    job_id = manager.submit(document(), options).job.job_id
    assert manager.options(job_id) == options


def test_idempotent_replay_returns_existing_job() -> None:
    manager = JobManager()
    options = IngestionOptions(idempotency_key="upload-1")
    first = manager.submit(document(), options)
    manager.begin_rasterizing(first.job.job_id)
    replay = manager.submit(document(), IngestionOptions(idempotency_key="upload-1"))
    assert not replay.created
    assert replay.job.job_id == first.job.job_id
    assert replay.job.status is JobStatus.RASTERIZING
    assert len(manager) == 1


@pytest.mark.parametrize(
    ("doc", "options"),
    [
        (document("b" * 64), IngestionOptions(idempotency_key="k")),
        (document(), IngestionOptions(idempotency_key="k", document_kind=DocumentKind.RECEIPT)),
    ],
)
def test_idempotency_key_reuse_with_different_request_conflicts(
    doc: DocumentDescriptor, options: IngestionOptions
) -> None:
    manager = JobManager()
    manager.submit(document(), IngestionOptions(idempotency_key="k"))
    with pytest.raises(IdempotencyConflictError):
        manager.submit(doc, options)


def test_without_key_every_submission_is_new() -> None:
    manager = JobManager()
    ids = {manager.submit(document(), IngestionOptions()).job.job_id for _ in range(3)}
    assert len(ids) == 3


def test_capacity_evicts_oldest_terminal_job_and_frees_its_key() -> None:
    manager = JobManager(max_jobs=2)
    old = manager.submit(document(), IngestionOptions(idempotency_key="old")).job
    manager.submit(document(), IngestionOptions())
    with pytest.raises(JobCapacityError):
        manager.submit(document(), IngestionOptions())
    manager.fail(old.job_id, problem())
    newest = manager.submit(document(), IngestionOptions()).job
    assert len(manager) == 2
    with pytest.raises(JobNotFoundError):
        manager.get(old.job_id)
    manager.fail(newest.job_id, problem())
    reused = manager.submit(document(), IngestionOptions(idempotency_key="old"))
    assert reused.created
    assert reused.job.job_id != old.job_id


def test_constructor_and_id_factory_guards() -> None:
    with pytest.raises(ValueError, match="max_jobs"):
        JobManager(max_jobs=0)
    fixed = UUID(int=1)
    manager = JobManager(id_factory=lambda: fixed)
    manager.submit(document(), IngestionOptions())
    with pytest.raises(RuntimeError, match="duplicate job id"):
        manager.submit(document(), IngestionOptions())


def test_concurrent_submissions_with_same_key_create_one_job() -> None:
    manager = JobManager()
    barrier = threading.Barrier(32)

    def submit(_: int) -> tuple[UUID, bool]:
        barrier.wait()
        job, created = manager.submit(document(), IngestionOptions(idempotency_key="race"))
        return job.job_id, created

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(submit, range(32)))
    assert sum(created for _, created in results) == 1
    assert len({job_id for job_id, _ in results}) == 1
    assert len(manager) == 1


def test_concurrent_transitions_have_exactly_one_winner() -> None:
    manager = JobManager()
    job_id = manager.submit(document(), IngestionOptions()).job.job_id
    barrier = threading.Barrier(16)

    def advance(_: int) -> bool:
        barrier.wait()
        try:
            manager.begin_rasterizing(job_id)
        except InvalidTransitionError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(advance, range(16)))
    assert outcomes.count(True) == 1
    assert manager.get(job_id).status is JobStatus.RASTERIZING


def test_concurrent_independent_jobs_progress_safely() -> None:
    manager = JobManager()

    def run(_: int) -> JobStatus:
        job_id = manager.submit(document(), IngestionOptions()).job.job_id
        drive_to(manager, job_id, JobStatus.RECONCILING)
        return manager.succeed(job_id, reconcile(invoice(), ReconciliationPolicy())).status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(run, range(64)))
    assert statuses == [JobStatus.SUCCEEDED] * 64
    assert len(manager) == 64


async def test_job_manager_usable_from_event_loop_and_threads() -> None:
    manager = JobManager()
    job_id = manager.submit(document(), IngestionOptions()).job.job_id
    await asyncio.to_thread(manager.begin_rasterizing, job_id)
    assert manager.get(job_id).status is JobStatus.RASTERIZING


# ===========================================================================
# Errors
# ===========================================================================
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (UnsupportedDocumentError("x"), 415),
        (DocumentTooLargeError("x"), 413),
        (PayloadTooLargeError("x"), 422),
        (ExtractionParseError("x"), 502),
        (JobNotFoundError("x"), 404),
        (IdempotencyConflictError("x"), 409),
        (JobCapacityError("x"), 503),
        (PipelineError(), 500),
    ],
)
def test_errors_map_to_problem_details(error: PipelineError, status: int) -> None:
    problem_ = error.to_problem(instance="/v1/jobs/1")
    assert problem_.status == status
    assert problem_.type.startswith(PROBLEM_TYPE_PREFIX)
    assert problem_.instance == "/v1/jobs/1"
    assert problem_.detail == (str(error) or None)
