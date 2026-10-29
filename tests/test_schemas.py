"""Phase 2 verification: strict domain schemas, wire contracts and reconciliation math.

Hermetic: pure in-memory validation, no files, network or subprocesses.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from multimodal_vision_pipeline import schemas
from multimodal_vision_pipeline.schemas import (
    ALL_RULES,
    DEFAULT_VISION_MODEL,
    MAX_CENTS,
    MAX_UPLOAD_BYTES,
    PROBLEM_JSON_MEDIA_TYPE,
    BoundingBox,
    ColorMode,
    Discrepancy,
    DocumentDescriptor,
    DocumentMimeType,
    Effort,
    ExtractedField,
    ExtractionModelConfig,
    ExtractionResult,
    ExtractionStatus,
    FieldAnchor,
    FieldStatus,
    IngestionAccepted,
    IngestionOptions,
    JobStatus,
    JobStatusResponse,
    LineItem,
    PageMetadata,
    PageRange,
    PipelineModel,
    ProblemDetails,
    RasterizationConfig,
    ReconciliationPolicy,
    ReconciliationReport,
    ReconciliationRule,
    ReconciliationStatus,
    RoundingMode,
    VisionModelId,
    VisionSettings,
)

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
JOB_ID = UUID("6f1c2a52-5d0e-4c3e-9b7a-1f2e3d4c5b6a")
SHA256 = "a" * 64


def parse[M: PipelineModel](model: type[M], data: object) -> M:
    """Validate ``data`` the way it arrives over the wire (JSON text)."""
    return model.model_validate_json(json.dumps(data))


def anchor(page: int = 1, bbox: tuple[float, ...] = (0.1, 0.1, 0.5, 0.2)) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    return {"page_number": page, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


def ef(
    value: object,
    *,
    status: str = "extracted",
    confidence: float = 0.95,
    anchor_: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"value": value, "status": status, "confidence": confidence, "anchor": anchor_}


def missing() -> dict[str, Any]:
    return ef(None, status="missing", confidence=0.0)


def invoice_data(**overrides: object) -> dict[str, Any]:
    """A self-consistent two-page invoice.

    2 x 12.50 = 25.00; 1.5 x 19.99 = 29.985 -> 29.99 (half-up); subtotal 54.99;
    23% of 54.99 = 12.6477 -> 12.65; total 54.99 + 12.65 - 0 = 67.64.
    """
    data: dict[str, Any] = {
        "document_kind": "invoice",
        "model": "claude-sonnet-5-5",
        "status": "complete",
        "page_count": 2,
        "vendor": {
            "name": ef("ACME Sp. z o.o.", anchor_=anchor(1)),
            "tax_id": ef("PL5250001009"),
            "address": ef("ul. Prosta 1, 00-001 Warszawa"),
        },
        "document_number": ef("FV/2026/10/0042"),
        "issue_date": ef("2026-10-29"),
        "currency": ef("PLN", status="inferred", confidence=0.8),
        "line_items": [
            {
                "description": "Widget",
                "quantity": "2",
                "unit_price_cents": 1250,
                "amount_cents": 2500,
                "tax_rate": "23",
                "confidence": 0.97,
                "anchor": anchor(1, (0.05, 0.4, 0.95, 0.45)),
            },
            {
                "description": "Copper wire (kg)",
                "quantity": "1.5",
                "unit_price_cents": 1999,
                "amount_cents": 2999,
                "tax_rate": "23",
                "confidence": 0.91,
                "anchor": anchor(2, (0.05, 0.1, 0.95, 0.15)),
            },
        ],
        "tax_breakdown": [
            {"rate": "23", "taxable_base_cents": 5499, "tax_cents": 1265, "confidence": 0.9}
        ],
        "subtotal": ef(5499),
        "tax_total": ef(1265),
        "discount": ef(0, status="inferred", confidence=0.6),
        "total": ef(6764, anchor_=anchor(2, (0.6, 0.8, 0.95, 0.85))),
    }
    data.update(overrides)
    return data


def report_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": "passed",
        "policy": {"tolerance_cents": 1, "rounding": "half_up", "rules": sorted(ALL_RULES)},
        "evaluated_rules": sorted(ALL_RULES),
        "skipped_rules": [],
        "discrepancies": [],
    }
    data.update(overrides)
    return data


def descriptor_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "filename": "invoice.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 48_213,
        "sha256": SHA256,
        "page_count": 2,
    }
    data.update(overrides)
    return data


def page_data(page: int) -> dict[str, Any]:
    return {
        "page_number": page,
        "width_px": 1654,
        "height_px": 2339,
        "dpi": 200,
        "color_mode": "RGB",
    }


def job_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "job_id": str(JOB_ID),
        "status": "succeeded",
        "document": descriptor_data(),
        "created_at": NOW.isoformat(),
        "updated_at": (NOW + timedelta(seconds=9)).isoformat(),
        "pages": [page_data(1), page_data(2)],
        "extraction": invoice_data(),
        "reconciliation": report_data(),
        "error": None,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Base model behaviour
# ---------------------------------------------------------------------------
def test_public_exports_resolve() -> None:
    for name in schemas.__all__:
        assert hasattr(schemas, name), name


def test_python_mode_is_strict_no_coercion() -> None:
    with pytest.raises(ValidationError):
        PageRange(start="1", end=2)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LineItem(
            description="x",
            quantity=Decimal(1),
            unit_price_cents=10.0,  # type: ignore[arg-type]
            amount_cents=10,
            confidence=0.9,
        )


def test_bool_is_not_an_amount() -> None:
    with pytest.raises(ValidationError, match=r"int_type|valid integer"):
        parse(ExtractedField[int], ef(True))
    with pytest.raises(ValidationError):
        parse(ExtractionResult, invoice_data(total=ef(True)))


def test_models_are_frozen() -> None:
    page_range = PageRange(start=1, end=2)
    with pytest.raises(ValidationError, match="frozen"):
        page_range.start = 5  # type: ignore[misc]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError, match="extra"):
        parse(PageRange, {"start": 1, "end": 2, "step": 1})


def test_utc_datetime_normalised_and_naive_rejected() -> None:
    accepted = parse(
        IngestionAccepted,
        {
            "job_id": str(JOB_ID),
            "status": "queued",
            "status_url": f"/v1/jobs/{JOB_ID}",
            "created_at": "2026-10-29T14:00:00+02:00",
        },
    )
    assert accepted.created_at == NOW
    assert accepted.created_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="timezone"):
        parse(
            IngestionAccepted,
            {
                "job_id": str(JOB_ID),
                "status": "queued",
                "status_url": f"/v1/jobs/{JOB_ID}",
                "created_at": "2026-10-29T12:00:00",
            },
        )


# ---------------------------------------------------------------------------
# Documents & rasterization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mime", ["application/pdf", "image/tiff", "image/png", "image/jpeg"])
def test_supported_mime_types(mime: str) -> None:
    assert (
        parse(DocumentDescriptor, descriptor_data(mime_type=mime, page_count=1)).mime_type == mime
    )


@pytest.mark.parametrize(
    "mime",
    ["application/zip", "image/gif", "text/plain", "APPLICATION/PDF", "image/jpg", "", "image/*"],
)
def test_unsupported_mime_types_rejected(mime: str) -> None:
    with pytest.raises(ValidationError):
        parse(DocumentDescriptor, descriptor_data(mime_type=mime))


def test_multipage_formats() -> None:
    assert {m for m in DocumentMimeType if m.is_multipage} == {
        DocumentMimeType.PDF,
        DocumentMimeType.TIFF,
    }


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
def test_single_page_formats_cannot_claim_many_pages(mime: str) -> None:
    with pytest.raises(ValidationError, match="exactly one page"):
        parse(DocumentDescriptor, descriptor_data(mime_type=mime, page_count=3))


@pytest.mark.parametrize(
    "filename", ["../etc/passwd", "a/b.pdf", "a\\b.pdf", "bad\x00.pdf", "tab\t.pdf", "", "x" * 256]
)
def test_unsafe_filenames_rejected(filename: str) -> None:
    with pytest.raises(ValidationError):
        parse(DocumentDescriptor, descriptor_data(filename=filename))


@pytest.mark.parametrize("size", [0, -1, MAX_UPLOAD_BYTES + 1])
def test_upload_size_bounds(size: int) -> None:
    with pytest.raises(ValidationError):
        parse(DocumentDescriptor, descriptor_data(size_bytes=size))


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_sha256_must_be_lowercase_hex(digest: str) -> None:
    with pytest.raises(ValidationError):
        parse(DocumentDescriptor, descriptor_data(sha256=digest))


@pytest.mark.parametrize("dpi", [72, 150, 300, 600])
def test_dpi_within_bounds(dpi: int) -> None:
    assert parse(RasterizationConfig, {"dpi": dpi}).dpi == dpi


@pytest.mark.parametrize("dpi", [71, 601, 0, -300, 1200])
def test_dpi_out_of_bounds_rejected(dpi: int) -> None:
    with pytest.raises(ValidationError):
        parse(RasterizationConfig, {"dpi": dpi})


def test_rasterization_defaults() -> None:
    config = RasterizationConfig()
    assert (config.dpi, config.color_mode, config.pages, config.max_pages) == (
        200,
        ColorMode.RGB,
        None,
        50,
    )
    assert config.selects(1)
    assert config.selects(50)
    assert not config.selects(51)
    assert not config.selects(0)


@pytest.mark.parametrize(("mode", "expected"), [("RGB", ColorMode.RGB), ("L", ColorMode.GRAYSCALE)])
def test_color_modes(mode: str, expected: ColorMode) -> None:
    assert parse(RasterizationConfig, {"color_mode": mode}).color_mode is expected


@pytest.mark.parametrize("mode", ["CMYK", "rgb", "1", "RGBA"])
def test_unsupported_color_modes_rejected(mode: str) -> None:
    with pytest.raises(ValidationError):
        parse(RasterizationConfig, {"color_mode": mode})


@pytest.mark.parametrize("max_pages", [0, -1, 501])
def test_max_pages_bounds(max_pages: int) -> None:
    with pytest.raises(ValidationError):
        parse(RasterizationConfig, {"max_pages": max_pages})


def test_page_range_semantics() -> None:
    page_range = PageRange(start=3, end=5)
    assert page_range.page_count == 3
    assert 3 in page_range
    assert 5 in page_range
    assert 6 not in page_range
    assert "4" not in page_range
    assert PageRange(start=7, end=7).page_count == 1


@pytest.mark.parametrize(("start", "end"), [(5, 3), (0, 2), (-1, 1), (1, 0)])
def test_illegal_page_ranges_rejected(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        parse(PageRange, {"start": start, "end": end})


def test_page_selection() -> None:
    config = parse(
        RasterizationConfig,
        {"pages": [{"start": 1, "end": 2}, {"start": 3, "end": 3}, {"start": 10, "end": 12}]},
    )
    assert [p for p in range(1, 14) if config.selects(p)] == [1, 2, 3, 10, 11, 12]


@pytest.mark.parametrize(
    "pages",
    [
        [{"start": 1, "end": 5}, {"start": 5, "end": 6}],
        [{"start": 1, "end": 5}, {"start": 3, "end": 4}],
        [{"start": 7, "end": 8}, {"start": 1, "end": 2}],
    ],
)
def test_overlapping_or_unordered_ranges_rejected(pages: list[dict[str, int]]) -> None:
    with pytest.raises(ValidationError, match="ascending and non-overlapping"):
        parse(RasterizationConfig, {"pages": pages})


def test_empty_page_selection_rejected() -> None:
    with pytest.raises(ValidationError):
        parse(RasterizationConfig, {"pages": []})


def test_selection_cannot_exceed_max_pages() -> None:
    parse(RasterizationConfig, {"pages": [{"start": 1, "end": 10}], "max_pages": 10})
    with pytest.raises(ValidationError, match="covers 11 pages"):
        parse(RasterizationConfig, {"pages": [{"start": 1, "end": 11}], "max_pages": 10})


def test_bounding_box_valid() -> None:
    box = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=0.5)
    assert box.area == pytest.approx(0.5)


@pytest.mark.parametrize(
    "coords",
    [
        (-0.01, 0.1, 0.5, 0.5),
        (0.1, 0.1, 1.01, 0.5),
        (0.1, 0.1, 0.5, 2.0),
        (10.0, 20.0, 300.0, 400.0),
        (0.1, -1e-9, 0.5, 0.5),
    ],
)
def test_unnormalized_bbox_rejected(coords: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError):
        parse(FieldAnchor, anchor(1, coords))


@pytest.mark.parametrize(
    "raw", ['{"x0": NaN, "y0": 0, "x1": 1, "y1": 1}', '{"x0": 0, "y0": 0, "x1": Infinity, "y1": 1}']
)
def test_non_finite_bbox_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        BoundingBox.model_validate_json(raw)


@pytest.mark.parametrize(
    "coords", [(0.5, 0.1, 0.5, 0.2), (0.1, 0.3, 0.5, 0.3), (0.6, 0.1, 0.5, 0.2)]
)
def test_degenerate_or_inverted_bbox_rejected(coords: tuple[float, ...]) -> None:
    x0, y0, x1, y1 = coords
    with pytest.raises(ValidationError, match="x0 < x1 and y0 < y1"):
        BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def test_page_metadata() -> None:
    page = parse(PageMetadata, {**page_data(1), "content_bbox": anchor()["bbox"]})
    assert page.content_bbox is not None
    for bad in ({"width_px": 0}, {"height_px": 30_001}, {"dpi": 50}, {"page_number": 0}):
        with pytest.raises(ValidationError):
            parse(PageMetadata, {**page_data(1), **bad})


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def test_model_config_defaults() -> None:
    config = ExtractionModelConfig()
    assert config.model is DEFAULT_VISION_MODEL is VisionModelId.SONNET_5_5
    assert config.effort is Effort.HIGH
    assert config.max_tokens == 16_000


def test_model_ids_are_exact_undated_aliases() -> None:
    assert {m.value for m in VisionModelId} == {
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    }


@pytest.mark.parametrize(
    "model",
    ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "gpt-4o", "claude-opus-5-5-20260401"],
)
def test_retired_or_foreign_models_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        parse(ExtractionModelConfig, {"model": model})


def test_sampling_parameters_not_accepted() -> None:
    with pytest.raises(ValidationError, match="extra"):
        parse(ExtractionModelConfig, {"temperature": 0.0})


@pytest.mark.parametrize(
    "override",
    [
        {"max_tokens": 0},
        {"max_tokens": 128_001},
        {"effort": "extreme"},
        {"request_timeout_seconds": 0},
        {"max_retries": 11},
    ],
)
def test_model_config_bounds(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse(ExtractionModelConfig, override)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:

    for key in list(os.environ):
        if key.startswith("MVP_"):
            monkeypatch.delenv(key)
    return monkeypatch


def test_settings_defaults(clean_env: pytest.MonkeyPatch) -> None:
    settings = VisionSettings()
    assert settings.extraction_config() == ExtractionModelConfig()
    assert settings.anthropic_api_key is None


def test_settings_from_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MVP_VISION_MODEL", "claude-opus-5-5")
    clean_env.setenv("MVP_VISION_MAX_TOKENS", "32000")
    clean_env.setenv("MVP_VISION_EFFORT", "xhigh")
    clean_env.setenv("MVP_ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    settings = VisionSettings()
    config = settings.extraction_config()
    assert config.model is VisionModelId.OPUS_5_5
    assert config.max_tokens == 32_000
    assert config.effort is Effort.XHIGH
    assert "sk-test" not in repr(settings)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-not-a-real-key"


@pytest.mark.parametrize(
    ("key", "value"),
    [("MVP_VISION_MODEL", "claude-3-5-sonnet-20241022"), ("MVP_VISION_MAX_TOKENS", "0")],
)
def test_settings_reject_invalid_environment(
    clean_env: pytest.MonkeyPatch, key: str, value: str
) -> None:
    clean_env.setenv(key, value)
    with pytest.raises(ValidationError):
        VisionSettings()


# ---------------------------------------------------------------------------
# Extraction results
# ---------------------------------------------------------------------------
def test_invoice_round_trip() -> None:
    result = parse(ExtractionResult, invoice_data())
    assert result.line_items[1].quantity == Decimal("1.5")
    assert result.issue_date.value == date(2026, 10, 29)
    assert result.total.value == 6764
    assert ExtractionResult.model_validate_json(result.model_dump_json()) == result


def test_decimals_serialise_as_strings() -> None:
    dumped = json.loads(parse(ExtractionResult, invoice_data()).model_dump_json())
    assert dumped["line_items"][1]["quantity"] == "1.5"
    assert dumped["tax_breakdown"][0]["rate"] == "23"


def test_json_float_quantity_is_parsed_exactly() -> None:
    item = parse(ExtractionResult, invoice_data()).line_items[0].model_dump(mode="json")
    item["quantity"] = 0.1
    raw = json.dumps(item)
    assert LineItem.model_validate_json(raw).quantity == Decimal("0.1")


def test_float_constructed_decimal_rejected() -> None:
    # Decimal(0.1) carries the binary float error: 0.1000000000000000055511151231257827...
    with pytest.raises(ValidationError, match=r"decimal_max_digits|decimal_max_places"):
        LineItem(
            description="x",
            quantity=Decimal(0.1),  # noqa: RUF032 - the float artifact is the point
            unit_price_cents=100,
            amount_cents=10,
            confidence=0.9,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unit_price_cents", -1),
        ("amount_cents", -2500),
        ("amount_cents", 12.5),
        ("amount_cents", "2500"),
        ("amount_cents", MAX_CENTS + 1),
        ("quantity", "0"),
        ("quantity", "-1"),
        ("quantity", "0.0000001"),
        ("quantity", "NaN"),
        ("tax_rate", "-1"),
        ("tax_rate", "100.0001"),
        ("tax_rate", "8.87501"),
        ("confidence", 1.01),
        ("confidence", -0.01),
        ("description", "   "),
    ],
)
def test_line_item_boundaries(field: str, value: object) -> None:
    item = invoice_data()["line_items"][0] | {field: value}
    with pytest.raises(ValidationError):
        parse(LineItem, item)


@pytest.mark.parametrize("field", ["subtotal", "tax_total", "discount", "total"])
def test_negative_totals_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        parse(ExtractionResult, invoice_data(**{field: ef(-1)}))


@pytest.mark.parametrize("code", ["usd", "US", "EURO", "12A", "zł"])
def test_currency_must_be_iso_4217_alpha(code: str) -> None:
    with pytest.raises(ValidationError):
        parse(ExtractionResult, invoice_data(currency=ef(code)))


def test_impossible_issue_date_rejected() -> None:
    with pytest.raises(ValidationError):
        parse(ExtractionResult, invoice_data(issue_date=ef("2026-02-30")))


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_inclusive_bounds(confidence: float) -> None:
    assert parse(ExtractedField[str], ef("x", confidence=confidence)).confidence == confidence


@pytest.mark.parametrize("raw", ['{"value": "x", "status": "extracted", "confidence": NaN}'])
def test_confidence_nan_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedField[str].model_validate_json(raw)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (ef(None, status="extracted"), "value must be set"),
        (ef(None, status="inferred"), "value must be set"),
        (ef("x", status="missing"), "value must be null"),
        (ef("x", status="illegible"), "value must be null"),
        (ef(None, status="missing", anchor_=anchor()), "cannot carry an anchor"),
    ],
)
def test_field_status_invariants(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse(ExtractedField[str], data)


def test_illegible_field_may_point_at_region() -> None:
    field = parse(ExtractedField[str], ef(None, status="illegible", anchor_=anchor()))
    assert field.status is FieldStatus.ILLEGIBLE
    assert not field.status.has_value


def test_complete_extraction_requires_total_and_currency() -> None:
    with pytest.raises(ValidationError, match="requires currency and total"):
        parse(ExtractionResult, invoice_data(total=missing()))
    with pytest.raises(ValidationError, match="requires currency and total"):
        parse(ExtractionResult, invoice_data(currency=missing()))
    partial = parse(ExtractionResult, invoice_data(status="partial", total=missing()))
    assert partial.status is ExtractionStatus.PARTIAL


def test_anchor_beyond_last_page_rejected() -> None:
    with pytest.raises(ValidationError, match="references page 2"):
        parse(ExtractionResult, invoice_data(page_count=1))


def test_anchors_in_document_order() -> None:
    result = parse(ExtractionResult, invoice_data())
    assert [a.page_number for a in result.anchors()] == [1, 1, 2, 2]


def test_line_item_cap() -> None:
    item = invoice_data()["line_items"][0]
    with pytest.raises(ValidationError):
        parse(ExtractionResult, invoice_data(line_items=[item] * 1001))


# ---------------------------------------------------------------------------
# Reconciliation math
# ---------------------------------------------------------------------------
HALF_UP = ReconciliationPolicy(rounding=RoundingMode.HALF_UP)
HALF_EVEN = ReconciliationPolicy(rounding=RoundingMode.HALF_EVEN)


@pytest.mark.parametrize(
    ("amount", "half_up", "half_even"),
    [
        (Decimal("2.5"), 3, 2),
        (Decimal("3.5"), 4, 4),
        (Decimal("2.4999999"), 2, 2),
        (Decimal("2.5000001"), 3, 3),
        (Decimal("-2.5"), -3, -2),
        (Decimal(0), 0, 0),
    ],
)
def test_rounding_modes(amount: Decimal, half_up: int, half_even: int) -> None:
    assert HALF_UP.to_cents(amount) == half_up
    assert HALF_EVEN.to_cents(amount) == half_even


@pytest.mark.parametrize("amount", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_non_finite_amount_cannot_be_rounded(amount: Decimal) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        HALF_UP.to_cents(amount)


def test_float_artifacts_do_not_leak_into_decimal_math() -> None:
    assert 0.1 + 0.2 != 0.3  # the bug being guarded against
    quantity = Decimal("0.1") + Decimal("0.2")
    assert HALF_UP.expected_line_amount(quantity, 1000) == 300
    # 1.15 * 100 in binary float is 114.99999999999999 and would truncate to 114.
    assert int(1.15 * 100) == 114
    assert HALF_UP.expected_line_amount(Decimal("1.15"), 100) == 115


def test_line_amount_rounding_boundary() -> None:
    # 1.5 x 19.99 = 29.985 exactly: half-up -> 29.99, half-even -> 29.98.
    assert HALF_UP.expected_line_amount(Decimal("1.5"), 1999) == 2999
    assert HALF_EVEN.expected_line_amount(Decimal("1.5"), 1999) == 2998


def test_tax_rate_application() -> None:
    assert HALF_UP.expected_tax(5499, Decimal(23)) == 1265  # 1264.77
    assert HALF_UP.expected_tax(50, Decimal(5)) == 3  # 2.5
    assert HALF_EVEN.expected_tax(50, Decimal(5)) == 2
    assert HALF_UP.expected_tax(1000, Decimal("8.875")) == 89  # 88.75
    assert HALF_UP.expected_tax(0, Decimal(23)) == 0
    assert HALF_UP.expected_tax(12345, Decimal(0)) == 0


def test_extreme_magnitudes_stay_exact() -> None:
    # 34 significant digits: beyond the default 28-digit Decimal context.
    quantity = Decimal("999999999999.999999")
    assert HALF_UP.expected_line_amount(quantity, MAX_CENTS) == 999_999_999_999_999_999 * 10**9
    assert HALF_UP.expected_tax(MAX_CENTS, Decimal("99.9999")) == 999_999 * 10**9


def test_grand_total_and_tolerance() -> None:
    assert ReconciliationPolicy.expected_total(5499, 1265, 0) == 6764
    assert ReconciliationPolicy.expected_total(100, 0, 250) == -150
    policy = ReconciliationPolicy(tolerance_cents=1)
    assert policy.within_tolerance(6764, 6765)
    assert policy.within_tolerance(6764, 6763)
    assert not policy.within_tolerance(6764, 6766)
    exact = ReconciliationPolicy(tolerance_cents=0)
    assert exact.within_tolerance(100, 100)
    assert not exact.within_tolerance(100, 101)


def test_fixture_invoice_reconciles_under_default_policy() -> None:
    result = parse(ExtractionResult, invoice_data())
    policy = ReconciliationPolicy()
    for item in result.line_items:
        expected = policy.expected_line_amount(item.quantity, item.unit_price_cents)
        assert policy.within_tolerance(expected, item.amount_cents)
    assert result.subtotal.value == sum(i.amount_cents for i in result.line_items)
    for line in result.tax_breakdown:
        assert policy.expected_tax(line.taxable_base_cents, line.rate) == line.tax_cents
    assert result.subtotal.value is not None
    assert result.tax_total.value is not None
    assert result.discount.value is not None
    assert result.total.value == policy.expected_total(
        result.subtotal.value, result.tax_total.value, result.discount.value
    )


def test_policy_defaults_and_wire_format() -> None:
    policy = ReconciliationPolicy()
    assert policy.rules == ALL_RULES
    assert policy.tolerance_cents == 1
    assert ReconciliationPolicy.model_validate_json(policy.model_dump_json()) == policy


@pytest.mark.parametrize(
    "override",
    [
        {"tolerance_cents": -1},
        {"tolerance_cents": 101},
        {"tolerance_cents": 0.5},
        {"rounding": "bankers"},
        {"rules": []},
        {"rules": ["sum_of_vibes"]},
    ],
)
def test_policy_bounds(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse(ReconciliationPolicy, override)


def test_discrepancy_delta_invariants() -> None:
    d = Discrepancy(
        rule=ReconciliationRule.GRAND_TOTAL,
        expected_cents=6764,
        actual_cents=6700,
        delta_cents=-64,
        location="/total",
    )
    assert d.delta_cents == d.actual_cents - d.expected_cents
    with pytest.raises(ValidationError, match="actual_cents - expected_cents"):
        Discrepancy(
            rule=ReconciliationRule.GRAND_TOTAL,
            expected_cents=6764,
            actual_cents=6700,
            delta_cents=64,
        )
    with pytest.raises(ValidationError, match="non-zero"):
        Discrepancy(
            rule=ReconciliationRule.GRAND_TOTAL,
            expected_cents=1,
            actual_cents=1,
            delta_cents=0,
        )


@pytest.mark.parametrize("pointer", ["line_items/2", "/a~2", "/a~", "x"])
def test_discrepancy_location_must_be_json_pointer(pointer: str) -> None:
    with pytest.raises(ValidationError):
        parse(
            Discrepancy,
            {
                "rule": "grand_total",
                "expected_cents": 1,
                "actual_cents": 5,
                "delta_cents": 4,
                "location": pointer,
            },
        )


@pytest.mark.parametrize("pointer", ["", "/line_items/2/amount_cents", "/a~0b~1c"])
def test_valid_json_pointers(pointer: str) -> None:
    data = {
        "rule": "grand_total",
        "expected_cents": 1,
        "actual_cents": 5,
        "delta_cents": 4,
        "location": pointer,
    }
    assert parse(Discrepancy, data).location == pointer


def _discrepancy(rule: str = "grand_total", delta: int = 64) -> dict[str, Any]:
    return {
        "rule": rule,
        "expected_cents": 6764,
        "actual_cents": 6764 + delta,
        "delta_cents": delta,
        "location": "/total",
    }


def test_report_passed_and_round_trip() -> None:
    report = parse(ReconciliationReport, report_data())
    assert report.status is ReconciliationStatus.PASSED
    assert report.max_abs_delta_cents == 0
    assert ReconciliationReport.model_validate_json(report.model_dump_json()) == report


def test_report_failed() -> None:
    report = parse(
        ReconciliationReport,
        report_data(
            status="failed", discrepancies=[_discrepancy(delta=-64), _discrepancy(delta=7)]
        ),
    )
    assert report.max_abs_delta_cents == 64


def test_report_skipped_when_nothing_evaluated() -> None:
    report = parse(
        ReconciliationReport,
        report_data(status="skipped", evaluated_rules=[], skipped_rules=sorted(ALL_RULES)),
    )
    assert report.status is ReconciliationStatus.SKIPPED


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"status": "failed"}, "status must be passed"),
        ({"discrepancies": [_discrepancy()]}, "status must be failed"),
        (
            {"status": "failed", "discrepancies": [_discrepancy(delta=1)]},
            "within tolerance",
        ),
        (
            {"status": "failed", "discrepancies": [_discrepancy(delta=-1)]},
            "within tolerance",
        ),
        ({"skipped_rules": ["grand_total"]}, "both evaluated and skipped"),
        ({"evaluated_rules": ["grand_total"]}, "partition"),
        (
            {
                "status": "failed",
                "evaluated_rules": sorted(ALL_RULES - {ReconciliationRule.GRAND_TOTAL}),
                "skipped_rules": ["grand_total"],
                "discrepancies": [_discrepancy("grand_total")],
            },
            "was not evaluated",
        ),
        (
            {"status": "passed", "evaluated_rules": [], "skipped_rules": sorted(ALL_RULES)},
            "status must be skipped",
        ),
    ],
)
def test_report_invariants(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse(ReconciliationReport, report_data(**override))


# ---------------------------------------------------------------------------
# API wire models
# ---------------------------------------------------------------------------
def test_job_status_terminality() -> None:
    assert {s for s in JobStatus if s.is_terminal} == {JobStatus.SUCCEEDED, JobStatus.FAILED}


def test_ingestion_options_defaults_and_nested_parsing() -> None:
    assert IngestionOptions().rasterization == RasterizationConfig()
    options = parse(
        IngestionOptions,
        {
            "document_kind": "receipt",
            "rasterization": {"dpi": 300, "color_mode": "L", "pages": [{"start": 1, "end": 1}]},
            "extraction": {"model": "claude-haiku-5-5", "effort": "low"},
            "reconciliation": {"tolerance_cents": 0, "rules": ["grand_total"]},
            "idempotency_key": "upload:2026-10-29:42",
        },
    )
    assert options.extraction is not None
    assert options.extraction.model is VisionModelId.HAIKU_5_5
    assert options.reconciliation.rules == frozenset({ReconciliationRule.GRAND_TOTAL})


@pytest.mark.parametrize(
    "override",
    [
        {"idempotency_key": ""},
        {"idempotency_key": "has space"},
        {"idempotency_key": "-leading-dash"},
        {"document_kind": "passport"},
        {"callback_url": "https://example.invalid"},
    ],
)
def test_ingestion_options_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse(IngestionOptions, override)


def test_ingestion_accepted_must_address_its_job() -> None:
    with pytest.raises(ValidationError, match="address the accepted job"):
        parse(
            IngestionAccepted,
            {
                "job_id": str(JOB_ID),
                "status": "queued",
                "status_url": "/v1/jobs/00000000-0000-0000-0000-000000000000",
                "created_at": NOW.isoformat(),
            },
        )


def test_problem_details_rfc9457_shape() -> None:
    problem = ProblemDetails(title="Unsupported Media Type", status=415)
    assert problem.type == "about:blank"
    assert PROBLEM_JSON_MEDIA_TYPE == "application/problem+json"
    body = json.loads(problem.model_dump_json(exclude_none=True, exclude_defaults=False))
    assert body == {
        "type": "about:blank",
        "title": "Unsupported Media Type",
        "status": 415,
        "errors": [],
    }
    detailed = parse(
        ProblemDetails,
        {
            "type": "https://errors.example.invalid/validation",
            "title": "Invalid ingestion options",
            "status": 422,
            "detail": "rasterization.dpi must be between 72 and 600",
            "instance": f"/v1/jobs/{JOB_ID}",
            "errors": [{"pointer": "/rasterization/dpi", "detail": "must be <= 600"}],
        },
    )
    assert detailed.errors[0].pointer == "/rasterization/dpi"


@pytest.mark.parametrize(
    "override",
    [
        {"status": 200},
        {"status": 399},
        {"status": 600},
        {"title": ""},
        {"type": "not a uri"},
        {"errors": [{"pointer": "rasterization", "detail": "x"}]},
        {"trace": "stack"},
    ],
)
def test_problem_details_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse(ProblemDetails, {"title": "Bad Request", "status": 400} | override)


def test_succeeded_job_round_trip() -> None:
    job = parse(JobStatusResponse, job_data())
    assert job.status is JobStatus.SUCCEEDED
    assert JobStatusResponse.model_validate_json(job.model_dump_json()) == job


def test_running_job_without_outcome() -> None:
    job = parse(
        JobStatusResponse,
        job_data(status="extracting", extraction=None, reconciliation=None),
    )
    assert not job.status.is_terminal


def test_failed_job_with_problem() -> None:
    problem = {"title": "Rasterization failed", "status": 422, "detail": "encrypted PDF"}
    job = parse(
        JobStatusResponse,
        job_data(status="failed", extraction=None, reconciliation=None, pages=[], error=problem),
    )
    assert job.error is not None
    assert job.error.status == 422


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"reconciliation": None}, "succeeded job requires"),
        ({"extraction": None}, "succeeded job requires"),
        ({"error": {"title": "x", "status": 500}}, "succeeded job requires"),
        ({"status": "failed", "error": None}, "failed job requires an error"),
        (
            {
                "status": "reconciling",
                "reconciliation": None,
                "error": {"title": "x", "status": 500},
            },
            "must not carry",
        ),
        ({"status": "queued"}, "must not carry"),
        ({"updated_at": (NOW - timedelta(seconds=1)).isoformat()}, "precedes created_at"),
        ({"pages": [page_data(2), page_data(1)]}, "ascending page order"),
        ({"pages": [page_data(1), page_data(1)]}, "ascending page order"),
        ({"pages": [page_data(1)]}, "page_count disagrees"),
    ],
)
def test_job_lifecycle_invariants(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse(JobStatusResponse, job_data(**override))


def test_job_timestamps_accept_any_offset() -> None:
    plus_two = timezone(timedelta(hours=2))
    job = parse(
        JobStatusResponse,
        job_data(updated_at=(NOW + timedelta(seconds=1)).astimezone(plus_two).isoformat()),
    )
    assert job.updated_at.tzinfo is UTC


def test_utc_now_is_aware_utc() -> None:
    assert schemas.utc_now().tzinfo is UTC
