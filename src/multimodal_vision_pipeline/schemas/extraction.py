"""Vision LLM extraction contracts: model selection and the structured extraction target.

All monetary values are integer minor units (cents) so reconciliation is exact.
"""

from collections.abc import Iterator
from datetime import date
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from multimodal_vision_pipeline.schemas._base import (
    Cents,
    Confidence,
    CurrencyCode,
    NonEmptyStr,
    PipelineModel,
    Quantity,
    TaxRate,
)
from multimodal_vision_pipeline.schemas.document import BoundingBox, PageNumber

MAX_OUTPUT_TOKENS = 128_000
MAX_LINE_ITEMS = 1000
MAX_TAX_LINES = 32


class VisionModelId(StrEnum):
    """Vision-capable models the extractor may call. Exact API ids, no date suffixes."""

    OPUS_5_5 = "claude-opus-5-5"
    SONNET_5_5 = "claude-sonnet-5-5"
    HAIKU_5_5 = "claude-haiku-5-5"


DEFAULT_VISION_MODEL = VisionModelId.SONNET_5_5


class Effort(StrEnum):
    """``output_config.effort``: trades reasoning depth against latency and token spend."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ExtractionModelConfig(PipelineModel):
    """Per-request vision model settings.

    Sampling parameters (``temperature``/``top_p``) are deliberately absent: current
    models reject them, and extra fields are forbidden, so they cannot slip in.
    """

    model: VisionModelId = DEFAULT_VISION_MODEL
    max_tokens: int = Field(default=16_000, ge=1, le=MAX_OUTPUT_TOKENS)
    effort: Effort = Effort.HIGH
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=10)


class DocumentKind(StrEnum):
    """Extraction target families."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    FINANCIAL_TABLE = "financial_table"


class FieldStatus(StrEnum):
    """How a field value was obtained."""

    EXTRACTED = "extracted"
    """Read directly from the page."""
    INFERRED = "inferred"
    """Derived by the model from surrounding content (e.g. currency from a symbol)."""
    MISSING = "missing"
    """Not present on the document."""
    ILLEGIBLE = "illegible"
    """Present but unreadable."""

    @property
    def has_value(self) -> bool:
        return self in {FieldStatus.EXTRACTED, FieldStatus.INFERRED}


class ExtractionStatus(StrEnum):
    """Overall outcome of an extraction pass."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class FieldAnchor(PipelineModel):
    """Where on which page a value was read."""

    page_number: PageNumber
    bbox: BoundingBox


class ExtractedField[T](PipelineModel):
    """A single extracted value with its confidence, provenance status and anchor.

    ``value`` is present exactly when ``status`` is ``extracted`` or ``inferred``.
    A ``missing`` field has nothing to anchor to.
    """

    value: T | None = None
    confidence: Confidence
    status: FieldStatus
    anchor: FieldAnchor | None = None

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status.has_value != (self.value is not None):
            raise ValueError(
                f"value must be {'set' if self.status.has_value else 'null'} "
                f"when status is {self.status}"
            )
        if self.status is FieldStatus.MISSING and self.anchor is not None:
            raise ValueError("a missing field cannot carry an anchor")
        return self


TaxId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class VendorInfo(PipelineModel):
    """Issuer of the document."""

    name: ExtractedField[NonEmptyStr]
    tax_id: ExtractedField[TaxId]
    address: ExtractedField[NonEmptyStr]


class LineItem(PipelineModel):
    """One billed row: ``amount_cents`` should equal ``quantity * unit_price_cents``."""

    description: NonEmptyStr
    quantity: Quantity
    unit_price_cents: Cents
    amount_cents: Cents
    tax_rate: TaxRate | None = None
    confidence: Confidence
    anchor: FieldAnchor | None = None


class TaxLine(PipelineModel):
    """One row of the tax breakdown: ``tax_cents`` should equal ``base * rate / 100``."""

    rate: TaxRate
    taxable_base_cents: Cents
    tax_cents: Cents
    confidence: Confidence
    anchor: FieldAnchor | None = None


class ExtractionResult(PipelineModel):
    """Structured extraction of an invoice, receipt or financial table.

    A ``complete`` result has a currency and a total. Every anchor must point at a
    page that exists in the rasterized document.
    """

    document_kind: DocumentKind
    model: VisionModelId
    status: ExtractionStatus
    page_count: PageNumber
    vendor: VendorInfo
    document_number: ExtractedField[NonEmptyStr]
    issue_date: ExtractedField[date]
    currency: ExtractedField[CurrencyCode]
    line_items: tuple[LineItem, ...] = Field(default=(), max_length=MAX_LINE_ITEMS)
    tax_breakdown: tuple[TaxLine, ...] = Field(default=(), max_length=MAX_TAX_LINES)
    subtotal: ExtractedField[Cents]
    tax_total: ExtractedField[Cents]
    discount: ExtractedField[Cents]
    total: ExtractedField[Cents]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.status is ExtractionStatus.COMPLETE and not (
            self.currency.status.has_value and self.total.status.has_value
        ):
            raise ValueError("a complete extraction requires currency and total")
        for anchor in self.anchors():
            if anchor.page_number > self.page_count:
                raise ValueError(
                    f"anchor references page {anchor.page_number} "
                    f"but the document has {self.page_count} page(s)"
                )
        return self

    def anchors(self) -> Iterator[FieldAnchor]:
        """Every anchor in the result, in document order of the fields."""
        candidates: tuple[FieldAnchor | None, ...] = (
            self.vendor.name.anchor,
            self.vendor.tax_id.anchor,
            self.vendor.address.anchor,
            self.document_number.anchor,
            self.issue_date.anchor,
            self.currency.anchor,
            *(item.anchor for item in self.line_items),
            *(line.anchor for line in self.tax_breakdown),
            self.subtotal.anchor,
            self.tax_total.anchor,
            self.discount.anchor,
            self.total.anchor,
        )
        yield from (anchor for anchor in candidates if anchor is not None)
