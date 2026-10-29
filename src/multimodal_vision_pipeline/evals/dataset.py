"""Synthetic invoice dataset: rendered page images with exact ground-truth labels.

Everything is drawn in memory with Pillow's bundled font, so generation needs no system
fonts, files or network. Labels are an ``ExtractionResult`` whose amounts reconcile
exactly and whose anchors are the measured boxes of the text actually drawn.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from multimodal_vision_pipeline.schemas import (
    DEFAULT_VISION_MODEL,
    BoundingBox,
    DocumentKind,
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    FieldAnchor,
    FieldStatus,
    LineItem,
    ReconciliationPolicy,
    TaxLine,
    VendorInfo,
)

PAGE_SIZE: Final = (850, 1100)
"""US Letter at 100 DPI."""
PAGE_DPI: Final = 100
MAX_LINE_ITEMS: Final = 12
"""Rows that fit the table on one page."""

_MARGIN: Final = 60
_COLUMNS: Final = (_MARGIN, 430, 520, 640, 790)
"""x of: description (left), qty, unit price, tax rate, amount (right edges for numbers)."""
_ROW_HEIGHT: Final = 30

# Printable ASCII only: Pillow's bundled font has no glyphs for e.g. Polish diacritics,
# and a label must match what is visibly on the page.
_VENDOR_PREFIXES: Final = (
    "Northwind",
    "Blue Harbor",
    "Kestrel",
    "Vistula",
    "Granite Peak",
    "Orbis",
)
_VENDOR_SUFFIXES: Final = ("Trading Ltd", "Logistics GmbH", "Supplies Inc", "Sp. z o.o.", "S.A.")
_STREETS: Final = ("Market Street", "Dluga Street", "Harbour Road", "Kantstrasse", "Rue de Lyon")
_CITIES: Final = ("Gdansk 80-831", "Berlin 10623", "Leeds LS1 4AP", "Lyon 69002", "Austin TX 78701")
_PRODUCTS: Final = (
    "Thermal paper rolls",
    "Toner cartridge black",
    "Pallet handling",
    "Consulting hours",
    "Cloud storage 1 TB",
    "Shipping insurance",
    "Barcode scanner",
    "Archive boxes",
    "Maintenance visit",
    "Software licence",
)
_QUANTITIES: Final = tuple(Decimal(q) for q in ("1", "2", "3", "4", "5", "10", "0.5", "2.5", "12"))
_CURRENCIES: Final = ("EUR", "USD", "PLN", "GBP")
_TAX_RATES: Final = tuple(Decimal(r) for r in ("0", "5", "8", "23"))
_DISCOUNT_SHARE: Final = 0.3
"""Fraction of invoices that print a discount line."""
_FIRST_ISSUE_DATE: Final = date(2025, 1, 1)
_ISSUE_DATE_SPAN_DAYS: Final = 640


@dataclass(frozen=True, slots=True)
class SyntheticDocument:
    """One rendered page and the labels a perfect extractor would return."""

    document_id: str
    image: Image.Image = field(repr=False)
    ground_truth: ExtractionResult
    dpi: int = PAGE_DPI


def format_cents(cents: int) -> str:
    """``123456`` -> ``"1,234.56"``."""
    return f"{cents // 100:,}.{cents % 100:02d}"


def format_quantity(quantity: Decimal) -> str:
    return format(quantity.normalize(), "f")


def format_rate(rate: Decimal) -> str:
    return f"{format(rate.normalize(), 'f')}%"


class _Canvas:
    """Draws text and returns the normalized box of what was drawn."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", PAGE_SIZE, "white")
        self._draw = ImageDraw.Draw(self.image)
        self._fonts = {size: ImageFont.load_default(size=size) for size in (14, 16, 26)}

    def text(
        self, x: float, y: float, text: str, *, size: int = 16, right: bool = False
    ) -> BoundingBox:
        font = self._fonts[size]
        anchor = "ra" if right else "la"
        self._draw.text((x, y), text, fill="black", font=font, anchor=anchor)
        left, top, right_edge, bottom = self._draw.textbbox((x, y), text, font=font, anchor=anchor)
        width, height = self.image.size
        return BoundingBox(
            x0=left / width, y0=top / height, x1=right_edge / width, y1=bottom / height
        )

    def rule(self, y: float) -> None:
        self._draw.line((_MARGIN, y, PAGE_SIZE[0] - _MARGIN, y), fill="black", width=1)


def _union(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _anchor(box: BoundingBox) -> FieldAnchor:
    return FieldAnchor(page_number=1, bbox=box)


def _value[T](value: T, box: BoundingBox) -> ExtractedField[T]:
    return ExtractedField[T](
        value=value, confidence=1.0, status=FieldStatus.EXTRACTED, anchor=_anchor(box)
    )


def _missing[T]() -> ExtractedField[T]:
    return ExtractedField[T](confidence=1.0, status=FieldStatus.MISSING)


def generate_document(index: int, *, seed: int = 0) -> SyntheticDocument:
    """Render invoice ``index`` of the dataset seeded with ``seed``; fully deterministic."""
    # Reproducible synthetic data, not security: a seeded PRNG is the point.
    rng = random.Random(f"invoice:{seed}:{index}")  # noqa: S311  # nosec B311
    policy = ReconciliationPolicy()
    canvas = _Canvas()
    currency = rng.choice(_CURRENCIES)

    # Header -----------------------------------------------------------------
    vendor_name = f"{rng.choice(_VENDOR_PREFIXES)} {rng.choice(_VENDOR_SUFFIXES)}"
    tax_id = f"{rng.choice(('PL', 'DE', 'GB', 'FR'))}{rng.randrange(10**9, 10**10)}"
    address = f"{rng.randrange(1, 240)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    number = f"INV-{2025 + index % 2}-{rng.randrange(1, 10**6):06d}"
    issued = _FIRST_ISSUE_DATE + timedelta(days=rng.randrange(_ISSUE_DATE_SPAN_DAYS))

    vendor = VendorInfo(
        name=_value(vendor_name, canvas.text(_MARGIN, 50, vendor_name, size=26)),
        tax_id=_value(tax_id, _labelled(canvas, 95, "Tax ID:", tax_id)),
        address=_value(address, _labelled(canvas, 120, "Address:", address)),
    )
    document_number = _value(number, _labelled(canvas, 175, "Invoice no.:", number))
    issue_date = _value(issued, _labelled(canvas, 200, "Issue date:", issued.isoformat()))
    currency_field = _value(currency, _labelled(canvas, 225, "Currency:", currency))

    # Line items -------------------------------------------------------------
    y = 280
    for x, title, right in zip(
        _COLUMNS,
        ("Description", "Qty", "Unit price", "Tax", "Amount"),
        (False, *[True] * 4),
        strict=True,
    ):
        canvas.text(x, y, title, size=14, right=right)
    canvas.rule(y + 22)
    items: list[LineItem] = []
    for _ in range(rng.randint(1, MAX_LINE_ITEMS)):
        y += _ROW_HEIGHT
        quantity = rng.choice(_QUANTITIES)
        unit_price = rng.randrange(100, 250_000)
        rate = rng.choice(_TAX_RATES)
        amount = policy.expected_line_amount(quantity, unit_price)
        description = rng.choice(_PRODUCTS)
        cells = [
            canvas.text(_COLUMNS[0], y, description),
            canvas.text(_COLUMNS[1], y, format_quantity(quantity), right=True),
            canvas.text(_COLUMNS[2], y, format_cents(unit_price), right=True),
            canvas.text(_COLUMNS[3], y, format_rate(rate), right=True),
            canvas.text(_COLUMNS[4], y, format_cents(amount), right=True),
        ]
        items.append(
            LineItem(
                description=description,
                quantity=quantity,
                unit_price_cents=unit_price,
                amount_cents=amount,
                tax_rate=rate,
                confidence=1.0,
                anchor=_anchor(_union(cells)),
            )
        )
    canvas.rule(y + _ROW_HEIGHT)

    # Tax breakdown and totals ----------------------------------------------
    y += _ROW_HEIGHT + 20
    taxes: list[TaxLine] = []
    for rate in sorted({item.tax_rate for item in items if item.tax_rate is not None}):
        base = sum(item.amount_cents for item in items if item.tax_rate == rate)
        tax = policy.expected_tax(base, rate)
        box = _union(
            [
                canvas.text(
                    _COLUMNS[2], y, f"VAT {format_rate(rate)} on {format_cents(base)}", right=True
                ),
                canvas.text(_COLUMNS[4], y, format_cents(tax), right=True),
            ]
        )
        taxes.append(
            TaxLine(
                rate=rate,
                taxable_base_cents=base,
                tax_cents=tax,
                confidence=1.0,
                anchor=_anchor(box),
            )
        )
        y += 24

    subtotal = sum(item.amount_cents for item in items)
    tax_total = sum(line.tax_cents for line in taxes)
    discount_cents = rng.randrange(1, subtotal // 10 + 2) if rng.random() < _DISCOUNT_SHARE else 0
    total = policy.expected_total(subtotal, tax_total, discount_cents)

    y += 12
    subtotal_field = _value(subtotal, _total_row(canvas, y, "Subtotal", subtotal))
    tax_field = _value(tax_total, _total_row(canvas, y + 26, "Tax", tax_total))
    discount_field: ExtractedField[int] = (
        _value(discount_cents, _total_row(canvas, y + 52, "Discount", discount_cents))
        if discount_cents
        else _missing()
    )
    total_field = _value(total, _total_row(canvas, y + 90, f"Total {currency}", total, size=26))

    truth = ExtractionResult(
        document_kind=DocumentKind.INVOICE,
        model=DEFAULT_VISION_MODEL,
        status=ExtractionStatus.COMPLETE,
        page_count=1,
        vendor=vendor,
        document_number=document_number,
        issue_date=issue_date,
        currency=currency_field,
        line_items=tuple(items),
        tax_breakdown=tuple(taxes),
        subtotal=subtotal_field,
        tax_total=tax_field,
        discount=discount_field,
        total=total_field,
    )
    return SyntheticDocument(
        document_id=f"synthetic-{seed}-{index:05d}", image=canvas.image, ground_truth=truth
    )


def _labelled(canvas: _Canvas, y: float, label: str, value: str) -> BoundingBox:
    """Draws ``label value``; the anchor covers the value only."""
    canvas.text(_MARGIN, y, label, size=14)
    return canvas.text(_MARGIN + 110, y, value)


def _total_row(canvas: _Canvas, y: float, label: str, cents: int, *, size: int = 16) -> BoundingBox:
    canvas.text(_COLUMNS[2], y, label, size=size, right=True)
    return canvas.text(_COLUMNS[4], y, format_cents(cents), size=size, right=True)


def generate_dataset(count: int, *, seed: int = 0) -> Iterator[SyntheticDocument]:
    """``count`` documents, generated lazily so large datasets stay out of memory."""
    if count < 1:
        raise ValueError("count must be >= 1")
    return (generate_document(index, seed=seed) for index in range(count))
