"""Extraction accuracy against ground-truth labels.

A result is broken into *slots*: the scalar fields, each line item and each tax line.
Every slot is scored once, and all metrics derive from that one table:

* text fields: normalized exact match (EM) and Levenshtein distance / similarity;
* amounts (integer cents): absolute and percentage delta;
* anchors: bounding-box IoU against the labelled region;
* precision / recall: a slot is a true positive when both sides have it and it is
  correct; a wrong value counts as one false positive *and* one false negative;
* hallucination: an extracted slot the ground truth does not support, either because
  the label has no such value, or because the value claims to be read from the page
  but its anchor misses the labelled region (``inferred`` values need no anchor).

Ratios with an empty denominator are ``None`` ("undefined"), never a fake 0 or 1.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

from pydantic import Field

from multimodal_vision_pipeline.schemas import (
    DEFAULT_VISION_MODEL,
    DocumentKind,
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    FieldAnchor,
    FieldStatus,
    LineItem,
    PipelineModel,
    TaxLine,
    VendorInfo,
    VisionModelId,
)

DEFAULT_IOU_THRESHOLD: Final = 0.5
LINE_ITEM_MATCH_THRESHOLD: Final = 0.5
"""Minimum pairing score (description similarity and amount closeness, averaged)."""

UnitInterval = Field(ge=0.0, le=1.0, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions, deletions, substitutions), O(len(a) * len(b))."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """``1 - distance / longer length``; two empty strings are identical (1.0)."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - levenshtein(a, b) / longest


def normalize_text(value: str) -> str:
    """NFKC, case-folded, whitespace collapsed: what EM compares."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mean(values: Sequence[float]) -> float | None:
    return ratio(sum(values), len(values))


def iou(a: FieldAnchor, b: FieldAnchor) -> float:
    """Intersection over union of two anchors; anchors on different pages score 0."""
    if a.page_number != b.page_number:
        return 0.0
    width = min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)
    height = min(a.bbox.y1, b.bbox.y1) - max(a.bbox.y0, b.bbox.y0)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / (a.bbox.area + b.bbox.area - intersection)


def percent_delta(expected_cents: int, actual_cents: int) -> float | None:
    """``|actual - expected| / expected`` in percent; undefined for a non-zero vs zero."""
    if expected_cents == 0:
        return 0.0 if actual_cents == 0 else None
    return abs(actual_cents - expected_cents) / expected_cents * 100


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


class TextFieldScore(PipelineModel):
    """A textual field; a side without a value compares as the empty string."""

    field: str
    expected: str | None
    actual: str | None
    exact_match: bool
    distance: int = Field(ge=0)
    similarity: float = UnitInterval


class AmountScore(PipelineModel):
    """A monetary field in integer cents; deltas are ``None`` when a side is missing."""

    field: str
    expected_cents: int | None
    actual_cents: int | None
    abs_delta_cents: int | None = Field(ge=0)
    pct_delta: float | None = Field(ge=0, allow_inf_nan=False)

    @classmethod
    def compare(cls, field: str, expected: int | None, actual: int | None) -> "AmountScore":
        if expected is None or actual is None:
            return cls(
                field=field,
                expected_cents=expected,
                actual_cents=actual,
                abs_delta_cents=None,
                pct_delta=None,
            )
        return cls(
            field=field,
            expected_cents=expected,
            actual_cents=actual,
            abs_delta_cents=abs(actual - expected),
            pct_delta=percent_delta(expected, actual),
        )


class AnchorScore(PipelineModel):
    field: str
    iou: float = UnitInterval


class MatchCounts(PipelineModel):
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)

    def __add__(self, other: "MatchCounts") -> "MatchCounts":
        return MatchCounts(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    @property
    def precision(self) -> float | None:
        return ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float | None:
        return ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None:
            return None
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class DocumentEvaluation(PipelineModel):
    """Every score for one (ground truth, extraction) pair."""

    text_fields: tuple[TextFieldScore, ...]
    amounts: tuple[AmountScore, ...]
    anchors: tuple[AnchorScore, ...]
    counts: MatchCounts
    extracted_slots: int = Field(ge=0)
    hallucinated: tuple[str, ...]
    line_items_expected: int = Field(ge=0)
    line_items_extracted: int = Field(ge=0)
    line_items_matched: int = Field(ge=0)
    iou_threshold: float = UnitInterval

    @property
    def exact_match_rate(self) -> float | None:
        return mean([float(score.exact_match) for score in self.text_fields])

    @property
    def hallucination_rate(self) -> float | None:
        return ratio(len(self.hallucinated), self.extracted_slots)

    @property
    def mean_iou(self) -> float | None:
        return mean([score.iou for score in self.anchors])

    @property
    def anchor_accuracy(self) -> float | None:
        return mean([float(score.iou >= self.iou_threshold) for score in self.anchors])


# ---------------------------------------------------------------------------
# Slot table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Slot:
    name: str
    expected: bool
    actual: bool
    correct: bool
    expected_anchor: FieldAnchor | None
    actual_anchor: FieldAnchor | None
    read_from_page: bool
    """The extraction claims the value was read at a location (status ``extracted``)."""

    def counts(self) -> MatchCounts:
        if self.expected and self.actual:
            if self.correct:
                return MatchCounts(true_positives=1)
            return MatchCounts(false_positives=1, false_negatives=1)
        return MatchCounts(false_positives=int(self.actual), false_negatives=int(self.expected))

    def anchor_iou(self) -> float | None:
        """IoU when there is a labelled region to hit and the extraction has a value."""
        if not (self.expected and self.actual) or self.expected_anchor is None:
            return None
        return 0.0 if self.actual_anchor is None else iou(self.expected_anchor, self.actual_anchor)

    def hallucinated(self) -> bool:
        if not self.actual:
            return False
        if not self.expected:
            return True
        overlap = self.anchor_iou()
        return self.read_from_page and overlap is not None and overlap == 0.0


def _value[T](field: ExtractedField[T]) -> T | None:
    return field.value if field.status.has_value else None


def _scalar_slot[T](name: str, expected: ExtractedField[T], actual: ExtractedField[T]) -> _Slot:
    expected_value, actual_value = _value(expected), _value(actual)
    if isinstance(expected_value, str) and isinstance(actual_value, str):
        correct = normalize_text(expected_value) == normalize_text(actual_value)
    else:
        correct = expected_value == actual_value
    return _Slot(
        name=name,
        expected=expected_value is not None,
        actual=actual_value is not None,
        correct=correct,
        expected_anchor=expected.anchor,
        actual_anchor=actual.anchor,
        read_from_page=actual.status is FieldStatus.EXTRACTED,
    )


def _text_score[T](
    name: str, expected: ExtractedField[T], actual: ExtractedField[T]
) -> TextFieldScore | None:
    expected_value, actual_value = _value(expected), _value(actual)
    if expected_value is None and actual_value is None:
        return None
    expected_text = "" if expected_value is None else _as_text(expected_value)
    actual_text = "" if actual_value is None else _as_text(actual_value)
    left, right = normalize_text(expected_text), normalize_text(actual_text)
    return TextFieldScore(
        field=name,
        expected=None if expected_value is None else expected_text,
        actual=None if actual_value is None else actual_text,
        exact_match=expected_value is not None and actual_value is not None and left == right,
        distance=levenshtein(left, right),
        similarity=similarity(left, right),
    )


def _as_text(value: object) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _line_item_pair_score(expected: LineItem, actual: LineItem) -> float:
    text = similarity(normalize_text(expected.description), normalize_text(actual.description))
    delta = abs(actual.amount_cents - expected.amount_cents)
    amount = max(0.0, 1.0 - delta / max(expected.amount_cents, 1))
    return (text + amount) / 2


def match_line_items(
    expected: Sequence[LineItem], actual: Sequence[LineItem]
) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing by score, best first; ties keep document order.

    Order-independent, so a skipped or inserted row does not shift every later pair.
    """
    candidates = sorted(
        (
            (-score, i, j)
            for i, left in enumerate(expected)
            for j, right in enumerate(actual)
            if (score := _line_item_pair_score(left, right)) >= LINE_ITEM_MATCH_THRESHOLD
        ),
    )
    used_expected: set[int] = set()
    used_actual: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i not in used_expected and j not in used_actual:
            used_expected.add(i)
            used_actual.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def match_tax_lines(
    expected: Sequence[TaxLine], actual: Sequence[TaxLine]
) -> list[tuple[int, int]]:
    """Pairs tax lines with the same rate, first come first served."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, left in enumerate(expected):
        for j, right in enumerate(actual):
            if j not in used and right.rate == left.rate:
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


def _row_slot(
    name: str,
    expected: LineItem | TaxLine | None,
    actual: LineItem | TaxLine | None,
    *,
    correct: bool,
) -> _Slot:
    return _Slot(
        name=name,
        expected=expected is not None,
        actual=actual is not None,
        correct=correct,
        expected_anchor=None if expected is None else expected.anchor,
        actual_anchor=None if actual is None else actual.anchor,
        read_from_page=actual is not None,
    )


def _line_item_correct(expected: LineItem, actual: LineItem) -> bool:
    return (
        normalize_text(expected.description) == normalize_text(actual.description)
        and expected.quantity == actual.quantity
        and expected.unit_price_cents == actual.unit_price_cents
        and expected.amount_cents == actual.amount_cents
        and expected.tax_rate == actual.tax_rate
    )


def _tax_line_correct(expected: TaxLine, actual: TaxLine) -> bool:
    return (
        expected.taxable_base_cents == actual.taxable_base_cents
        and expected.tax_cents == actual.tax_cents
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _text_fields(result: ExtractionResult) -> list[tuple[str, ExtractedField[Any]]]:
    return [
        ("vendor.name", result.vendor.name),
        ("vendor.tax_id", result.vendor.tax_id),
        ("vendor.address", result.vendor.address),
        ("document_number", result.document_number),
        ("issue_date", result.issue_date),
        ("currency", result.currency),
    ]


def _amount_fields(result: ExtractionResult) -> list[tuple[str, ExtractedField[Any]]]:
    return [
        ("subtotal", result.subtotal),
        ("tax_total", result.tax_total),
        ("discount", result.discount),
        ("total", result.total),
    ]


def evaluate(
    expected: ExtractionResult,
    actual: ExtractionResult,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> DocumentEvaluation:
    """Score ``actual`` against the ``expected`` labels."""
    slots: list[_Slot] = []
    text_scores: list[TextFieldScore] = []
    amounts: list[AmountScore] = []

    for (name, want), (_, got) in zip(_text_fields(expected), _text_fields(actual), strict=True):
        slots.append(_scalar_slot(name, want, got))
        if (score := _text_score(name, want, got)) is not None:
            text_scores.append(score)

    for (name, want), (_, got) in zip(
        _amount_fields(expected),
        _amount_fields(actual),
        strict=True,
    ):
        slots.append(_scalar_slot(name, want, got))
        want_cents, got_cents = _value(want), _value(got)
        if want_cents is not None or got_cents is not None:
            amounts.append(AmountScore.compare(name, want_cents, got_cents))

    item_pairs = match_line_items(expected.line_items, actual.line_items)
    paired_items = dict(item_pairs)
    for i, want_item in enumerate(expected.line_items):
        j = paired_items.get(i)
        got_item = None if j is None else actual.line_items[j]
        correct = got_item is not None and _line_item_correct(want_item, got_item)
        slots.append(_row_slot(f"line_items/{i}", want_item, got_item, correct=correct))
        amounts.append(
            AmountScore.compare(
                f"line_items/{i}/amount_cents",
                want_item.amount_cents,
                None if got_item is None else got_item.amount_cents,
            )
        )
    matched_actual = {j for _, j in item_pairs}
    for j, got_item in enumerate(actual.line_items):
        if j not in matched_actual:
            slots.append(_row_slot(f"extracted_line_items/{j}", None, got_item, correct=False))
            amounts.append(
                AmountScore.compare(
                    f"extracted_line_items/{j}/amount_cents", None, got_item.amount_cents
                )
            )

    tax_pairs = dict(match_tax_lines(expected.tax_breakdown, actual.tax_breakdown))
    for i, want_tax in enumerate(expected.tax_breakdown):
        j = tax_pairs.get(i)
        got_tax = None if j is None else actual.tax_breakdown[j]
        correct = got_tax is not None and _tax_line_correct(want_tax, got_tax)
        slots.append(_row_slot(f"tax_breakdown/{i}", want_tax, got_tax, correct=correct))
        amounts.append(
            AmountScore.compare(
                f"tax_breakdown/{i}/tax_cents",
                want_tax.tax_cents,
                None if got_tax is None else got_tax.tax_cents,
            )
        )
    matched_taxes = set(tax_pairs.values())
    for j, got_tax in enumerate(actual.tax_breakdown):
        if j not in matched_taxes:
            slots.append(_row_slot(f"extracted_tax_breakdown/{j}", None, got_tax, correct=False))
            amounts.append(
                AmountScore.compare(
                    f"extracted_tax_breakdown/{j}/tax_cents", None, got_tax.tax_cents
                )
            )

    counts = MatchCounts()
    for slot in slots:
        counts += slot.counts()
    return DocumentEvaluation(
        text_fields=tuple(text_scores),
        amounts=tuple(amounts),
        anchors=tuple(
            AnchorScore(field=slot.name, iou=overlap)
            for slot in slots
            if (overlap := slot.anchor_iou()) is not None
        ),
        counts=counts,
        extracted_slots=sum(slot.actual for slot in slots),
        hallucinated=tuple(slot.name for slot in slots if slot.hallucinated()),
        line_items_expected=len(expected.line_items),
        line_items_extracted=len(actual.line_items),
        line_items_matched=len(item_pairs),
        iou_threshold=iou_threshold,
    )


def empty_extraction(
    *, page_count: int = 1, model: VisionModelId = DEFAULT_VISION_MODEL
) -> ExtractionResult:
    """What a failed extraction scores as: every field missing, no rows."""

    def missing() -> ExtractedField[Any]:
        return ExtractedField[Any](confidence=0.0, status=FieldStatus.MISSING)

    return ExtractionResult(
        document_kind=DocumentKind.INVOICE,
        model=model,
        status=ExtractionStatus.FAILED,
        page_count=page_count,
        vendor=VendorInfo(name=missing(), tax_id=missing(), address=missing()),
        document_number=missing(),
        issue_date=missing(),
        currency=missing(),
        subtotal=missing(),
        tax_total=missing(),
        discount=missing(),
        total=missing(),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class AccuracySummary(PipelineModel):
    """Micro-averaged metrics over many documents (every slot weighs the same)."""

    documents: int = Field(ge=0)
    counts: MatchCounts
    precision: float | None
    recall: float | None
    f1: float | None
    exact_match_rate: float | None
    mean_similarity: float | None
    mean_levenshtein: float | None
    amount_exact_rate: float | None
    mean_abs_delta_cents: float | None
    mean_pct_delta: float | None
    mean_iou: float | None
    anchor_accuracy: float | None
    hallucination_rate: float | None
    line_item_recall: float | None


def summarize(evaluations: Sequence[DocumentEvaluation]) -> AccuracySummary:
    counts = MatchCounts()
    for evaluation in evaluations:
        counts += evaluation.counts
    text = [score for evaluation in evaluations for score in evaluation.text_fields]
    amounts = [score for evaluation in evaluations for score in evaluation.amounts]
    deltas = [score.abs_delta_cents for score in amounts if score.abs_delta_cents is not None]
    percents = [score.pct_delta for score in amounts if score.pct_delta is not None]
    anchors = [
        (score.iou, evaluation.iou_threshold)
        for evaluation in evaluations
        for score in evaluation.anchors
    ]
    return AccuracySummary(
        documents=len(evaluations),
        counts=counts,
        precision=counts.precision,
        recall=counts.recall,
        f1=counts.f1,
        exact_match_rate=mean([float(score.exact_match) for score in text]),
        mean_similarity=mean([score.similarity for score in text]),
        mean_levenshtein=mean([float(score.distance) for score in text]),
        amount_exact_rate=mean([float(delta == 0) for delta in deltas]),
        mean_abs_delta_cents=mean([float(delta) for delta in deltas]),
        mean_pct_delta=mean(percents),
        mean_iou=mean([overlap for overlap, _ in anchors]),
        anchor_accuracy=mean([float(overlap >= threshold) for overlap, threshold in anchors]),
        hallucination_rate=ratio(
            sum(len(evaluation.hallucinated) for evaluation in evaluations),
            sum(evaluation.extracted_slots for evaluation in evaluations),
        ),
        line_item_recall=ratio(
            sum(evaluation.line_items_matched for evaluation in evaluations),
            sum(evaluation.line_items_expected for evaluation in evaluations),
        ),
    )
