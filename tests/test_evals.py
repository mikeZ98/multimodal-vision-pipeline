"""Phase 9: evaluation suite (perturbations, synthetic dataset, metrics, benchmark runner).

Hermetic: images are drawn and degraded in memory with Pillow, the vision API is an
``httpx.MockTransport`` or an in-process stub, and nothing touches the network or spawns
a process.
"""

import asyncio
import importlib
import json
import random
import string
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat
from pydantic import SecretStr, ValidationError

from multimodal_vision_pipeline.core.errors import ExtractionParseError
from multimodal_vision_pipeline.core.rasterizer import RasterizedPage
from multimodal_vision_pipeline.core.reconciler import reconcile
from multimodal_vision_pipeline.evals import cli
from multimodal_vision_pipeline.evals.dataset import (
    PAGE_SIZE,
    SyntheticDocument,
    format_cents,
    format_quantity,
    format_rate,
    generate_dataset,
    generate_document,
)
from multimodal_vision_pipeline.evals.evaluator import (
    MatchCounts,
    empty_extraction,
    evaluate,
    iou,
    levenshtein,
    match_line_items,
    normalize_text,
    percent_delta,
    similarity,
    summarize,
)
from multimodal_vision_pipeline.evals.perturbation import (
    PROFILES,
    ContrastLoss,
    DegradationProfile,
    GaussianBlur,
    JpegCompression,
    Perturbation,
    PerturbationStep,
    SaltAndPepper,
    Skew,
)
from multimodal_vision_pipeline.evals.runner import (
    BenchmarkReport,
    VisionPipelineExtractor,
    degrade_labels,
    percentile,
    prepare_sample,
    run_benchmark,
)
from multimodal_vision_pipeline.schemas import (
    BoundingBox,
    ExtractedField,
    ExtractionResult,
    FieldAnchor,
    FieldStatus,
    ReconciliationPolicy,
    ReconciliationStatus,
    VisionModelId,
    VisionSettings,
)
from multimodal_vision_pipeline.tools.vision_client import VisionClient
from tests.factories import messages_api_body

FIXED_NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
STEPS: tuple[Perturbation, ...] = (
    GaussianBlur(radius=1.5),
    SaltAndPepper(amount=0.05),
    Skew(degrees=7.0),
    ContrastLoss(factor=0.5),
    JpegCompression(quality=10),
)


@pytest.fixture(scope="module")
def document() -> SyntheticDocument:
    return generate_document(0, seed=7)


@pytest.fixture(scope="module")
def truth(document: SyntheticDocument) -> ExtractionResult:
    return document.ground_truth


def _box(x0: float, y0: float, x1: float, y1: float, page: int = 1) -> FieldAnchor:
    return FieldAnchor(page_number=page, bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))


def _with_value[T](field: ExtractedField[T], value: T) -> ExtractedField[T]:
    return field.model_copy(update={"value": value})


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)  # noqa: S311 - reproducible test noise, not crypto


def _darkest(image: Image.Image) -> int:
    """Lowest luminance present in ``image``."""
    return next(level for level, count in enumerate(image.convert("L").histogram()) if count)


def _missing() -> ExtractedField[Any]:
    return ExtractedField[Any](confidence=0.0, status=FieldStatus.MISSING)


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------


class TestPerturbations:
    @pytest.mark.parametrize("mode", ["RGB", "L"])
    @pytest.mark.parametrize("step", STEPS, ids=lambda step: step.kind)
    def test_size_and_mode_preserved_and_pixels_changed(
        self, document: SyntheticDocument, step: PerturbationStep, mode: str
    ) -> None:
        source = document.image.convert(mode)
        out = step.apply(source, _rng(1))
        assert (out.size, out.mode) == (source.size, source.mode)
        assert ImageChops.difference(out, source).getbbox() is not None, "nothing changed"
        assert source.tobytes() == document.image.convert(mode).tobytes(), "input mutated"

    @pytest.mark.parametrize("step", STEPS, ids=lambda step: step.kind)
    def test_deterministic_for_a_seed(
        self, document: SyntheticDocument, step: Perturbation
    ) -> None:
        profile = DegradationProfile(name="one", steps=(step,))
        first = profile.apply(document.image, seed="s").image
        assert first.tobytes() == profile.apply(document.image, seed="s").image.tobytes()

    def test_salt_and_pepper_flips_the_requested_share(self) -> None:
        gray = Image.new("L", (400, 400), 128)
        noisy = SaltAndPepper(amount=0.1).apply(gray, _rng(3))
        histogram = noisy.histogram()
        pepper, salt = histogram[0], histogram[255]
        total = 400 * 400
        assert sum(histogram[1:255]) == histogram[128], "only pure black/white is introduced"
        # round(0.1 * 128) = 13 of 256 byte values per side: 5.08 % each, 10.2 % in all.
        assert pepper / total == pytest.approx(13 / 256, abs=0.002)
        assert salt / total == pytest.approx(13 / 256, abs=0.002)
        other = SaltAndPepper(amount=0.1).apply(gray, _rng(4))
        assert other.tobytes() != noisy.tobytes(), "noise follows the rng"

    def test_blur_spreads_ink(self) -> None:
        dot = Image.new("L", (64, 64), 255)
        dot.putpixel((32, 32), 0)
        blurred = GaussianBlur(radius=2.0).apply(dot, _rng(0))
        assert sum(blurred.histogram()[:255]) > 1

    def test_contrast_loss_flattens_the_histogram(self, document: SyntheticDocument) -> None:
        source = document.image.convert("L")
        faded = ContrastLoss(factor=0.3).apply(source, _rng(0))
        assert ImageStat.Stat(faded).stddev[0] < ImageStat.Stat(source).stddev[0] * 0.5
        assert _darkest(faded) > 0, "black ink turns gray"

    def test_lower_jpeg_quality_means_more_artifacts(self, document: SyntheticDocument) -> None:
        def error(quality: int) -> float:
            out = JpegCompression(quality=quality).apply(document.image, _rng(0))
            return sum(ImageStat.Stat(ImageChops.difference(out, document.image)).mean)

        assert error(5) > error(50) > error(95)

    @pytest.mark.parametrize("degrees", [-15.0, -7.0, 3.0, 15.0])
    def test_skew_moves_boxes_with_the_pixels(self, degrees: float) -> None:
        width, height = PAGE_SIZE
        page = Image.new("L", PAGE_SIZE, 255)
        ImageDraw.Draw(page).rectangle((150, 200, 170, 220), fill=0)
        box = BoundingBox(x0=150 / width, y0=200 / height, x1=171 / width, y1=221 / height)
        step = Skew(degrees=degrees)
        rotated = step.apply(page, _rng(0))
        moved = step.box_transform(PAGE_SIZE)(box)
        assert moved is not None
        ink = rotated.point([255 if value < 128 else 0 for value in range(256)]).getbbox()
        assert ink is not None
        left, top, right, bottom = ink
        center = ((left + right) / 2 / width, (top + bottom) / 2 / height)
        assert center[0] == pytest.approx((moved.x0 + moved.x1) / 2, abs=2 / width)
        assert center[1] == pytest.approx((moved.y0 + moved.y1) / 2, abs=2 / height)
        assert moved.x0 <= left / width + 1e-3
        assert right / width <= moved.x1 + 2e-3

    def test_box_rotated_off_the_page_is_dropped(self) -> None:
        corner = BoundingBox(x0=0.0, y0=0.0, x1=0.01, y1=0.01)
        assert Skew(degrees=15.0).box_transform(PAGE_SIZE)(corner) is None
        profile = DegradationProfile(name="two", steps=(Skew(degrees=15.0), Skew(degrees=1.0)))
        page = profile.apply(Image.new("RGB", PAGE_SIZE, "white"), seed=0)
        assert page.transform_box(corner) is None

    def test_photometric_steps_keep_boxes(self) -> None:
        box = BoundingBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2)
        for step in STEPS:
            if not isinstance(step, Skew):
                assert step.box_transform(PAGE_SIZE)(box) == box

    @pytest.mark.parametrize(
        "build",
        [
            lambda: Skew(degrees=15.5),
            lambda: Skew(degrees=-16.0),
            lambda: GaussianBlur(radius=0.0),
            lambda: SaltAndPepper(amount=0.6),
            lambda: ContrastLoss(factor=1.5),
            lambda: JpegCompression(quality=0),
            lambda: JpegCompression(quality=100),
            lambda: DegradationProfile(name="Bad Name"),
        ],
    )
    def test_parameters_are_bounded(self, build: Callable[[], object]) -> None:
        with pytest.raises(ValidationError):
            build()

    def test_unsupported_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="RGBA"):
            PROFILES["clean"].apply(Image.new("RGBA", (10, 10)), seed=0)

    def test_profiles_round_trip_through_json(self) -> None:
        for profile in PROFILES.values():
            assert DegradationProfile.model_validate_json(profile.model_dump_json()) == profile
            for step in profile.steps:
                if isinstance(step, Skew):
                    assert abs(step.degrees) <= 15
        assert next(iter(PROFILES)) == "clean"
        assert PROFILES["clean"].steps == ()


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_deterministic_per_seed_and_index(self, document: SyntheticDocument) -> None:
        again = generate_document(0, seed=7)
        assert again.ground_truth == document.ground_truth
        assert again.image.tobytes() == document.image.tobytes()
        assert generate_document(1, seed=7).ground_truth != document.ground_truth
        assert generate_document(0, seed=8).ground_truth != document.ground_truth

    def test_labels_reconcile_exactly(self) -> None:
        for sample in generate_dataset(25, seed=3):
            report = reconcile(sample.ground_truth, ReconciliationPolicy(tolerance_cents=0))
            assert report.status is ReconciliationStatus.PASSED, sample.document_id

    def test_every_anchor_frames_drawn_ink(self) -> None:
        for sample in generate_dataset(5, seed=1):
            width, height = sample.image.size
            anchors = list(sample.ground_truth.anchors())
            assert len(anchors) >= 9
            for anchor in anchors:
                box = anchor.bbox
                crop = sample.image.convert("L").crop(
                    (box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)
                )
                assert _darkest(crop) < 128, f"no ink inside {box}"

    def test_labels_only_use_glyphs_the_font_draws(self) -> None:
        printable = set(string.printable)
        for sample in generate_dataset(25, seed=5):
            data = sample.ground_truth.model_dump(mode="json")
            texts = [
                data["vendor"]["name"]["value"],
                data["vendor"]["tax_id"]["value"],
                data["vendor"]["address"]["value"],
                data["document_number"]["value"],
                *(item["description"] for item in data["line_items"]),
            ]
            for text in texts:
                assert set(text) <= printable, text

    def test_discount_is_sometimes_printed(self) -> None:
        statuses = {sample.ground_truth.discount.status for sample in generate_dataset(30)}
        assert statuses == {FieldStatus.EXTRACTED, FieldStatus.MISSING}

    def test_dataset_is_lazy_and_sized(self) -> None:
        dataset = generate_dataset(3, seed=2)
        assert not isinstance(dataset, list)
        assert [sample.document_id for sample in dataset] == [
            "synthetic-2-00000",
            "synthetic-2-00001",
            "synthetic-2-00002",
        ]
        with pytest.raises(ValueError, match="count"):
            generate_dataset(0)

    def test_number_formatting(self) -> None:
        assert format_cents(123456789) == "1,234,567.89"
        assert format_cents(5) == "0.05"
        assert format_quantity(Decimal("2.500")) == "2.5"
        assert format_rate(Decimal("23.00")) == "23%"


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------


class TestPrimitiveMetrics:
    @pytest.mark.parametrize(
        ("a", "b", "distance"),
        [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("", "abc", 3),
            ("abc", "", 3),
            ("INV-2026-000123", "INV-2026-000123", 0),
            ("INV-2026-000123", "INV-2026-00O123", 1),
            ("EUR", "USD", 3),
        ],
    )
    def test_levenshtein(self, a: str, b: str, distance: int) -> None:
        assert levenshtein(a, b) == distance
        assert levenshtein(b, a) == distance

    def test_similarity_and_empty_strings(self) -> None:
        assert similarity("", "") == 1.0
        assert similarity("abcd", "abcf") == 0.75
        assert similarity("abc", "") == 0.0

    def test_normalization(self) -> None:
        assert normalize_text("  ACME\tSp.  z o.o. ") == "acme sp. z o.o."
        assert normalize_text("\uff29\uff2e\uff36\uff0d\uff11")  # full-width "INV-1" == "inv-1"

    def test_iou(self) -> None:
        a = _box(0.0, 0.0, 0.2, 0.2)
        assert iou(a, a) == pytest.approx(1.0)
        assert iou(a, _box(0.1, 0.0, 0.3, 0.2)) == pytest.approx(1 / 3)
        assert iou(a, _box(0.2, 0.0, 0.4, 0.2)) == 0.0, "touching edges do not overlap"
        assert iou(a, _box(0.5, 0.5, 0.6, 0.6)) == 0.0
        assert iou(a, _box(0.0, 0.0, 0.2, 0.2, page=2)) == 0.0
        assert iou(a, _box(0.05, 0.05, 0.15, 0.15)) == pytest.approx(0.25)

    def test_percent_delta_and_zero_division(self) -> None:
        assert percent_delta(1000, 1100) == pytest.approx(10.0)
        assert percent_delta(1000, 900) == pytest.approx(10.0)
        assert percent_delta(0, 0) == 0.0
        assert percent_delta(0, 5) is None

    def test_match_counts_zero_division(self) -> None:
        empty = MatchCounts()
        assert (empty.precision, empty.recall, empty.f1) == (None, None, None)
        wrong = MatchCounts(false_positives=1, false_negatives=1)
        assert (wrong.precision, wrong.recall, wrong.f1) == (0.0, 0.0, 0.0)
        mixed = MatchCounts(true_positives=3, false_positives=1, false_negatives=2)
        assert mixed.precision == pytest.approx(0.75)
        assert mixed.recall == pytest.approx(0.6)
        assert mixed.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
        assert mixed + wrong == MatchCounts(true_positives=3, false_positives=2, false_negatives=3)


# ---------------------------------------------------------------------------
# Document evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_perfect_extraction(self, truth: ExtractionResult) -> None:
        result = evaluate(truth, truth)
        slots = 10 - (truth.discount.status is FieldStatus.MISSING) + len(truth.line_items)
        slots += len(truth.tax_breakdown)
        assert result.counts == MatchCounts(true_positives=slots)
        assert result.counts.f1 == 1.0
        assert result.exact_match_rate == 1.0
        assert all(score.distance == 0 for score in result.text_fields)
        assert all(score.abs_delta_cents == 0 for score in result.amounts)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.anchor_accuracy == 1.0
        assert result.hallucination_rate == 0.0
        assert result.line_items_matched == len(truth.line_items)

    def test_known_field_errors(self, truth: ExtractionResult) -> None:
        name = truth.vendor.name.value
        assert name is not None
        total = truth.total.value
        assert total is not None
        wrong = truth.model_copy(
            update={
                "vendor": truth.vendor.model_copy(
                    update={"name": _with_value(truth.vendor.name, name[:-1] + "#")}
                ),
                "document_number": _with_value(
                    truth.document_number, "  " + str(truth.document_number.value).lower() + " "
                ),
                "currency": _missing(),
                "total": _with_value(truth.total, total + 150),
            }
        )
        result = evaluate(truth, wrong)
        scores = {score.field: score for score in result.text_fields}
        assert scores["vendor.name"].exact_match is False
        assert scores["vendor.name"].distance == 1
        assert scores["document_number"].exact_match is True, "EM is normalized"
        assert scores["currency"].actual is None
        assert scores["currency"].distance == 3
        assert scores["currency"].similarity == 0.0
        amount = next(score for score in result.amounts if score.field == "total")
        assert amount.abs_delta_cents == 150
        assert amount.pct_delta == pytest.approx(150 / total * 100)
        perfect = evaluate(truth, truth).counts
        # name and total wrong (FP+FN each); currency missing (FN only).
        assert result.counts == MatchCounts(
            true_positives=perfect.true_positives - 3, false_positives=2, false_negatives=3
        )

    def test_completely_empty_extraction(self, truth: ExtractionResult) -> None:
        result = evaluate(truth, empty_extraction())
        assert result.counts.true_positives == result.counts.false_positives == 0
        assert result.counts.precision is None, "no predictions: precision is undefined"
        assert result.counts.recall == 0.0
        assert result.counts.f1 is None
        assert result.exact_match_rate == 0.0
        for score in result.text_fields:
            assert score.actual is None
            assert score.distance == len(normalize_text(score.expected or ""))
        assert result.extracted_slots == 0
        assert result.hallucination_rate is None
        assert result.anchors == ()
        assert result.mean_iou is None
        assert result.line_items_matched == 0
        assert all(score.abs_delta_cents is None for score in result.amounts)

    def test_empty_against_empty_is_vacuous(self) -> None:
        result = evaluate(empty_extraction(), empty_extraction())
        assert result.counts == MatchCounts()
        assert result.text_fields == ()
        assert result.amounts == ()
        assert result.exact_match_rate is None

    def test_mismatched_line_item_count(self, truth: ExtractionResult) -> None:
        items = list(truth.line_items)
        assert len(items) >= 2
        dropped = items.pop(0)
        spurious = dropped.model_copy(
            update={
                "description": "Unrelated freight surcharge",
                "amount_cents": 7,
                "unit_price_cents": 7,
            }
        )
        extracted = truth.model_copy(update={"line_items": (*items, spurious, spurious)})
        result = evaluate(truth, extracted)
        assert result.line_items_expected == len(truth.line_items)
        assert result.line_items_extracted == len(items) + 2
        assert result.line_items_matched == len(items)
        assert "line_items/0" not in result.hallucinated
        spurious_names = {
            f"extracted_line_items/{len(items)}",
            f"extracted_line_items/{len(items) + 1}",
        }
        assert spurious_names <= set(result.hallucinated)
        missed = next(
            score for score in result.amounts if score.field == "line_items/0/amount_cents"
        )
        assert (missed.actual_cents, missed.abs_delta_cents) == (None, None)
        perfect = evaluate(truth, truth).counts
        assert result.counts == MatchCounts(
            true_positives=perfect.true_positives - 1, false_positives=2, false_negatives=1
        )

    def test_line_item_matching_ignores_order(self, truth: ExtractionResult) -> None:
        reordered = truth.model_copy(update={"line_items": tuple(reversed(truth.line_items))})
        result = evaluate(truth, reordered)
        assert result.counts == evaluate(truth, truth).counts
        assert result.line_items_matched == len(truth.line_items)
        pairs = match_line_items(truth.line_items, reordered.line_items)
        n = len(truth.line_items)
        for i, j in pairs:
            assert truth.line_items[i] == reordered.line_items[j]
        assert sorted(j for _, j in pairs) == list(range(n))

    def test_wrong_amount_on_a_matched_row(self, truth: ExtractionResult) -> None:
        first = truth.line_items[0]
        changed = first.model_copy(update={"amount_cents": first.amount_cents + 1})
        result = evaluate(
            truth, truth.model_copy(update={"line_items": (changed, *truth.line_items[1:])})
        )
        assert result.line_items_matched == len(truth.line_items)
        score = next(s for s in result.amounts if s.field == "line_items/0/amount_cents")
        assert score.abs_delta_cents == 1
        assert result.counts.false_positives == result.counts.false_negatives == 1

    def test_tax_lines_pair_by_rate(self, truth: ExtractionResult) -> None:
        first = truth.tax_breakdown[0]
        wrong = first.model_copy(update={"tax_cents": first.tax_cents + 10})
        extra = first.model_copy(update={"rate": Decimal("99")})
        extracted = truth.model_copy(
            update={"tax_breakdown": (*truth.tax_breakdown[1:], wrong, extra)}
        )
        result = evaluate(truth, extracted)
        score = next(s for s in result.amounts if s.field == "tax_breakdown/0/tax_cents")
        assert score.abs_delta_cents == 10
        assert any(name.startswith("extracted_tax_breakdown/") for name in result.hallucinated)

    def test_hallucination_rules(self, truth: ExtractionResult) -> None:
        far = _box(0.9, 0.95, 0.99, 0.99)
        invented_discount = ExtractedField[int](
            value=500, confidence=0.9, status=FieldStatus.EXTRACTED, anchor=far
        )
        extracted = truth.model_copy(
            update={
                # value correct, but read from the wrong place
                "subtotal": truth.subtotal.model_copy(update={"anchor": far}),
                # value correct, claims to be read, no anchor at all
                "tax_total": truth.tax_total.model_copy(update={"anchor": None}),
                # inferred values need no anchor
                "currency": truth.currency.model_copy(
                    update={"status": FieldStatus.INFERRED, "anchor": None}
                ),
                # nothing printed, something extracted
                "discount": invented_discount
                if truth.discount.status is FieldStatus.MISSING
                else truth.discount,
            }
        )
        result = evaluate(truth, extracted)
        assert {"subtotal", "tax_total"} <= set(result.hallucinated)
        assert "currency" not in result.hallucinated
        if truth.discount.status is FieldStatus.MISSING:
            assert "discount" in result.hallucinated
        ious = {score.field: score.iou for score in result.anchors}
        assert ious["subtotal"] == 0.0
        assert ious["tax_total"] == 0.0
        assert result.hallucination_rate == pytest.approx(
            len(result.hallucinated) / result.extracted_slots
        )

    def test_anchor_threshold_drives_accuracy(self, truth: ExtractionResult) -> None:
        anchor = truth.total.anchor
        assert anchor is not None
        box = anchor.bbox
        half = (box.x1 - box.x0) / 2
        shifted = anchor.model_copy(
            update={"bbox": box.model_copy(update={"x0": box.x0 + half, "x1": box.x1 + half})}
        )
        extracted = truth.model_copy(
            update={"total": truth.total.model_copy(update={"anchor": shifted})}
        )
        loose = evaluate(truth, extracted, iou_threshold=0.3)
        strict = evaluate(truth, extracted, iou_threshold=0.5)
        overlap = next(score.iou for score in loose.anchors if score.field == "total")
        assert overlap == pytest.approx(1 / 3, abs=1e-6)
        assert loose.anchor_accuracy == 1.0
        assert strict.anchor_accuracy is not None
        assert strict.anchor_accuracy < 1.0
        assert "total" not in strict.hallucinated, "partial overlap is a localization error"

    def test_summary_is_micro_averaged(self, truth: ExtractionResult) -> None:
        perfect = evaluate(truth, truth)
        empty = evaluate(truth, empty_extraction())
        summary = summarize([perfect, empty])
        assert summary.documents == 2
        assert summary.counts == perfect.counts + empty.counts
        assert summary.recall == pytest.approx(0.5)
        assert summary.precision == 1.0
        assert summary.exact_match_rate == pytest.approx(0.5)
        assert summary.hallucination_rate == 0.0
        assert summary.line_item_recall == pytest.approx(0.5)
        assert summary.amount_exact_rate == 1.0, "missing amounts have no delta"

    def test_summary_of_nothing(self) -> None:
        summary = summarize([])
        assert summary.documents == 0
        assert summary.precision is summary.recall is summary.f1 is None
        assert summary.mean_iou is summary.hallucination_rate is summary.mean_pct_delta is None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class Oracle:
    """Returns the undegraded labels, optionally corrupted; records concurrency."""

    def __init__(
        self,
        corpus: Sequence[SyntheticDocument],
        *,
        corrupt: Callable[[ExtractionResult], ExtractionResult] | None = None,
        fail: Callable[[str], Exception | None] | None = None,
    ) -> None:
        self.truth = {doc.document_id: doc.ground_truth for doc in corpus}
        self.corrupt = corrupt
        self.fail = fail
        self.active = 0
        self.peak = 0
        self.pages: list[RasterizedPage] = []

    async def extract_document(
        self, document_id: str, pages: Sequence[RasterizedPage], *, page_count: int
    ) -> ExtractionResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0)
            self.pages.extend(pages)
            if self.fail is not None and (error := self.fail(document_id)) is not None:
                raise error
            result = self.truth[document_id]
            return result if self.corrupt is None else self.corrupt(result)
        finally:
            self.active -= 1


class Rendezvous(Oracle):
    """Holds each call until ``meet`` calls are in flight at once (proves parallelism)."""

    def __init__(self, corpus: Sequence[SyntheticDocument], *, meet: int) -> None:
        super().__init__(corpus)
        self.meet = meet
        self.met = asyncio.Event()

    async def extract_document(
        self, document_id: str, pages: Sequence[RasterizedPage], *, page_count: int
    ) -> ExtractionResult:
        if self.active + 1 >= self.meet:
            self.met.set()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.wait_for(self.met.wait(), timeout=5)
            return self.truth[document_id]
        finally:
            self.active -= 1


def _ticking_clock(step: float = 0.25) -> Callable[[], float]:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


@pytest.fixture(scope="module")
def corpus() -> list[SyntheticDocument]:
    return list(generate_dataset(3, seed=11))


async def _run(
    oracle: Oracle, corpus: Sequence[SyntheticDocument], **kwargs: Any
) -> BenchmarkReport:
    kwargs.setdefault("profiles", [PROFILES["clean"], PROFILES["fax"]])
    return await run_benchmark(
        oracle, corpus, clock=_ticking_clock(), now=lambda: FIXED_NOW, **kwargs
    )


class TestRunner:
    async def test_oracle_scores_perfectly_on_clean_pages(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        oracle = Oracle(corpus)
        # One at a time: each sample reads the shared fake clock exactly twice.
        report = await _run(oracle, corpus, max_concurrency=1)
        assert report.documents == 3
        assert len(report.samples) == 6
        assert [summary.profile for summary in report.summaries] == ["clean", "fax"]
        clean, fax = report.summaries
        for summary in (clean, fax):
            assert summary.documents == 3
            assert summary.failures == 0
            assert summary.accuracy.precision == summary.accuracy.recall == 1.0
            assert summary.discrepancy_rate == 0.0
            assert summary.latency.p50_ms == pytest.approx(250.0)
        assert clean.accuracy.anchor_accuracy == 1.0
        assert clean.accuracy.hallucination_rate == 0.0
        # Undegraded coordinates on a skewed page miss the moved labels.
        assert fax.accuracy.mean_iou is not None
        assert fax.accuracy.mean_iou < 0.5
        assert report.generated_at == FIXED_NOW

    async def test_extractor_receives_degraded_png_pages(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        oracle = Oracle(corpus)
        await _run(oracle, corpus[:1])
        clean, fax = oracle.pages
        for page in (clean, fax):
            assert page.png.startswith(b"\x89PNG")
            assert (page.metadata.width_px, page.metadata.height_px) == PAGE_SIZE
            assert page.metadata.page_number == 1
        assert clean.png != fax.png

    async def test_degraded_labels_follow_the_skew(self, corpus: list[SyntheticDocument]) -> None:
        doc = corpus[0]
        _, labels = prepare_sample(doc, PROFILES["fax"], seed=0)
        original = list(doc.ground_truth.anchors())
        moved = list(labels.anchors())
        assert len(moved) == len(original)
        skew = Skew(degrees=7.0).box_transform(PAGE_SIZE)
        assert [anchor.bbox for anchor in moved] == [skew(anchor.bbox) for anchor in original]
        identity = degrade_labels(doc.ground_truth, lambda box: box)
        assert identity == doc.ground_truth

    async def test_seeded_runs_are_reproducible(self, corpus: list[SyntheticDocument]) -> None:
        first = Oracle(corpus)
        second = Oracle(corpus)
        await _run(first, corpus[:1], seed=5)
        await _run(second, corpus[:1], seed=5)
        assert [page.png for page in first.pages] == [page.png for page in second.pages]

    async def test_expected_failures_are_scored_not_raised(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        bad = corpus[1].document_id
        oracle = Oracle(
            corpus,
            fail=lambda doc_id: ExtractionParseError("schema violation") if doc_id == bad else None,
        )
        report = await _run(oracle, corpus, profiles=[PROFILES["clean"]])
        (summary,) = report.summaries
        assert summary.failures == 1
        failed = next(sample for sample in report.samples if sample.document_id == bad)
        assert failed.error == "ExtractionParseError"
        assert failed.reconciliation is ReconciliationStatus.SKIPPED
        assert failed.evaluation.counts.true_positives == 0
        assert summary.accuracy.recall is not None
        assert 0.5 < summary.accuracy.recall < 0.8
        assert summary.accuracy.precision == 1.0
        assert summary.discrepancy_rate == 0.0, "failed documents are not counted as reconciled"

    async def test_unexpected_errors_abort_the_run(self, corpus: list[SyntheticDocument]) -> None:
        oracle = Oracle(corpus, fail=lambda _: RuntimeError("bug"))
        with pytest.raises(ExceptionGroup) as raised:
            await _run(oracle, corpus)
        assert raised.group_contains(RuntimeError)

    async def test_arithmetic_errors_raise_the_discrepancy_rate(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        def inflate(result: ExtractionResult) -> ExtractionResult:
            total = result.total.value
            assert total is not None
            return result.model_copy(
                update={"total": result.total.model_copy(update={"value": total + 500})}
            )

        bad = corpus[0].document_id
        oracle = Oracle(
            corpus,
            corrupt=inflate,
            fail=lambda doc_id: ExtractionParseError("x") if doc_id == bad else None,
        )
        report = await _run(oracle, corpus, profiles=[PROFILES["clean"]])
        (summary,) = report.summaries
        assert summary.discrepancy_rate == 1.0, "2 of the 2 extracted; the failure is excluded"
        assert all(s.discrepancies >= 1 for s in report.samples if s.error is None)
        assert summary.accuracy.mean_abs_delta_cents is not None
        assert summary.accuracy.amount_exact_rate is not None
        assert summary.accuracy.amount_exact_rate < 1.0

    async def test_concurrency_is_bounded(self, corpus: list[SyntheticDocument]) -> None:
        serial = Oracle(corpus)
        await _run(serial, corpus, max_concurrency=1)
        assert serial.peak == 1
        parallel = Rendezvous(corpus, meet=2)
        await _run(parallel, corpus, max_concurrency=2)
        assert parallel.peak == 2, "two samples were in flight together, never three"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_concurrency": 0}, "max_concurrency"),
            ({"profiles": []}, "at least one"),
            ({"profiles": [PROFILES["clean"], PROFILES["clean"]]}, "unique"),
        ],
    )
    async def test_invalid_runs_are_rejected(
        self, corpus: list[SyntheticDocument], kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            await _run(Oracle(corpus), corpus, **kwargs)

    async def test_reports_render_as_json_and_markdown(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        report = await _run(Oracle(corpus), corpus[:2])
        assert BenchmarkReport.model_validate_json(report.to_json()) == report
        parsed = json.loads(report.to_json())
        assert {"summaries", "samples", "profiles"} <= parsed.keys()
        markdown = report.to_markdown()
        rows = [line for line in markdown.splitlines() if line.startswith("|")]
        assert len(rows) == 2 + len(report.summaries)
        assert len({row.count("|") for row in rows}) == 1, "every row has the same columns"
        assert "| clean | 2 | 0 | 100.0% |" in markdown

    async def test_undefined_metrics_render_as_na(self, corpus: list[SyntheticDocument]) -> None:
        oracle = Oracle(corpus, fail=lambda _: ExtractionParseError("x"))
        report = await _run(oracle, corpus[:1], profiles=[PROFILES["clean"]])
        markdown = report.to_markdown()
        assert "n/a" in markdown
        (summary,) = report.summaries
        assert summary.accuracy.precision is None
        assert summary.discrepancy_rate is None

    def test_percentile_nearest_rank(self) -> None:
        values = [float(v) for v in range(10, 0, -1)]
        assert percentile(values, 50) == 5.0
        assert percentile(values, 95) == 10.0
        assert percentile(values, 0) == 1.0
        assert percentile(values, 33) == 4.0, "ceil(3.3): nearest rank rounds up"
        assert percentile([float(v) for v in range(1, 7)], 75) == 5.0, "ceil(4.5), not round"
        assert percentile([4.0], 95) == 4.0
        assert percentile([], 50) is None


class TestVisionPipelineExtractor:
    async def test_production_path_through_a_mocked_api(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        doc = corpus[0]
        reply = doc.ground_truth.model_dump(mode="json", exclude={"model", "page_count"})
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=messages_api_body(json.dumps(reply)))

        async with VisionClient(
            api_key=SecretStr("sk-ant-test-key-000000"),
            transport=httpx.MockTransport(handler),
            sleep=lambda _: asyncio.sleep(0),
            rng=_rng(0),
        ) as client:
            extractor = VisionPipelineExtractor(client)
            report = await run_benchmark(
                extractor, [doc], profiles=[PROFILES["clean"]], now=lambda: FIXED_NOW
            )
        (summary,) = report.summaries
        assert summary.failures == 0
        assert summary.accuracy.f1 == 1.0
        assert summary.accuracy.anchor_accuracy == 1.0
        (body,) = seen
        assert body["model"] == VisionModelId.SONNET_5_5.value
        images = [block for block in body["messages"][0]["content"] if block["type"] == "image"]
        assert len(images) == 1

    async def test_unparseable_reply_is_a_recorded_failure(
        self, corpus: list[SyntheticDocument]
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=messages_api_body('{"not": "an extraction"}'))

        async with VisionClient(
            api_key=SecretStr("sk-ant-test-key-000000"),
            transport=httpx.MockTransport(handler),
            sleep=lambda _: asyncio.sleep(0),
            rng=_rng(0),
        ) as client:
            report = await run_benchmark(
                VisionPipelineExtractor(client), corpus[:1], profiles=[PROFILES["clean"]]
            )
        (sample,) = report.samples
        assert sample.error == "ExtractionParseError"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _oracle_factory(corpus_seed: int, documents: int) -> cli.ExtractorFactory:
    @asynccontextmanager
    async def factory(
        settings: VisionSettings, model: VisionModelId | None
    ) -> AsyncIterator[Oracle]:
        yield Oracle(list(generate_dataset(documents, seed=corpus_seed)))

    return factory


SETTINGS = VisionSettings(anthropic_api_key=SecretStr("sk-ant-test-key-000000"))


class TestCli:
    def test_writes_json_and_markdown_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(
            [
                "--documents",
                "2",
                "--seed",
                "4",
                "--profiles",
                "clean, office_scan",
                "--output-dir",
                str(tmp_path),
            ],
            settings=SETTINGS,
            extractor_factory=_oracle_factory(4, 2),
        )
        assert code == 0
        report = BenchmarkReport.model_validate_json((tmp_path / "benchmark.json").read_text())
        assert [s.profile for s in report.summaries] == ["clean", "office_scan"]
        assert report.seed == 4
        markdown = (tmp_path / "benchmark.md").read_text()
        out, err = capsys.readouterr()
        assert out == markdown
        assert str(tmp_path) in err

    @pytest.mark.parametrize(
        "argv",
        [
            ["--profiles", "clean,blurry"],
            ["--profiles", " , "],
            ["--documents", "0"],
            ["--concurrency", "0"],
            ["--model", "gpt-vision"],
        ],
    )
    def test_rejects_bad_arguments(self, argv: list[str], tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exited:
            cli.main(
                [*argv, "--output-dir", str(tmp_path)],
                settings=SETTINGS,
                extractor_factory=_oracle_factory(0, 1),
            )
        assert exited.value.code == 2
        assert not list(tmp_path.iterdir())

    async def test_default_factory_builds_the_production_extractor(self) -> None:
        async with cli.vision_extractor(SETTINGS, VisionModelId.HAIKU_5_5) as extractor:
            assert isinstance(extractor, VisionPipelineExtractor)
            assert extractor.config.model is VisionModelId.HAIKU_5_5
        async with cli.vision_extractor(SETTINGS, None) as extractor:
            assert isinstance(extractor, VisionPipelineExtractor)
            assert extractor.config == SETTINGS.extraction_config()

    async def test_default_factory_requires_an_api_key(self) -> None:
        with pytest.raises(ValueError, match="MVP_ANTHROPIC_API_KEY"):
            async with cli.vision_extractor(VisionSettings(anthropic_api_key=None), None):
                pass

    def test_module_entry_point_imports_without_running(self) -> None:
        module = importlib.import_module("multimodal_vision_pipeline.evals.__main__")
        assert module.main is cli.main
        assert "--profiles" in cli.build_parser().format_help()
