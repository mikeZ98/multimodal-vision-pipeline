"""Benchmark runner: every document under every degradation profile, scored and summarized.

Per sample: degrade the page (seeded), move the labelled anchors with it, send the PNG
through an extractor, score against the labels and reconcile the extraction. Expected
failures (``PipelineError``: refusals, parse errors, upstream errors) are recorded and
scored as an empty extraction, so they lower recall instead of aborting the run.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Final, Protocol, override

from pydantic import Field

from multimodal_vision_pipeline.core.errors import PipelineError
from multimodal_vision_pipeline.core.rasterizer import RasterizedPage, encode_png
from multimodal_vision_pipeline.core.reconciler import reconcile
from multimodal_vision_pipeline.core.vision_builder import VisionPayloadBuilder, parse_extraction
from multimodal_vision_pipeline.evals.dataset import SyntheticDocument
from multimodal_vision_pipeline.evals.evaluator import (
    DEFAULT_IOU_THRESHOLD,
    AccuracySummary,
    DocumentEvaluation,
    empty_extraction,
    evaluate,
    ratio,
    summarize,
)
from multimodal_vision_pipeline.evals.perturbation import (
    PROFILES,
    BoxTransform,
    DegradationProfile,
)
from multimodal_vision_pipeline.schemas import (
    ColorMode,
    DocumentKind,
    ExtractedField,
    ExtractionModelConfig,
    ExtractionResult,
    FieldAnchor,
    PageMetadata,
    PipelineModel,
    ReconciliationPolicy,
    ReconciliationStatus,
    UtcDatetime,
    utc_now,
)
from multimodal_vision_pipeline.tools.pipeline_worker import VisionExtractor

logger = logging.getLogger(__name__)

DEFAULT_PROFILES: Final = tuple(PROFILES.values())


class BenchmarkExtractor(Protocol):
    """Anything that turns rendered pages into an extraction. Never sees the labels."""

    async def extract_document(
        self, document_id: str, pages: Sequence[RasterizedPage], *, page_count: int
    ) -> ExtractionResult: ...


class VisionPipelineExtractor(BenchmarkExtractor):
    """The production path: payload builder, vision API client, strict parsing."""

    def __init__(
        self,
        extractor: VisionExtractor,
        *,
        config: ExtractionModelConfig | None = None,
        builder: VisionPayloadBuilder | None = None,
        document_kind: DocumentKind | None = DocumentKind.INVOICE,
    ) -> None:
        self.extractor = extractor
        self.config = config if config is not None else ExtractionModelConfig()
        self.builder = builder if builder is not None else VisionPayloadBuilder()
        self.document_kind = document_kind

    @override
    async def extract_document(
        self, document_id: str, pages: Sequence[RasterizedPage], *, page_count: int
    ) -> ExtractionResult:
        request = await asyncio.to_thread(
            self.builder.build,
            pages,
            self.config,
            page_count=page_count,
            document_kind=self.document_kind,
        )
        response = await self.extractor.extract(request)
        return parse_extraction(response.text, request)


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class SampleResult(PipelineModel):
    document_id: str
    profile: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    error: str | None = None
    """Exception type of an expected extraction failure (no message: it may echo content)."""
    evaluation: DocumentEvaluation
    reconciliation: ReconciliationStatus
    discrepancies: int = Field(ge=0)


class LatencyStats(PipelineModel):
    """Extractor wall time per document, nearest-rank percentiles."""

    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Nearest-rank percentile; ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_stats(values: Sequence[float]) -> LatencyStats:
    return LatencyStats(
        mean_ms=ratio(sum(values), len(values)),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        max_ms=max(values, default=None),
    )


class ProfileSummary(PipelineModel):
    profile: str
    documents: int = Field(ge=0)
    failures: int = Field(ge=0)
    accuracy: AccuracySummary
    discrepancy_rate: float | None
    """Share of extracted documents whose arithmetic did not reconcile."""
    latency: LatencyStats


class BenchmarkReport(PipelineModel):
    generated_at: UtcDatetime
    seed: int
    documents: int = Field(ge=0)
    iou_threshold: float
    profiles: tuple[DegradationProfile, ...]
    summaries: tuple[ProfileSummary, ...]
    samples: tuple[SampleResult, ...]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Extraction benchmark",
            "",
            f"Generated {self.generated_at.isoformat()} - {self.documents} synthetic document(s), "
            f"seed {self.seed}, anchor IoU threshold {self.iou_threshold:g}.",
            "",
            "| Profile | Docs | Failures | Precision | Recall | F1 | Text EM | Amount exact "
            "| Mean abs Δ cents | Anchor acc. | Mean IoU | Hallucination | Discrepancy rate "
            "| p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in self.summaries:
            accuracy = summary.accuracy
            cells = [
                summary.profile,
                str(summary.documents),
                str(summary.failures),
                _percent(accuracy.precision),
                _percent(accuracy.recall),
                _percent(accuracy.f1),
                _percent(accuracy.exact_match_rate),
                _percent(accuracy.amount_exact_rate),
                _number(accuracy.mean_abs_delta_cents),
                _percent(accuracy.anchor_accuracy),
                _number(accuracy.mean_iou, digits=3),
                _percent(accuracy.hallucination_rate),
                _percent(summary.discrepancy_rate),
                _number(summary.latency.p50_ms),
                _number(summary.latency.p95_ms),
            ]
            lines.append(f"| {' | '.join(cells)} |")
        lines += [
            "",
            "Precision/recall count field slots (scalars, line items, tax lines); a wrong "
            "value is both a false positive and a false negative. `n/a` means undefined "
            "(empty denominator).",
            "",
        ]
        return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, *, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _move(anchor: FieldAnchor | None, transform: BoxTransform) -> FieldAnchor | None:
    if anchor is None:
        return None
    box = transform(anchor.bbox)
    return None if box is None else anchor.model_copy(update={"bbox": box})


def degrade_labels(truth: ExtractionResult, transform: BoxTransform) -> ExtractionResult:
    """The labels with every anchor mapped into the degraded page."""

    def field[T](value: ExtractedField[T]) -> ExtractedField[T]:
        return value.model_copy(update={"anchor": _move(value.anchor, transform)})

    return truth.model_copy(
        update={
            "vendor": truth.vendor.model_copy(
                update={
                    "name": field(truth.vendor.name),
                    "tax_id": field(truth.vendor.tax_id),
                    "address": field(truth.vendor.address),
                }
            ),
            "document_number": field(truth.document_number),
            "issue_date": field(truth.issue_date),
            "currency": field(truth.currency),
            "line_items": tuple(
                item.model_copy(update={"anchor": _move(item.anchor, transform)})
                for item in truth.line_items
            ),
            "tax_breakdown": tuple(
                line.model_copy(update={"anchor": _move(line.anchor, transform)})
                for line in truth.tax_breakdown
            ),
            "subtotal": field(truth.subtotal),
            "tax_total": field(truth.tax_total),
            "discount": field(truth.discount),
            "total": field(truth.total),
        }
    )


def prepare_sample(
    document: SyntheticDocument, profile: DegradationProfile, *, seed: int
) -> tuple[RasterizedPage, ExtractionResult]:
    """Degraded page as the pipeline would see it, plus labels moved to match."""
    degraded = profile.apply(document.image, seed=f"{seed}:{document.document_id}:{profile.name}")
    width, height = degraded.image.size
    page = RasterizedPage(
        metadata=PageMetadata(
            page_number=1,
            width_px=width,
            height_px=height,
            dpi=document.dpi,
            color_mode=ColorMode.RGB,
        ),
        png=encode_png(degraded.image),
    )
    return page, degrade_labels(document.ground_truth, degraded.transform_box)


async def run_benchmark(
    extractor: BenchmarkExtractor,
    documents: Iterable[SyntheticDocument],
    *,
    profiles: Sequence[DegradationProfile] = DEFAULT_PROFILES,
    seed: int = 0,
    max_concurrency: int = 4,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    policy: ReconciliationPolicy | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = utc_now,
) -> BenchmarkReport:
    """Run every document under every profile, at most ``max_concurrency`` at a time.

    ``documents`` is materialized (the pages are needed once per profile). ``clock`` and
    ``now`` are injectable so tests can pin latency and timestamps.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if not profiles:
        raise ValueError("at least one degradation profile is required")
    names = [profile.name for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError(f"profile names must be unique: {names}")
    corpus = list(documents)
    rules = policy if policy is not None else ReconciliationPolicy()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sample(document: SyntheticDocument, profile: DegradationProfile) -> SampleResult:
        async with semaphore:
            page, labels = await asyncio.to_thread(prepare_sample, document, profile, seed=seed)
            error: str | None = None
            started = clock()
            try:
                extracted = await extractor.extract_document(
                    document.document_id, [page], page_count=labels.page_count
                )
            except PipelineError as exc:
                error = type(exc).__name__
                extracted = empty_extraction(page_count=labels.page_count)
            finally:
                latency_ms = max(0.0, (clock() - started) * 1000)
        report = reconcile(extracted, rules)
        return SampleResult(
            document_id=document.document_id,
            profile=profile.name,
            latency_ms=latency_ms,
            error=error,
            evaluation=evaluate(labels, extracted, iou_threshold=iou_threshold),
            reconciliation=report.status,
            discrepancies=len(report.discrepancies),
        )

    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(sample(document, profile))
            for profile in profiles
            for document in corpus
        ]
    samples = tuple(task.result() for task in tasks)
    summaries = tuple(_summarize(profile.name, samples) for profile in profiles)
    for summary in summaries:
        logger.info(
            "benchmark profile %s finished",
            summary.profile,
            extra={
                "documents": summary.documents,
                "failures": summary.failures,
                "f1": summary.accuracy.f1,
            },
        )
    return BenchmarkReport(
        generated_at=now(),
        seed=seed,
        documents=len(corpus),
        iou_threshold=iou_threshold,
        profiles=tuple(profiles),
        summaries=summaries,
        samples=samples,
    )


def _summarize(profile: str, samples: Sequence[SampleResult]) -> ProfileSummary:
    selected = [sample for sample in samples if sample.profile == profile]
    extracted = [sample for sample in selected if sample.error is None]
    return ProfileSummary(
        profile=profile,
        documents=len(selected),
        failures=len(selected) - len(extracted),
        accuracy=summarize([sample.evaluation for sample in selected]),
        discrepancy_rate=ratio(
            sum(sample.reconciliation is ReconciliationStatus.FAILED for sample in extracted),
            len(extracted),
        ),
        latency=latency_stats([sample.latency_ms for sample in selected]),
    )
