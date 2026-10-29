"""End-to-end job execution: rasterize → build payload → call vision → parse → reconcile.

Each stage boundary is a ``JobManager`` transition. Any failure, a job deadline or a
cancellation ends the job in ``failed`` with an RFC 9457 problem, so a job is never
left stranded in a running state. Concurrency is bounded by a semaphore; jobs
waiting for a slot stay ``queued``. Admission is bounded too: each scheduled job holds
its upload in memory, so at most ``max_queued_jobs`` may be scheduled at once.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, Protocol
from uuid import UUID

from multimodal_vision_pipeline.core.errors import (
    InvalidTransitionError,
    JobCancelledError,
    JobCapacityError,
    JobTimeoutError,
    PipelineError,
)
from multimodal_vision_pipeline.core.job_manager import JobManager
from multimodal_vision_pipeline.core.rasterizer import RasterizedPage, Rasterizer
from multimodal_vision_pipeline.core.reconciler import reconcile
from multimodal_vision_pipeline.core.vision_builder import (
    VisionPayloadBuilder,
    VisionRequest,
    parse_extraction,
)
from multimodal_vision_pipeline.observability.context import bind
from multimodal_vision_pipeline.observability.metrics import PipelineMetrics
from multimodal_vision_pipeline.schemas import (
    ExtractionModelConfig,
    JobStatusResponse,
    VisionSettings,
    job_path,
)
from multimodal_vision_pipeline.tools.vision_client import VisionResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY: Final = 4
DEFAULT_JOB_TIMEOUT_SECONDS: Final = 900.0
DEFAULT_MAX_QUEUED_JOBS: Final = 32


class VisionExtractor(Protocol):
    async def extract(self, request: VisionRequest) -> VisionResponse: ...


class PipelineWorker:
    """Runs jobs through the pipeline, at most ``max_concurrency`` at a time."""

    def __init__(
        self,
        *,
        jobs: JobManager,
        rasterizer: Rasterizer,
        extractor: VisionExtractor,
        builder: VisionPayloadBuilder | None = None,
        default_extraction: ExtractionModelConfig | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        max_queued_jobs: int = DEFAULT_MAX_QUEUED_JOBS,
        metrics: PipelineMetrics | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queued_jobs < max_concurrency:
            raise ValueError("max_queued_jobs must be >= max_concurrency")
        if job_timeout_seconds <= 0:
            raise ValueError("job_timeout_seconds must be > 0")
        self.jobs = jobs
        self.rasterizer = rasterizer
        self.extractor = extractor
        self.builder = builder or VisionPayloadBuilder()
        self.default_extraction = default_extraction or ExtractionModelConfig()
        self.max_concurrency = max_concurrency
        self.job_timeout_seconds = job_timeout_seconds
        self.max_queued_jobs = max_queued_jobs
        self.metrics = metrics if metrics is not None else PipelineMetrics()
        self._slots = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[UUID, asyncio.Task[JobStatusResponse]] = {}
        self._accepting = True

    @classmethod
    def from_settings(
        cls,
        settings: VisionSettings,
        *,
        jobs: JobManager,
        rasterizer: Rasterizer,
        extractor: VisionExtractor,
        metrics: PipelineMetrics | None = None,
    ) -> "PipelineWorker":
        return cls(
            jobs=jobs,
            rasterizer=rasterizer,
            extractor=extractor,
            default_extraction=settings.extraction_config(),
            max_concurrency=settings.worker_max_concurrency,
            job_timeout_seconds=settings.worker_job_timeout_seconds,
            max_queued_jobs=max(settings.worker_max_queued_jobs, settings.worker_max_concurrency),
            metrics=metrics,
        )

    # -- background execution -------------------------------------------------
    @property
    def active_jobs(self) -> frozenset[UUID]:
        return frozenset(self._tasks)

    @property
    def accepting(self) -> bool:
        return self._accepting

    def ensure_capacity(self) -> None:
        """Raise ``JobCapacityError`` unless another job could be scheduled right now.

        Callers that create a job first should check this before creating it, with no
        ``await`` in between, so a rejected request never leaves an orphaned job.
        """
        if not self._accepting:
            raise JobCapacityError("the worker is shutting down")
        if len(self._tasks) >= self.max_queued_jobs:
            raise JobCapacityError(f"{self.max_queued_jobs} jobs are already scheduled")

    def submit(self, job_id: UUID, document: bytes) -> asyncio.Task[JobStatusResponse]:
        """Schedule a queued job; it starts once a concurrency slot is free."""
        if job_id in self._tasks:
            raise InvalidTransitionError(f"job {job_id} is already scheduled")
        self.ensure_capacity()
        task = asyncio.create_task(self._run_bounded(job_id, document), name=f"job-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return task

    async def _run_bounded(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        with bind(job_id=str(job_id)):
            try:
                async with self._slots:
                    return await self.process(job_id, document)
            except asyncio.CancelledError:
                # Cancelled while waiting for a slot: the job never left ``queued``.
                self._fail(job_id, JobCancelledError("cancelled before processing started"))
                raise

    async def drain(self, grace_seconds: float) -> None:
        """Stop admitting jobs, let scheduled ones finish for up to ``grace_seconds``,
        then cancel whatever is still running."""
        self._accepting = False
        tasks = list(self._tasks.values())
        if tasks and grace_seconds > 0:
            await asyncio.wait(tasks, timeout=grace_seconds)
        await self.shutdown()

    async def shutdown(self) -> None:
        """Cancel every scheduled job and wait until each has reached a terminal state."""
        self._accepting = False
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- one job ------------------------------------------------------------
    async def process(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        """Run one queued job to a terminal state and return its final snapshot.

        Raises ``InvalidTransitionError`` without touching the job if it is not queued
        (for example, already being processed elsewhere).
        """
        with bind(job_id=str(job_id)):
            self.jobs.begin_rasterizing(job_id)
            with self.metrics.tracking_active_job():
                return await self._process_started(job_id, document)

    async def _process_started(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        try:
            async with asyncio.timeout(self.job_timeout_seconds):
                return await self._pipeline(job_id, document)
        except PipelineError as exc:
            logger.info("job failed: %s (%s)", exc.slug, exc, extra={"error_slug": exc.slug})
            return self._fail(job_id, exc)
        except TimeoutError:
            logger.warning("job exceeded its %.0fs deadline", self.job_timeout_seconds)
            return self._fail(
                job_id, JobTimeoutError(f"deadline of {self.job_timeout_seconds:g}s exceeded")
            )
        except asyncio.CancelledError:
            self._fail(job_id, JobCancelledError("cancelled during processing"))
            raise
        except Exception:
            logger.exception("job failed unexpectedly")
            return self._fail(job_id, PipelineError())

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        """Bind the stage for logs, time it, and log its latency when it completes."""
        start = time.perf_counter()
        with bind(stage=name), self.metrics.stage(name):
            yield
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info("stage %s completed", name, extra={"latency_ms": latency_ms})

    async def _pipeline(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        snapshot = self.jobs.get(job_id)
        options = self.jobs.options(job_id)
        mime_type = snapshot.document.mime_type

        with self._stage("rasterization"):
            page_count = await self.rasterizer.count_pages(document, mime_type)
            pages: list[RasterizedPage] = [
                page
                async for page in self.rasterizer.rasterize(
                    document, mime_type, options.rasterization, page_count=page_count
                )
            ]
            self.metrics.rasterized_pages.inc(len(pages))
        self.jobs.begin_extracting(job_id, [p.metadata for p in pages])

        with self._stage("extraction"):
            request = await asyncio.to_thread(
                self.builder.build,
                pages,
                options.extraction or self.default_extraction,
                page_count=page_count,
                document_kind=options.document_kind,
            )
            del pages  # the encoded request now holds the only copy of the page images
            response = await self.extractor.extract(request)
            logger.info(
                "model %s answered after %d attempt(s)",
                response.model,
                response.attempts,
                extra={"upstream_request_id": response.request_id, "pages": len(request.images)},
            )
            extraction = parse_extraction(response.text, request)
        self.jobs.begin_reconciling(job_id, extraction)

        with self._stage("reconciliation"):
            report = reconcile(extraction, options.reconciliation)
            for discrepancy in report.discrepancies:
                self.metrics.discrepancies.labels(rule_id=discrepancy.rule.value).inc()
        final = self.jobs.succeed(job_id, report)
        self._record_outcome(final)
        logger.info(
            "job succeeded; reconciliation %s",
            report.status,
            extra={"discrepancy_count": len(report.discrepancies)},
        )
        return final

    def _record_outcome(self, snapshot: JobStatusResponse) -> None:
        self.metrics.job_finished(
            status=snapshot.status.value, mime_type=snapshot.document.mime_type.value
        )

    def _fail(self, job_id: UUID, error: PipelineError) -> JobStatusResponse:
        problem = error.to_problem(instance=job_path(job_id))
        try:
            snapshot = self.jobs.fail(job_id, problem)
        except InvalidTransitionError:  # already terminal; keep (and count) the first outcome
            return self.jobs.get(job_id)
        self._record_outcome(snapshot)
        return snapshot
