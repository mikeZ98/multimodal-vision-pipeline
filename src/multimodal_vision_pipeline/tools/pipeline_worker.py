"""End-to-end job execution: rasterize → build payload → call vision → parse → reconcile.

Each stage boundary is a ``JobManager`` transition. Any failure, a job deadline or a
cancellation ends the job in ``failed`` with an RFC 9457 problem, so a job is never
left stranded in a running state. Concurrency is bounded by a semaphore; jobs
waiting for a slot stay ``queued``.
"""

import asyncio
import logging
from typing import Final, Protocol
from uuid import UUID

from multimodal_vision_pipeline.core.errors import (
    InvalidTransitionError,
    JobCancelledError,
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
from multimodal_vision_pipeline.schemas import (
    ExtractionModelConfig,
    JobStatusResponse,
    VisionSettings,
)
from multimodal_vision_pipeline.tools.vision_client import VisionResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY: Final = 4
DEFAULT_JOB_TIMEOUT_SECONDS: Final = 900.0


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
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if job_timeout_seconds <= 0:
            raise ValueError("job_timeout_seconds must be > 0")
        self.jobs = jobs
        self.rasterizer = rasterizer
        self.extractor = extractor
        self.builder = builder or VisionPayloadBuilder()
        self.default_extraction = default_extraction or ExtractionModelConfig()
        self.max_concurrency = max_concurrency
        self.job_timeout_seconds = job_timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[UUID, asyncio.Task[JobStatusResponse]] = {}

    @classmethod
    def from_settings(
        cls,
        settings: VisionSettings,
        *,
        jobs: JobManager,
        rasterizer: Rasterizer,
        extractor: VisionExtractor,
    ) -> "PipelineWorker":
        return cls(
            jobs=jobs,
            rasterizer=rasterizer,
            extractor=extractor,
            default_extraction=settings.extraction_config(),
            max_concurrency=settings.worker_max_concurrency,
            job_timeout_seconds=settings.worker_job_timeout_seconds,
        )

    # -- background execution -------------------------------------------------
    @property
    def active_jobs(self) -> frozenset[UUID]:
        return frozenset(self._tasks)

    def submit(self, job_id: UUID, document: bytes) -> asyncio.Task[JobStatusResponse]:
        """Schedule a queued job; it starts once a concurrency slot is free."""
        if job_id in self._tasks:
            raise InvalidTransitionError(f"job {job_id} is already scheduled")
        task = asyncio.create_task(self._run_bounded(job_id, document), name=f"job-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return task

    async def _run_bounded(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        try:
            async with self._slots:
                return await self.process(job_id, document)
        except asyncio.CancelledError:
            # Cancelled while waiting for a slot: the job never left ``queued``.
            self._fail(job_id, JobCancelledError("cancelled before processing started"))
            raise

    async def shutdown(self) -> None:
        """Cancel every scheduled job and wait until each has reached a terminal state."""
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
        self.jobs.begin_rasterizing(job_id)
        try:
            async with asyncio.timeout(self.job_timeout_seconds):
                return await self._pipeline(job_id, document)
        except PipelineError as exc:
            logger.info("job %s failed: %s (%s)", job_id, exc.slug, exc)
            return self._fail(job_id, exc)
        except TimeoutError:
            logger.warning("job %s exceeded %.0fs", job_id, self.job_timeout_seconds)
            return self._fail(
                job_id, JobTimeoutError(f"deadline of {self.job_timeout_seconds:g}s exceeded")
            )
        except asyncio.CancelledError:
            self._fail(job_id, JobCancelledError("cancelled during processing"))
            raise
        except Exception:
            logger.exception("job %s failed unexpectedly", job_id)
            return self._fail(job_id, PipelineError())

    async def _pipeline(self, job_id: UUID, document: bytes) -> JobStatusResponse:
        snapshot = self.jobs.get(job_id)
        options = self.jobs.options(job_id)
        mime_type = snapshot.document.mime_type

        page_count = await self.rasterizer.count_pages(document, mime_type)
        pages: list[RasterizedPage] = [
            page
            async for page in self.rasterizer.rasterize(
                document, mime_type, options.rasterization, page_count=page_count
            )
        ]
        self.jobs.begin_extracting(job_id, [p.metadata for p in pages])

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
            "job %s: model %s answered after %d attempt(s), request %s",
            job_id,
            response.model,
            response.attempts,
            response.request_id,
        )
        extraction = parse_extraction(response.text, request)
        self.jobs.begin_reconciling(job_id, extraction)

        report = reconcile(extraction, options.reconciliation)
        return self.jobs.succeed(job_id, report)

    def _fail(self, job_id: UUID, error: PipelineError) -> JobStatusResponse:
        problem = error.to_problem(instance=f"/v1/jobs/{job_id}")
        try:
            return self.jobs.fail(job_id, problem)
        except InvalidTransitionError:  # already terminal; keep the first outcome
            return self.jobs.get(job_id)
