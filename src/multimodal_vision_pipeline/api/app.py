"""FastAPI application factory and worker lifecycle.

Run with ``uvicorn --factory multimodal_vision_pipeline.api:create_app``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from multimodal_vision_pipeline import __version__
from multimodal_vision_pipeline.api import health, routes
from multimodal_vision_pipeline.api.dependencies import Services
from multimodal_vision_pipeline.api.errors import install_problem_handlers
from multimodal_vision_pipeline.api.middleware import BodySizeLimitMiddleware
from multimodal_vision_pipeline.api.uploads import MULTIPART_OVERHEAD_BYTES
from multimodal_vision_pipeline.core.job_manager import JobManager
from multimodal_vision_pipeline.core.rasterizer import DocumentRasterizer, Rasterizer
from multimodal_vision_pipeline.observability.logging import configure_logging
from multimodal_vision_pipeline.observability.metrics import PipelineMetrics
from multimodal_vision_pipeline.observability.middleware import CorrelationMiddleware
from multimodal_vision_pipeline.schemas import VisionSettings
from multimodal_vision_pipeline.tools.pipeline_worker import PipelineWorker, VisionExtractor
from multimodal_vision_pipeline.tools.vision_client import VisionClient

logger = logging.getLogger(__name__)


def create_app(
    settings: VisionSettings | None = None,
    *,
    jobs: JobManager | None = None,
    rasterizer: Rasterizer | None = None,
    extractor: VisionExtractor | None = None,
    metrics: PipelineMetrics | None = None,
) -> FastAPI:
    """Build the service. Collaborators default to production implementations.

    A ``VisionClient`` is created (and closed) by the lifespan only when no
    ``extractor`` is injected; it requires ``MVP_ANTHROPIC_API_KEY``. Each app gets its
    own metrics registry unless one is passed in.
    """
    config = settings if settings is not None else VisionSettings()
    app_metrics = metrics if metrics is not None else PipelineMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if config.log_json:
            configure_logging(config.log_level)
        owned_client: VisionClient | None = None
        worker_extractor: VisionExtractor
        if extractor is None:
            owned_client = worker_extractor = VisionClient.from_settings(
                config, metrics=app_metrics
            )
        else:
            worker_extractor = extractor
        # Explicit None checks: an empty JobManager is falsy (it defines __len__).
        job_store = jobs if jobs is not None else JobManager(max_jobs=config.job_store_max_jobs)
        page_source = (
            rasterizer
            if rasterizer is not None
            else DocumentRasterizer(max_document_bytes=config.max_upload_bytes)
        )
        worker = PipelineWorker.from_settings(
            config,
            jobs=job_store,
            rasterizer=page_source,
            extractor=worker_extractor,
            metrics=app_metrics,
        )
        app.state.services = Services(
            settings=config,
            jobs=job_store,
            rasterizer=page_source,
            extractor=worker_extractor,
            worker=worker,
            metrics=app_metrics,
        )
        logger.info("pipeline worker started (concurrency %d)", worker.max_concurrency)
        try:
            yield
        finally:
            logger.info("draining jobs for up to %.0fs", config.worker_shutdown_grace_seconds)
            await worker.drain(config.worker_shutdown_grace_seconds)
            if owned_client is not None:
                await owned_client.aclose()

    app = FastAPI(
        title="multimodal-vision-pipeline",
        version=__version__,
        summary="Asynchronous document vision extraction with arithmetic reconciliation.",
        lifespan=lifespan,
    )
    app.add_middleware(
        BodySizeLimitMiddleware, max_body_bytes=config.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    )
    # Added last, so it is outermost: ids and access logs cover every response,
    # including problems produced by the body size limit.
    app.add_middleware(CorrelationMiddleware, metrics=app_metrics)
    app.state.metrics = app_metrics
    install_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(routes.router)
    return app
