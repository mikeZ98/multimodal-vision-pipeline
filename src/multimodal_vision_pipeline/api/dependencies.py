"""Request-scoped access to the services the lifespan wires up."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from multimodal_vision_pipeline.api.errors import ServiceNotReadyError
from multimodal_vision_pipeline.core.job_manager import JobManager
from multimodal_vision_pipeline.core.rasterizer import Rasterizer
from multimodal_vision_pipeline.observability.metrics import PipelineMetrics
from multimodal_vision_pipeline.schemas import VisionSettings
from multimodal_vision_pipeline.tools.pipeline_worker import PipelineWorker, VisionExtractor


@dataclass(frozen=True, slots=True)
class Services:
    settings: VisionSettings
    jobs: JobManager
    rasterizer: Rasterizer
    extractor: VisionExtractor
    worker: PipelineWorker
    metrics: PipelineMetrics


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, Services):
        raise ServiceNotReadyError("the application has not finished starting")
    return services


def get_settings(services: Annotated[Services, Depends(get_services)]) -> VisionSettings:
    return services.settings


def get_job_manager(services: Annotated[Services, Depends(get_services)]) -> JobManager:
    return services.jobs


def get_worker(services: Annotated[Services, Depends(get_services)]) -> PipelineWorker:
    return services.worker


SettingsDep = Annotated[VisionSettings, Depends(get_settings)]
JobsDep = Annotated[JobManager, Depends(get_job_manager)]
WorkerDep = Annotated[PipelineWorker, Depends(get_worker)]
