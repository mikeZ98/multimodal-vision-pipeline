"""Liveness and readiness probes."""

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from multimodal_vision_pipeline.api.dependencies import JobsDep, WorkerDep
from multimodal_vision_pipeline.schemas import PipelineModel

router = APIRouter(tags=["health"])


class Liveness(PipelineModel):
    status: Literal["ok"] = "ok"


class WorkerStatus(PipelineModel):
    accepting: bool
    active_jobs: int
    max_concurrency: int
    max_queued_jobs: int


class Readiness(PipelineModel):
    status: Literal["ready", "not_ready"]
    worker: WorkerStatus
    stored_jobs: int
    max_jobs: int


@router.get("/healthz", response_model=Liveness, summary="Process is alive")
async def healthz() -> Liveness:
    return Liveness()


@router.get(
    "/readyz",
    response_model=Readiness,
    summary="Ready to accept documents",
    responses={503: {"model": Readiness, "description": "Shutting down or at capacity"}},
)
async def readyz(jobs: JobsDep, worker: WorkerDep) -> JSONResponse:
    active = len(worker.active_jobs)
    ready = worker.accepting and active < worker.max_queued_jobs
    report = Readiness(
        status="ready" if ready else "not_ready",
        worker=WorkerStatus(
            accepting=worker.accepting,
            active_jobs=active,
            max_concurrency=worker.max_concurrency,
            max_queued_jobs=worker.max_queued_jobs,
        ),
        stored_jobs=len(jobs),
        max_jobs=jobs.max_jobs,
    )
    return JSONResponse(report.model_dump(mode="json"), status_code=200 if ready else 503)
