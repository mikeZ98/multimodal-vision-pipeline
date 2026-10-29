"""In-memory, thread-safe job store and lifecycle state machine.

``queued → rasterizing → extracting → reconciling → succeeded``; any non-terminal
state may move to ``failed``. Every transition rebuilds the immutable
``JobStatusResponse`` snapshot, so the schema's lifecycle invariants are re-checked
before a change becomes visible. A rejected transition leaves the job untouched.
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, NamedTuple
from uuid import UUID, uuid4

from multimodal_vision_pipeline.core.errors import (
    IdempotencyConflictError,
    InvalidTransitionError,
    JobCapacityError,
    JobNotFoundError,
)
from multimodal_vision_pipeline.schemas import (
    DocumentDescriptor,
    ExtractionResult,
    IngestionOptions,
    JobStatus,
    JobStatusResponse,
    PageMetadata,
    ProblemDetails,
    ReconciliationReport,
    utc_now,
)

DEFAULT_MAX_JOBS: Final = 10_000

TRANSITIONS: Final[Mapping[JobStatus, frozenset[JobStatus]]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RASTERIZING, JobStatus.FAILED}),
    JobStatus.RASTERIZING: frozenset({JobStatus.EXTRACTING, JobStatus.FAILED}),
    JobStatus.EXTRACTING: frozenset({JobStatus.RECONCILING, JobStatus.FAILED}),
    JobStatus.RECONCILING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
}


class Submission(NamedTuple):
    job: JobStatusResponse
    created: bool
    """``False`` when an idempotency key replayed an existing job."""


@dataclass(slots=True)
class _Entry:
    snapshot: JobStatusResponse
    options: IngestionOptions


class JobManager:
    """Owns job state. All public methods are atomic under one lock."""

    def __init__(
        self,
        *,
        max_jobs: int = DEFAULT_MAX_JOBS,
        clock: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be >= 1")
        self._max_jobs = max_jobs
        self._clock = clock
        self._id_factory = id_factory
        self._lock = threading.Lock()
        self._jobs: dict[UUID, _Entry] = {}  # insertion order == creation order
        self._by_key: dict[str, UUID] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    # -- creation ---------------------------------------------------------
    def submit(self, document: DocumentDescriptor, options: IngestionOptions) -> Submission:
        """Create a queued job, or replay the job already bound to the idempotency key.

        Reusing a key for a different document or different options is a conflict.
        """
        key = options.idempotency_key
        with self._lock:
            if key is not None and (existing_id := self._by_key.get(key)) is not None:
                entry = self._jobs[existing_id]
                if entry.snapshot.document.sha256 != document.sha256 or entry.options != options:
                    raise IdempotencyConflictError(
                        f"idempotency key {key!r} is bound to job {existing_id}"
                    )
                return Submission(entry.snapshot, created=False)
            self._make_room()
            now = self._clock()
            job_id = self._id_factory()
            if job_id in self._jobs:
                raise RuntimeError(f"id factory produced duplicate job id {job_id}")
            snapshot = JobStatusResponse(
                job_id=job_id,
                status=JobStatus.QUEUED,
                document=document,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = _Entry(snapshot=snapshot, options=options)
            if key is not None:
                self._by_key[key] = job_id
            return Submission(snapshot, created=True)

    def _make_room(self) -> None:
        if len(self._jobs) < self._max_jobs:
            return
        victim = next(
            (job_id for job_id, e in self._jobs.items() if e.snapshot.status.is_terminal), None
        )
        if victim is None:
            raise JobCapacityError(f"{self._max_jobs} jobs in flight; none can be evicted")
        evicted = self._jobs.pop(victim)
        if (key := evicted.options.idempotency_key) is not None:
            self._by_key.pop(key, None)

    # -- reads ------------------------------------------------------------
    def _entry(self, job_id: UUID) -> _Entry:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(f"job {job_id} does not exist") from None

    def get(self, job_id: UUID) -> JobStatusResponse:
        with self._lock:
            return self._entry(job_id).snapshot

    def options(self, job_id: UUID) -> IngestionOptions:
        with self._lock:
            return self._entry(job_id).options

    # -- transitions ------------------------------------------------------
    def _advance(self, job_id: UUID, target: JobStatus, **changes: Any) -> JobStatusResponse:
        with self._lock:
            entry = self._entry(job_id)
            current = entry.snapshot
            if target not in TRANSITIONS[current.status]:
                raise InvalidTransitionError(
                    f"job {job_id}: {current.status} -> {target} is not allowed"
                )
            fields = dict(current)
            fields.update(changes, status=target, updated_at=max(self._clock(), current.updated_at))
            entry.snapshot = JobStatusResponse(**fields)  # validates before it is stored
            return entry.snapshot

    def begin_rasterizing(self, job_id: UUID) -> JobStatusResponse:
        return self._advance(job_id, JobStatus.RASTERIZING)

    def begin_extracting(self, job_id: UUID, pages: Sequence[PageMetadata]) -> JobStatusResponse:
        return self._advance(job_id, JobStatus.EXTRACTING, pages=tuple(pages))

    def begin_reconciling(self, job_id: UUID, extraction: ExtractionResult) -> JobStatusResponse:
        return self._advance(job_id, JobStatus.RECONCILING, extraction=extraction)

    def succeed(self, job_id: UUID, reconciliation: ReconciliationReport) -> JobStatusResponse:
        return self._advance(job_id, JobStatus.SUCCEEDED, reconciliation=reconciliation)

    def fail(self, job_id: UUID, error: ProblemDetails) -> JobStatusResponse:
        return self._advance(job_id, JobStatus.FAILED, error=error)
