"""Prometheus metrics for the pipeline, held in a per-application registry.

A private ``CollectorRegistry`` per ``PipelineMetrics`` keeps applications (and tests)
isolated from each other and from the process-global default registry.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

STAGES: Final = ("rasterization", "extraction", "reconciliation")
STAGE_BUCKETS: Final = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
UPSTREAM_BUCKETS: Final = (0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)
HTTP_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
UNMATCHED_ROUTE: Final = "unmatched"
"""Route label for requests no route matched, keeping label cardinality bounded."""


class PipelineMetrics:
    """All service metrics. Every method is thread-safe (prometheus_client locks)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.jobs = Counter(
            "mvp_jobs", "Jobs that reached a terminal state.", ["status", "mime_type"], registry=r
        )
        self.job_duration = Histogram(
            "mvp_job_duration_seconds",
            "Wall time spent in each pipeline stage.",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=r,
        )
        self.rasterized_pages = Counter(
            "mvp_rasterized_pages", "Pages rendered by the rasterizer.", registry=r
        )
        self.upstream_requests = Counter(
            "mvp_upstream_requests",
            "Vision API attempts by outcome (HTTP status, 'timeout' or 'connection_error').",
            ["model", "status_code"],
            registry=r,
        )
        self.upstream_latency = Histogram(
            "mvp_upstream_latency_seconds",
            "Latency of each vision API attempt.",
            ["model"],
            buckets=UPSTREAM_BUCKETS,
            registry=r,
        )
        self.discrepancies = Counter(
            "mvp_reconciliation_discrepancies",
            "Arithmetic discrepancies found, by reconciliation rule.",
            ["rule_id"],
            registry=r,
        )
        self.active_jobs = Gauge(
            "mvp_active_jobs", "Jobs currently being processed by the worker.", registry=r
        )
        self.http_requests = Counter(
            "mvp_http_requests",
            "HTTP requests by route template and status.",
            ["method", "route", "status_code"],
            registry=r,
        )
        self.http_latency = Histogram(
            "mvp_http_request_duration_seconds",
            "HTTP request latency by route template.",
            ["method", "route"],
            buckets=HTTP_BUCKETS,
            registry=r,
        )

    # -- recording helpers -------------------------------------------------------
    def job_finished(self, *, status: str, mime_type: str) -> None:
        self.jobs.labels(status=status, mime_type=mime_type).inc()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Observe the wall time of a pipeline stage, whether it succeeds or fails."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.job_duration.labels(stage=name).observe(time.perf_counter() - start)

    @contextmanager
    def tracking_active_job(self) -> Iterator[None]:
        self.active_jobs.inc()
        try:
            yield
        finally:
            self.active_jobs.dec()

    def upstream_attempt(self, *, model: str, status_code: str, seconds: float) -> None:
        self.upstream_requests.labels(model=model, status_code=status_code).inc()
        self.upstream_latency.labels(model=model).observe(seconds)

    def http_request(self, *, method: str, route: str, status_code: int, seconds: float) -> None:
        self.http_requests.labels(method=method, route=route, status_code=str(status_code)).inc()
        self.http_latency.labels(method=method, route=route).observe(seconds)

    def render(self) -> tuple[bytes, str]:
        """Prometheus text exposition and its content type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
