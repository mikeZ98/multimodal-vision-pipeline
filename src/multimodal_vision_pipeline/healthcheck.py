"""Container health probe: ``python -m multimodal_vision_pipeline.healthcheck [PATH ...]``.

The runtime image ships neither curl nor wget, so the Dockerfile ``HEALTHCHECK`` and the
compose healthcheck run this. Exit status 0 means every path answered 200 from the local
server on ``UVICORN_PORT``; 1 means at least one did not; 2 means bad arguments.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Final

import httpx

DEFAULT_PATHS: Final = ("/healthz",)
DEFAULT_PORT: Final = 8000
TIMEOUT_SECONDS: Final = 2.0

EXIT_HEALTHY: Final = 0
EXIT_UNHEALTHY: Final = 1
EXIT_USAGE: Final = 2


class UsageError(ValueError):
    """Arguments or environment the probe cannot work with."""


def base_url(environ: Mapping[str, str]) -> str:
    raw = environ.get("UVICORN_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw)
    except ValueError:
        raise UsageError(f"UVICORN_PORT is not a port number: {raw!r}") from None
    if not 0 < port < 2**16:
        raise UsageError(f"UVICORN_PORT out of range: {port}")
    return f"http://127.0.0.1:{port}"


def check_paths(paths: Sequence[str]) -> None:
    if bad := [path for path in paths if not path.startswith("/")]:
        raise UsageError(f"paths must start with '/': {', '.join(bad)}")


def local_client(url: str, *, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    # trust_env=False: a proxy configured for outbound API calls must not intercept loopback.
    return httpx.Client(base_url=url, timeout=TIMEOUT_SECONDS, transport=transport, trust_env=False)


def probe(paths: Sequence[str], *, client: httpx.Client) -> list[str]:
    """One failure line per path that did not answer 200; empty when healthy."""
    failures: list[str] = []
    for path in paths:
        try:
            response = client.get(path)
        except httpx.HTTPError as error:
            failures.append(f"{path}: {type(error).__name__}")
            continue
        if response.status_code != httpx.codes.OK:
            failures.append(f"{path}: HTTP {response.status_code}")
    return failures


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    paths = tuple(sys.argv[1:] if argv is None else argv) or DEFAULT_PATHS
    try:
        check_paths(paths)
        url = base_url(os.environ if environ is None else environ)
    except UsageError as error:
        sys.stderr.write(f"healthcheck: {error}\n")
        return EXIT_USAGE
    with local_client(url, transport=transport) as client:
        failures = probe(paths, client=client)
    for failure in failures:
        sys.stderr.write(f"unhealthy {failure}\n")
    return EXIT_UNHEALTHY if failures else EXIT_HEALTHY


if __name__ == "__main__":
    sys.exit(main())
