"""CI gates: static inspection of the GitHub Actions workflows.

Pure YAML/text checks: no GitHub API, no runner, no subprocess. The workflow is
cross-checked against the local gates in CLAUDE.md, the uv/Python pins, the coverage
threshold in pyproject.toml, and the compose file the smoke test drives.
"""

import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_container import parse_dockerfile

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
UV_RUN = ["uv", "run", "--no-sync"]
PINNED_USES = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$")


def _load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    # YAML 1.1 reads the bare key `on` as the boolean True.
    if True in document:
        document["on"] = document.pop(True)
    return document


def _workflow_files() -> list[Path]:
    files = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    assert files, "no workflows found"
    return files


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return steps


def _step(job: dict[str, Any], *, uses: str) -> dict[str, Any]:
    (found,) = [step for step in _steps(job) if str(step.get("uses", "")).startswith(f"{uses}@")]
    return found


def _commands(job: dict[str, Any]) -> list[list[str]]:
    """Each shell command of each ``run`` step, tokenized; ``uv run --no-sync`` stripped."""
    commands: list[list[str]] = []
    for step in _steps(job):
        script = step.get("run")
        if script is None:
            continue
        joined = re.sub(r"\\\n\s*", " ", script)
        for line in joined.splitlines():
            for part in re.split(r"&&|\|\|", line):
                tokens = shlex.split(part, comments=True)
                if tokens[: len(UV_RUN)] == UV_RUN:
                    tokens = tokens[len(UV_RUN) :]
                if tokens:
                    commands.append(tokens)
    return commands


def _index(job: dict[str, Any], prefix: list[str]) -> int:
    """Position of the first step whose ``run`` invokes ``prefix``."""
    for position, step in enumerate(_steps(job)):
        if "run" in step and any(
            command[: len(prefix)] == prefix for command in _commands({"steps": [step]})
        ):
            return position
    pytest.fail(f"no step runs {' '.join(prefix)}")


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    remaining = iter(haystack)
    return all(token in remaining for token in needle)


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return _load(CI)


@pytest.fixture(scope="module")
def quality(workflow: dict[str, Any]) -> dict[str, Any]:
    job: dict[str, Any] = workflow["jobs"]["quality"]
    return job


@pytest.fixture(scope="module")
def container(workflow: dict[str, Any]) -> dict[str, Any]:
    job: dict[str, Any] = workflow["jobs"]["container"]
    return job


# ---------------------------------------------------------------------------
# Triggers, concurrency, permissions, runners
# ---------------------------------------------------------------------------


class TestWorkflowShape:
    def test_triggers_on_main_pushes_and_pull_requests(self, workflow: dict[str, Any]) -> None:
        assert workflow["on"] == {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]},
        }

    def test_concurrency_cancels_superseded_pr_runs_only(self, workflow: dict[str, Any]) -> None:
        concurrency = workflow["concurrency"]
        group = concurrency["group"]
        assert "github.workflow" in group
        assert "github.event.pull_request.number" in group
        assert "github.ref" in group
        assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"

    def test_token_is_read_only(self, workflow: dict[str, Any]) -> None:
        assert workflow["permissions"] == {"contents": "read"}
        for name, job in workflow["jobs"].items():
            assert "permissions" not in job, f"{name} must not widen the token"

    def test_jobs_run_on_hosted_ubuntu_with_timeouts(self, workflow: dict[str, Any]) -> None:
        assert set(workflow["jobs"]) == {"quality", "container"}
        for name, job in workflow["jobs"].items():
            assert job["runs-on"] == "ubuntu-latest", name
            assert 0 < job["timeout-minutes"] <= 60, name

    def test_bash_with_errexit_is_the_default_shell(self, workflow: dict[str, Any]) -> None:
        # GitHub's `bash` shell runs `bash --noprofile --norc -eo pipefail {0}`.
        assert workflow["defaults"] == {"run": {"shell": "bash"}}


# ---------------------------------------------------------------------------
# Supply chain and injection hardening (every workflow file)
# ---------------------------------------------------------------------------


class TestWorkflowHardening:
    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_every_action_is_pinned_to_a_commit_sha(self, path: Path) -> None:
        pinned = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            match = USES_LINE.match(line)
            if match is None:
                continue
            reference, trailer = match.groups()
            if reference.startswith("./"):
                continue
            assert PINNED_USES.match(reference), f"not pinned to a full SHA: {reference}"
            assert re.fullmatch(r"\s*# v\d+\.\d+\.\d+", trailer), f"{reference}: name the release"
            pinned += 1
        parsed = [
            step["uses"]
            for job in _load(path)["jobs"].values()
            for step in _steps(job)
            if "uses" in step
        ]
        assert pinned == len(parsed), "a `uses:` escaped the line-level check"

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_no_dangerous_triggers_or_secrets(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert set(_load(path)["on"]) <= {"push", "pull_request", "workflow_dispatch"}
        assert "secrets." not in text, "CI needs no secrets; the tests never call the API"

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_expressions_never_reach_shell_scripts(self, path: Path) -> None:
        # ${{ }} inside `run` is template-expanded into the script (script injection);
        # values go through `env:` and are quoted as shell variables instead.
        for job in _load(path)["jobs"].values():
            for step in _steps(job):
                assert "${{" not in step.get("run", ""), step.get("name")

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_checkout_does_not_persist_the_token(self, path: Path) -> None:
        for job in _load(path)["jobs"].values():
            checkout = _step(job, uses="actions/checkout")
            assert checkout["with"]["persist-credentials"] is False


# ---------------------------------------------------------------------------
# Quality job
# ---------------------------------------------------------------------------


class TestQualityJob:
    def test_checkout_has_full_history_first(self, quality: dict[str, Any]) -> None:
        first = _steps(quality)[0]
        assert str(first["uses"]).startswith("actions/checkout@")
        assert first["with"]["fetch-depth"] == 0, "the audit walks the whole history"

    def test_attribution_audit_covers_the_pr_head_not_the_merge_commit(
        self, quality: dict[str, Any]
    ) -> None:
        (audit,) = [step for step in _steps(quality) if "audit-attribution" in step.get("run", "")]
        assert shlex.split(audit["run"]) == [
            "scripts/audit-attribution.sh",
            "--history-only",
            "$AUDIT_REV",
        ]
        assert audit["env"]["AUDIT_REV"] == (
            "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        # Fail fast: before any dependency is downloaded.
        assert _index(quality, ["scripts/audit-attribution.sh"]) < _index(quality, ["uv", "sync"])

    def test_history_only_mode_still_audits_every_commit(self) -> None:
        script = (ROOT / "scripts" / "audit-attribution.sh").read_text(encoding="utf-8")
        assert '"${1:-}" == "--history-only"' in script
        loop = script[script.index("count=0") :]
        assert "policy_check_identity" in loop
        assert "policy_check_message" in loop
        assert "((check_identity))" not in loop, "the flag must only skip the local identity"
        hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
        assert "--history-only" not in hook, "locally the next commit's identity is checked too"

    def test_uv_with_cache_and_pinned_toolchain(self, quality: dict[str, Any]) -> None:
        setup = _step(quality, uses="astral-sh/setup-uv")["with"]
        assert setup["enable-cache"] is True
        assert "uv.lock" in setup["cache-dependency-glob"]
        python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        assert setup["python-version"] == python_version
        dockerfile = parse_dockerfile((ROOT / "Dockerfile").read_text(encoding="utf-8"))
        uv_image = re.search(r":(\d+\.\d+\.\d+)@", dockerfile.global_args["UV_IMAGE"])
        assert uv_image
        assert setup["version"] == uv_image.group(1), "CI and image must resolve uv.lock alike"

    def test_dependencies_install_strictly_from_the_lock(self, quality: dict[str, Any]) -> None:
        installs = [command for command in _commands(quality) if command[:2] == ["uv", "sync"]]
        assert installs == [["uv", "sync", "--locked"]]
        for step in _steps(quality):
            script = step.get("run", "")
            if "uv run" in script:
                assert "uv run --no-sync" in script, "tools must run in the locked env as synced"

    @pytest.mark.parametrize(
        "gate",
        [
            ["ruff", "check"],
            ["ruff", "format", "--check", "."],
            ["mypy", "--strict"],
            ["bandit", "-c", "pyproject.toml", "-r", "src"],
            ["pytest"],
            ["scripts/audit-attribution.sh"],
        ],
        ids=" ".join,
    )
    def test_quality_gate_present(self, quality: dict[str, Any], gate: list[str]) -> None:
        assert any(command[: len(gate)] == gate for command in _commands(quality))

    def test_ruff_lints_the_whole_tree(self, quality: dict[str, Any]) -> None:
        (lint,) = [command for command in _commands(quality) if command[:2] == ["ruff", "check"]]
        assert lint[-1] == "."

    def test_gates_run_in_cheap_to_expensive_order(self, quality: dict[str, Any]) -> None:
        order = [
            ["scripts/audit-attribution.sh"],
            ["uv", "sync"],
            ["ruff", "check"],
            ["ruff", "format"],
            ["mypy"],
            ["bandit"],
            ["pytest"],
        ]
        positions = [_index(quality, prefix) for prefix in order]
        assert positions == sorted(positions)

    def test_pytest_enforces_the_project_coverage_floor(self, quality: dict[str, Any]) -> None:
        (pytest_run,) = [command for command in _commands(quality) if command[0] == "pytest"]
        (floor,) = [arg for arg in pytest_run if arg.startswith("--cov-fail-under=")]
        with (ROOT / "pyproject.toml").open("rb") as handle:
            configured = tomllib.load(handle)["tool"]["coverage"]["report"]["fail_under"]
        assert int(floor.split("=")[1]) == configured >= 90
        assert not [arg for arg in pytest_run if "socket" in arg], "keep --disable-socket"

    def test_coverage_report_survives_test_failures(self, quality: dict[str, Any]) -> None:
        upload = _step(quality, uses="actions/upload-artifact")
        assert upload["if"] == "${{ !cancelled() }}"
        assert upload["with"]["path"] == "coverage.xml"

    def test_ci_runs_every_local_gate_from_claude_md(self, quality: dict[str, Any]) -> None:
        guide = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        block = re.search(r"## Commands\s+```bash\n(.*?)```", guide, re.S)
        assert block
        local = [
            shlex.split(part, comments=True)
            for line in block.group(1).splitlines()
            for part in line.split("&&")
        ]
        local = [tokens for tokens in local if tokens]
        assert len(local) >= 6, "Commands block not parsed"
        ci = _commands(quality)
        missing = [
            " ".join(tokens)
            for tokens in local
            if not any(_is_subsequence(tokens, command) for command in ci)
        ]
        assert not missing, f"local gates absent from CI: {missing}"


# ---------------------------------------------------------------------------
# Container job
# ---------------------------------------------------------------------------


def _compose() -> dict[str, Any]:
    document = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    service: dict[str, Any] = document["services"]["api"]
    return service


def _default(template: str, env: dict[str, str]) -> str:
    """Compose interpolation of ``${VAR:-default}`` / ``${VAR:?err}`` against ``env``."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(3) or ""
        return env.get(name, default)

    return re.sub(r"\$\{(\w+)(?::([-?])([^}]*))?\}", replace, template)


class TestContainerJob:
    def test_dockerfile_build_checks_are_fatal(self, container: dict[str, Any]) -> None:
        dockerfile = parse_dockerfile((ROOT / "Dockerfile").read_text(encoding="utf-8"))
        assert dockerfile.directives.get("check") == "error=true"
        assert any(
            command == ["docker", "buildx", "build", "--check", "."]
            for command in _commands(container)
        )

    def test_buildx_precedes_the_build(self, container: dict[str, Any]) -> None:
        steps = _steps(container)
        uses = [str(step.get("uses", "")).split("@")[0] for step in steps]
        assert uses.index("docker/setup-buildx-action") < uses.index("docker/build-push-action")
        assert uses.index("docker/setup-buildx-action") < _index(container, ["docker", "buildx"])

    def test_image_build_is_cached_and_loaded_locally(self, container: dict[str, Any]) -> None:
        build = _step(container, uses="docker/build-push-action")["with"]
        assert build["context"] == "."
        assert build["target"] == "runtime"
        assert build["load"] is True
        assert build["push"] is False, "CI verifies images; it never publishes them"
        assert build["cache-from"].startswith("type=gha")
        assert build["cache-to"].startswith("type=gha")
        assert "mode=max" in build["cache-to"], "cache the builder stage too"

    def test_smoke_test_runs_the_image_that_was_built(self, container: dict[str, Any]) -> None:
        built = _step(container, uses="docker/build-push-action")["with"]["tags"]
        image = _default(_compose()["image"], container["env"])
        assert built == image

    def test_compose_is_validated_before_use(self, container: dict[str, Any]) -> None:
        validate = _index(container, ["docker", "compose", "config"])
        assert validate < _index(container, ["docker", "compose", "up"])
        assert "--quiet" in next(
            c for c in _commands(container) if c[:3] == ["docker", "compose", "config"]
        )

    def test_required_settings_have_a_placeholder_not_a_secret(
        self, container: dict[str, Any]
    ) -> None:
        required = re.findall(r"\$\{(\w+):\?", (ROOT / "docker-compose.yml").read_text("utf-8"))
        assert required
        for name in required:
            value = container["env"][name]
            assert "placeholder" in value
            assert not value.startswith("sk-"), "must not look like a real credential"

    def test_service_starts_healthy_with_production_hardening(
        self, container: dict[str, Any]
    ) -> None:
        (up,) = [c for c in _commands(container) if c[:3] == ["docker", "compose", "up"]]
        assert {"--wait", "--no-build", "--detach"} <= set(up)
        assert up[up.index("--pull") + 1] == "never", "only the image built above may run"
        assert int(up[up.index("--wait-timeout") + 1]) > 0
        assert _index(container, ["docker", "compose", "up"]) > _index(
            container, ["docker", "compose", "config"]
        )

    def test_smoke_test_probes_healthz_on_the_published_port(
        self, container: dict[str, Any]
    ) -> None:
        (mapping,) = _compose()["ports"]
        address, port, _ = _default(mapping, container["env"]).split(":")
        probes = [c for c in _commands(container) if c[0] == "curl"]
        urls = {token for command in probes for token in command if token.startswith("http")}
        assert f"http://{address}:{port}/healthz" in urls
        healthz = next(c for c in probes if c[-1].endswith("/healthz"))
        assert "--fail" in healthz, "non-2xx must fail the step"
        assert "--max-time" in healthz

    def test_smoke_test_asserts_the_non_root_uid(self, container: dict[str, Any]) -> None:
        user = _compose()["user"].split(":")[0]
        script = next(
            step["run"] for step in _steps(container) if "/healthz" in step.get("run", "")
        )
        assert "id -u" in script
        assert f'"{user}"' in script

    def test_logs_on_failure_and_always_tear_down(self, container: dict[str, Any]) -> None:
        steps = _steps(container)
        (logs,) = [s for s in steps if "docker compose logs" in s.get("run", "")]
        (down,) = [s for s in steps if "docker compose down" in s.get("run", "")]
        assert logs["if"] == "${{ failure() }}"
        assert down["if"] == "${{ always() }}"
        assert steps.index(down) == len(steps) - 1
