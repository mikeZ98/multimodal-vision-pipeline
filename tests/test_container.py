"""Container gates: static inspection of Dockerfile, docker-compose.yml and .dockerignore.

Nothing here talks to a Docker daemon, builds an image or spawns a process. The files are
parsed as text/YAML and cross-checked against the code they configure: settings fields,
uvicorn CLI options, the health probe and the logging setup.
"""

import importlib
import io
import json
import logging
import logging.config
import os
import re
import shlex
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import SecretStr
from uvicorn.main import main as uvicorn_cli

from multimodal_vision_pipeline import healthcheck
from multimodal_vision_pipeline.observability import logging as mvp_logging
from multimodal_vision_pipeline.schemas import VisionSettings

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
DOCKERIGNORE = ROOT / ".dockerignore"
APP_UID_GID = "10001:10001"
REQUIRED_PACKAGES = frozenset({"poppler-utils", "libopenjp2-7", "libtiff6"})
ALLOWED_PACKAGES = REQUIRED_PACKAGES | {"tini"}
TEST_API_KEY = "sk-ant-test-0000000000"

# ---------------------------------------------------------------------------
# Dockerfile parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    keyword: str
    value: str

    def exec_form(self) -> list[str]:
        parsed = json.loads(self.value)
        assert isinstance(parsed, list), f"{self.keyword} must use the JSON exec form"
        return [str(item) for item in parsed]


@dataclass
class Stage:
    base: str
    name: str
    instructions: list[Instruction] = field(default_factory=list)

    def all(self, keyword: str) -> list[Instruction]:
        return [item for item in self.instructions if item.keyword == keyword]

    def one(self, keyword: str) -> Instruction:
        found = self.all(keyword)
        assert len(found) == 1, f"stage {self.name}: expected one {keyword}, found {len(found)}"
        return found[0]

    def env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for item in self.all("ENV"):
            for pair in shlex.split(item.value):
                key, _, value = pair.partition("=")
                values[key] = value
        return values

    def run_commands(self) -> list[str]:
        return [item.value for item in self.all("RUN")]


@dataclass
class Dockerfile:
    directives: dict[str, str]
    global_args: dict[str, str]
    stages: list[Stage]

    def stage(self, name: str) -> Stage:
        return next(stage for stage in self.stages if stage.name == name)

    def resolve(self, text: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: self.global_args[m.group(1)], text)


def _logical_lines(text: str) -> Iterator[str]:
    """Joins ``\\`` continuations and drops comment lines, as the Docker parser does."""
    buffer: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].strip())
            continue
        buffer.append(stripped)
        yield " ".join(buffer)
        buffer = []
    assert not buffer, "dangling line continuation at end of Dockerfile"


def parse_dockerfile(text: str) -> Dockerfile:
    directives = {
        match.group(1).lower(): match.group(2)
        for match in re.finditer(r"^#\s*(\w+)=(\S+)\s*$", text.split("\n\n", 1)[0], re.M)
    }
    global_args: dict[str, str] = {}
    stages: list[Stage] = []
    for line in _logical_lines(text):
        keyword, _, value = line.partition(" ")
        keyword, value = keyword.upper(), value.strip()
        if keyword == "FROM":
            match = re.fullmatch(r"(\S+) AS (\S+)", value, re.IGNORECASE)
            assert match, f"every stage must be named: FROM {value}"
            stages.append(Stage(base=match.group(1), name=match.group(2)))
        elif not stages:
            assert keyword == "ARG", f"only ARG may precede the first FROM, got {keyword}"
            name, _, default = value.partition("=")
            global_args[name] = default
        else:
            stages[-1].instructions.append(Instruction(keyword, value))
    return Dockerfile(directives, global_args, stages)


@pytest.fixture(scope="module")
def dockerfile() -> Dockerfile:
    return parse_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runtime(dockerfile: Dockerfile) -> Stage:
    return dockerfile.stage("runtime")


@pytest.fixture(scope="module")
def builder(dockerfile: Dockerfile) -> Stage:
    return dockerfile.stage("builder")


def _copies(stage: Stage) -> list[tuple[str | None, list[str], str]]:
    """``(from_stage, sources, destination)`` for each COPY."""
    result: list[tuple[str | None, list[str], str]] = []
    for item in stage.all("COPY"):
        tokens = shlex.split(item.value)
        flags = {
            key: val
            for key, _, val in (tok[2:].partition("=") for tok in tokens if tok[:2] == "--")
        }
        paths = [tok for tok in tokens if not tok.startswith("--")]
        result.append((flags.get("from"), paths[:-1], paths[-1]))
    return result


def _apt_packages(stage: Stage) -> set[str]:
    packages: set[str] = set()
    for command in stage.run_commands():
        for step in command.split("&&"):
            words = shlex.split(step)
            if words[:2] == ["apt-get", "install"]:
                packages.update(word for word in words[2:] if not word.startswith("-"))
    return packages


class TestDockerfileParser:
    def test_continuations_and_comments(self) -> None:
        parsed = parse_dockerfile(
            "# syntax=docker/dockerfile:1.7\n\nARG A=x\nFROM img:${A} AS one\n"
            "RUN echo a \\\n    # interleaved comment\n    && echo b\n"
        )
        assert parsed.directives == {"syntax": "docker/dockerfile:1.7"}
        assert parsed.resolve(parsed.stages[0].base) == "img:x"
        assert parsed.stages[0].run_commands() == ["echo a && echo b"]

    def test_rejects_unnamed_stage_and_dangling_continuation(self) -> None:
        with pytest.raises(AssertionError, match="named"):
            parse_dockerfile("FROM scratch\n")
        with pytest.raises(AssertionError, match="dangling"):
            parse_dockerfile("FROM scratch AS a\nRUN true \\\n")


# ---------------------------------------------------------------------------
# Dockerfile architecture
# ---------------------------------------------------------------------------


class TestDockerfileArchitecture:
    def test_buildkit_syntax_is_pinned(self, dockerfile: Dockerfile) -> None:
        assert re.fullmatch(r"docker/dockerfile:\d+\.\d+", dockerfile.directives["syntax"])

    def test_multi_stage_builder_then_runtime(self, dockerfile: Dockerfile) -> None:
        assert [stage.name for stage in dockerfile.stages] == ["uv", "builder", "runtime"]
        assert dockerfile.stages[-1].name == "runtime", "the final stage is what ships"

    def test_builder_and_runtime_share_the_python_base(self, dockerfile: Dockerfile) -> None:
        # The venv's interpreter is a symlink into the base image's /usr/local/bin.
        assert dockerfile.stage("builder").base == dockerfile.stage("runtime").base
        assert dockerfile.stage("runtime").base == "${PYTHON_IMAGE}"

    def test_base_images_pinned_by_version_and_digest(self, dockerfile: Dockerfile) -> None:
        digest = r"@sha256:[0-9a-f]{64}"
        python_image = dockerfile.global_args["PYTHON_IMAGE"]
        uv_image = dockerfile.global_args["UV_IMAGE"]
        assert re.fullmatch(r"python:3\.12\.\d+-slim-bookworm" + digest, python_image)
        assert re.fullmatch(r"ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+" + digest, uv_image)
        assert dockerfile.resolve(dockerfile.stage("uv").base) == uv_image

    def test_python_minor_matches_project(self, dockerfile: Dockerfile) -> None:
        pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        match = re.match(r"python:(\d+\.\d+)\.", dockerfile.global_args["PYTHON_IMAGE"])
        assert match
        assert match.group(1) == pinned


class TestBuilderStage:
    def test_uv_comes_from_the_pinned_stage(self, builder: Stage) -> None:
        assert (("uv", ["/uv", "/uvx"], "/bin/")) in _copies(builder)

    def test_bytecode_compiled_locked_venv(self, builder: Stage) -> None:
        env = builder.env()
        assert env["UV_COMPILE_BYTECODE"] == "1"
        assert env["UV_PYTHON_DOWNLOADS"] == "never", "use the base image interpreter"
        assert env["UV_PROJECT_ENVIRONMENT"] == "/opt/venv"

    def test_dependencies_sync_before_sources_for_layer_caching(self, builder: Stage) -> None:
        syncs = [cmd for cmd in builder.run_commands() if "uv sync" in cmd]
        assert len(syncs) == 2
        deps, project = syncs
        for command in syncs:
            assert "--locked" in command, "the build must fail if uv.lock is stale"
            assert "--no-dev" in command, "dev tooling never reaches the image"
        assert "--no-install-project" in deps
        assert "--no-editable" in project, "the runtime copies the venv, not the sources"
        assert "type=bind,source=uv.lock" in deps, "resolve from the lock before sources exist"
        steps = [item for item in builder.instructions if item.keyword in {"RUN", "COPY"}]
        deps_step = next(i for i, item in enumerate(steps) if "--no-install-project" in item.value)
        source_step = next(
            i for i, item in enumerate(steps) if item.keyword == "COPY" and "src" in item.value
        )
        assert deps_step < source_step, "editing code must not invalidate the dependency layer"


class TestRuntimeStage:
    def test_installs_required_system_packages_only(self, runtime: Stage) -> None:
        packages = _apt_packages(runtime)
        assert packages >= REQUIRED_PACKAGES
        assert packages <= ALLOWED_PACKAGES, f"unexpected packages: {packages - ALLOWED_PACKAGES}"

    def test_apt_layer_is_minimal_and_cleaned(self, runtime: Stage) -> None:
        (install,) = [cmd for cmd in runtime.run_commands() if "apt-get install" in cmd]
        assert "--no-install-recommends" in install
        assert "rm -rf /var/lib/apt/lists/*" in install

    def test_no_build_toolchain_in_runtime(self, runtime: Stage) -> None:
        assert all(source != "uv" for source, _, _ in _copies(runtime))
        joined = " ".join(runtime.run_commands())
        for tool in ("uv ", "pip install", "build-essential", "gcc"):
            assert tool not in joined

    def test_only_the_venv_comes_from_the_builder(self, runtime: Stage) -> None:
        from_builder = [(src, dst) for stage, src, dst in _copies(runtime) if stage == "builder"]
        assert from_builder == [(["/opt/venv"], "/opt/venv")]
        assert runtime.env()["PATH"].startswith("/opt/venv/bin:")

    def test_runs_as_dedicated_non_root_user(self, runtime: Stage) -> None:
        (setup,) = [cmd for cmd in runtime.run_commands() if "useradd" in cmd]
        assert "groupadd --system --gid 10001 appuser" in setup
        assert re.search(r"useradd --system --uid 10001 --gid appuser\b", setup)
        assert "--shell /usr/sbin/nologin" in setup
        users = [item.value for item in runtime.all("USER")]
        assert users == [APP_UID_GID], "numeric, so runAsNonRoot can verify it"
        keywords = [item.keyword for item in runtime.instructions]
        assert keywords.index("USER") < keywords.index("ENTRYPOINT")
        assert keywords.index("USER") > max(
            index for index, keyword in enumerate(keywords) if keyword in {"RUN", "COPY"}
        ), "nothing runs or is copied as root after dropping privileges"

    def test_setuid_and_setgid_bits_are_stripped(self, runtime: Stage) -> None:
        assert any("-perm /6000" in cmd and "chmod a-s" in cmd for cmd in runtime.run_commands())

    def test_read_only_rootfs_friendly_environment(self, runtime: Stage) -> None:
        env = runtime.env()
        assert env["PYTHONDONTWRITEBYTECODE"] == "1", "bytecode was compiled in the builder"
        assert env["TMPDIR"] == "/tmp"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_no_secrets_or_remote_sources_baked_in(self, dockerfile: Dockerfile) -> None:
        secret = re.compile(r"(?i)(api_?key|secret|token|password)")
        for stage in dockerfile.stages:
            assert not stage.all("ADD"), "ADD can fetch URLs and unpack archives; use COPY"
            names = list(stage.env()) + [item.value for item in stage.all("ARG")]
            assert not [name for name in names if secret.search(name)]
        assert not [name for name in dockerfile.global_args if secret.search(name)]


# ---------------------------------------------------------------------------
# Dockerfile: startup, signals, health
# ---------------------------------------------------------------------------


def _uvicorn_params(
    runtime: Stage, monkeypatch: pytest.MonkeyPatch, **overrides: str
) -> dict[str, Any]:
    """Parses the image's CMD with its ENV exactly as the uvicorn CLI would, without running it."""
    for name in [name for name in os.environ if name.startswith(("UVICORN_", "WEB_CONCURRENCY"))]:
        monkeypatch.delenv(name)
    for name, value in {**runtime.env(), **overrides}.items():
        if name.startswith("UVICORN_"):
            monkeypatch.setenv(name, value)
    command = runtime.one("CMD").exec_form()
    assert command[0] == "uvicorn"
    # --log-config must exist; map the image path back to its source in the repo.
    copied = {dst: src[0] for stage, src, dst in _copies(runtime) if stage is None}
    args = [str(ROOT / copied[arg]) if arg in copied else arg for arg in command[1:]]
    return dict(uvicorn_cli.make_context("uvicorn", args).params)


class TestStartup:
    def test_tini_is_pid_1_without_process_group_signalling(self, runtime: Stage) -> None:
        assert runtime.one("ENTRYPOINT").exec_form() == ["tini", "--"]

    def test_uvicorn_serves_the_app_factory(
        self, runtime: Stage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = _uvicorn_params(runtime, monkeypatch)
        module, _, attribute = params["app"].partition(":")
        factory = getattr(importlib.import_module(module), attribute)
        assert callable(factory)
        assert params["factory"] is True
        assert (params["host"], params["port"], params["workers"]) == ("0.0.0.0", 8000, 1)
        assert params["access_log"] is False, "CorrelationMiddleware writes the access log"
        assert params["server_header"] is False
        assert params["timeout_graceful_shutdown"] == 10

    def test_workers_host_and_port_are_configurable_at_run_time(
        self, runtime: Stage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = _uvicorn_params(
            runtime, monkeypatch, UVICORN_WORKERS="3", UVICORN_PORT="9000", UVICORN_HOST="::"
        )
        assert (params["host"], params["port"], params["workers"]) == ("::", 9000, 3)

    def test_uvicorn_variables_name_real_cli_options(self, runtime: Stage) -> None:
        options = {param.name for param in uvicorn_cli.params}
        compose_env = _service()["environment"]
        names = [name for name in [*runtime.env(), *compose_env] if name.startswith("UVICORN_")]
        assert names
        unknown = [name for name in names if name.removeprefix("UVICORN_").lower() not in options]
        assert not unknown, f"uvicorn silently ignores {unknown}"

    def test_port_is_exposed_and_sigterm_is_the_stop_signal(self, runtime: Stage) -> None:
        assert runtime.one("EXPOSE").value == runtime.env()["UVICORN_PORT"]
        assert runtime.one("STOPSIGNAL").value == "SIGTERM"

    def test_image_healthcheck_runs_the_bundled_probe(self, runtime: Stage) -> None:
        options, _, command = runtime.one("HEALTHCHECK").value.partition(" CMD ")
        argv = json.loads(command)
        assert argv[:3] == ["python", "-m", "multimodal_vision_pipeline.healthcheck"]
        assert argv[3:] == ["/healthz"], "image-level check is liveness only"
        flags = dict(re.findall(r"--([\w-]+)=(\S+)", options))
        paths = len(argv) - 3
        assert paths * healthcheck.TIMEOUT_SECONDS < _seconds(flags["timeout"])
        assert int(flags["retries"]) >= 1


class TestUvicornLoggingConfig:
    @pytest.fixture
    def isolated_logging(self) -> Iterator[None]:
        root = logging.getLogger()
        names = ("uvicorn", "uvicorn.error", "uvicorn.access")
        saved_root = (root.level, list(root.handlers))
        saved = {
            name: {
                **vars(logging.getLogger(name)),
                "handlers": list(logging.getLogger(name).handlers),
            }
            for name in names
        }
        # dictConfig would otherwise flush and unregister every live handler (pytest's too).
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(logging.config, "_clearExistingHandlers", lambda: None)
            yield
        for handler in root.handlers:
            if handler not in saved_root[1]:
                handler.close()
        root.handlers[:] = saved_root[1]
        root.setLevel(saved_root[0])
        for name, state in saved.items():
            logging.getLogger(name).__dict__.update(state)

    def test_uvicorn_logs_share_the_redacting_json_handler(
        self, isolated_logging: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        config = json.loads((ROOT / "docker" / "uvicorn-logging.json").read_text("utf-8"))
        logging.config.dictConfig(config)
        logging.getLogger("uvicorn.error").info("Started server process key=%s", TEST_API_KEY)
        entry = json.loads(stream.getvalue().splitlines()[-1])
        assert entry["logger"] == "uvicorn.error"
        assert TEST_API_KEY not in entry["message"]

        # The app lifespan replaces (not duplicates) the handler: same name.
        mvp_logging.configure_logging("INFO", stream=stream)
        named = [h for h in logging.getLogger().handlers if h.get_name() == "mvp-json"]
        assert len(named) == 1


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

_INTERPOLATION = re.compile(r"\$\{(\w+)(?::([-?])([^}]*))?\}")


def _service() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    services = document["services"]
    assert list(services) == ["api"]
    service = services["api"]
    assert isinstance(service, dict)
    return service


def _interpolate_defaults(value: object, required: dict[str, str]) -> str:
    """Resolves ``${VAR:-default}`` to its default and ``${VAR:?err}`` from ``required``."""

    def replace(match: re.Match[str]) -> str:
        name, operator, operand = match.groups()
        if operator == "?":
            return required[name]
        assert operator == "-", f"${{{name}}} has no default"
        return operand

    return _INTERPOLATION.sub(replace, str(value))


def _seconds(duration: str) -> float:
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", duration)
    assert parts, f"not a duration: {duration}"
    assert "".join(number + unit for number, unit in parts) == duration
    return sum(float(number) * units[unit] for number, unit in parts)


def _bytes(size: str) -> int:
    match = re.fullmatch(r"(\d+)([kmg]?)b?", size.lower())
    assert match, f"not a size: {size}"
    return int(match.group(1)) << 10 * "_kmg".index(match.group(2) or "_")


@pytest.fixture(scope="module")
def service() -> dict[str, Any]:
    return _service()


@pytest.fixture(scope="module")
def compose_env(service: dict[str, Any]) -> dict[str, str]:
    environment = service["environment"]
    assert isinstance(environment, dict), "use the mapping form"
    return {
        name: _interpolate_defaults(value, {"MVP_ANTHROPIC_API_KEY": TEST_API_KEY})
        for name, value in environment.items()
    }


class TestComposeFile:
    def test_builds_the_runtime_stage(self, service: dict[str, Any]) -> None:
        build = service["build"]
        assert (build["context"], build["target"]) == (".", "runtime")
        assert (ROOT / build["dockerfile"]).is_file()

    def test_healthcheck_polls_liveness_and_readiness(self, service: dict[str, Any]) -> None:
        check = service["healthcheck"]
        command = check["test"]
        assert command[:4] == ["CMD", "python", "-m", "multimodal_vision_pipeline.healthcheck"]
        assert command[4:] == ["/healthz", "/readyz"]
        assert len(command[4:]) * healthcheck.TIMEOUT_SECONDS < _seconds(check["timeout"])
        assert _seconds(check["timeout"]) < _seconds(check["interval"])
        assert check["retries"] >= 1
        assert _seconds(check["start_period"]) > 0

    def test_resource_limits_bound_rasterization_spikes(self, service: dict[str, Any]) -> None:
        resources = service["deploy"]["resources"]
        limits, reservations = resources["limits"], resources["reservations"]
        assert 0 < float(reservations["cpus"]) <= float(limits["cpus"])
        assert 0 < _bytes(reservations["memory"]) <= _bytes(limits["memory"])
        assert limits["pids"] > 0, "poppler subprocesses must not fork-bomb the host"

    def test_tmpfs_fits_concurrent_pdf_spools_within_memory(
        self, service: dict[str, Any], compose_env: dict[str, str]
    ) -> None:
        (tmp,) = [mount for mount in service["tmpfs"] if mount.startswith("/tmp:")]
        flags = tmp.removeprefix("/tmp:").split(",")
        assert {"noexec", "nosuid", "nodev"} <= set(flags)
        (size_flag,) = [flag for flag in flags if flag.startswith("size=")]
        size = _bytes(size_flag.removeprefix("size="))
        spools = int(compose_env["MVP_WORKER_MAX_CONCURRENCY"]) * int(
            compose_env["MVP_MAX_UPLOAD_BYTES"]
        )
        assert spools <= size
        memory = _bytes(service["deploy"]["resources"]["limits"]["memory"])
        assert size < memory, "tmpfs pages count against the memory limit"

    def test_container_hardening(self, service: dict[str, Any], runtime: Stage) -> None:
        assert service["user"] == runtime.one("USER").value == APP_UID_GID
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "cap_add" not in service
        assert "no-new-privileges:true" in service["security_opt"]
        assert not service.get("privileged", False)
        assert "volumes" not in service, "no host mounts (and never the Docker socket)"

    def test_port_published_on_loopback_by_default(self, service: dict[str, Any]) -> None:
        (mapping,) = service["ports"]
        assert _interpolate_defaults(mapping, {}) == "127.0.0.1:8000:8000"

    def test_stop_grace_period_covers_connection_and_job_draining(
        self, service: dict[str, Any], compose_env: dict[str, str]
    ) -> None:
        assert service["stop_signal"] == "SIGTERM"
        drain = float(compose_env["UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"]) + float(
            compose_env["MVP_WORKER_SHUTDOWN_GRACE_SECONDS"]
        )
        assert _seconds(service["stop_grace_period"]) > drain

    def test_single_uvicorn_worker_by_default(self, compose_env: dict[str, str]) -> None:
        # The job store is per process; more workers would 404 job lookups at random.
        assert compose_env["UVICORN_WORKERS"] == "1"

    def test_log_rotation(self, service: dict[str, Any]) -> None:
        options = service["logging"]["options"]
        assert _bytes(options["max-size"]) > 0
        assert int(options["max-file"]) >= 1


class TestComposeEnvironment:
    def test_every_setting_is_mapped_and_nothing_else(self, compose_env: dict[str, str]) -> None:
        prefix = VisionSettings.model_config.get("env_prefix", "")
        expected = {f"{prefix}{name}".upper() for name in VisionSettings.model_fields}
        mapped = {name for name in compose_env if name.startswith(prefix)}
        assert mapped == expected

    def test_every_variable_is_overridable(self, service: dict[str, Any]) -> None:
        for name, value in service["environment"].items():
            match = _INTERPOLATION.fullmatch(str(value))
            assert match, f"{name} is hard-coded"
            assert match.group(1) == name, f"{name} interpolates a different variable"

    def test_api_key_is_required_and_never_defaulted(self, service: dict[str, Any]) -> None:
        match = _INTERPOLATION.fullmatch(service["environment"]["MVP_ANTHROPIC_API_KEY"])
        assert match
        assert match.group(2) == "?"

    def test_defaults_are_valid_and_match_the_code(
        self, compose_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in [name for name in os.environ if name.startswith("MVP_")]:
            monkeypatch.delenv(name)
        for name, value in compose_env.items():
            monkeypatch.setenv(name, value)
        from_compose = VisionSettings()
        for name in compose_env:
            monkeypatch.delenv(name)
        assert from_compose == VisionSettings(anthropic_api_key=SecretStr(TEST_API_KEY))


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Docker/Go ``filepath.Match`` semantics plus ``**`` for any number of directories."""
    out, index = "", 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out, index = out + "(?:.*/)?", index + 3
        elif pattern.startswith("**", index):
            out, index = out + ".*", index + 2
        elif pattern[index] == "*":
            out, index = out + "[^/]*", index + 1
        elif pattern[index] == "?":
            out, index = out + "[^/]", index + 1
        elif pattern[index] == "[":
            end = pattern.index("]", index)
            out, index = out + pattern[index : end + 1], end + 1
        else:
            out, index = out + re.escape(pattern[index]), index + 1
    return re.compile(out)


@dataclass(frozen=True)
class IgnoreRules:
    rules: tuple[tuple[bool, str, re.Pattern[str]], ...]

    @classmethod
    def parse(cls, text: str) -> "IgnoreRules":
        rules = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            pattern = line.removeprefix("!").strip().strip("/")
            rules.append((negated, pattern, _pattern_regex(pattern)))
        return cls(tuple(rules))

    def excluded(self, path: str) -> bool:
        """Last matching rule wins; a rule matching a parent directory covers its contents."""
        parts = path.split("/")
        prefixes = ["/".join(parts[:depth]) for depth in range(1, len(parts) + 1)]
        result = False
        for negated, _, regex in self.rules:
            if any(regex.fullmatch(prefix) for prefix in prefixes):
                result = not negated
        return result

    def can_reinclude_below(self, directory: str) -> bool:
        return any(
            negated and (pattern.startswith(f"{directory}/") or pattern.startswith("**"))
            for negated, pattern, _ in self.rules
        )


@pytest.fixture(scope="module")
def ignore() -> IgnoreRules:
    return IgnoreRules.parse(DOCKERIGNORE.read_text(encoding="utf-8"))


def _context_files(ignore: IgnoreRules) -> list[str]:
    """Files ``docker build .`` would send, computed from the working tree."""
    sent: list[str] = []
    for current, dirs, files in os.walk(ROOT):
        base = Path(current).relative_to(ROOT).as_posix()
        prefix = "" if base == "." else f"{base}/"
        dirs[:] = [
            name
            for name in dirs
            if not ignore.excluded(prefix + name) or ignore.can_reinclude_below(prefix + name)
        ]
        sent.extend(prefix + name for name in files if not ignore.excluded(prefix + name))
    return sorted(sent)


class TestDockerignore:
    def test_matcher_semantics(self) -> None:
        rules = IgnoreRules.parse("*\n!src/\n**/__pycache__/\n**/*.py[cod]\n")
        assert rules.excluded("anything.txt")
        assert not rules.excluded("src/pkg/module.py")
        assert rules.excluded("src/pkg/__pycache__/module.cpython-312.pyc")
        assert rules.excluded("src/pkg/stale.pyo")
        assert not rules.excluded("src/pkg/notes.pyx")

    @pytest.mark.parametrize(
        "path",
        [
            ".cache/venv/bin/python",
            ".cache/uv/wheels/pillow.whl",
            ".venv/lib/python3.12/site-packages/x.py",
            "src/multimodal_vision_pipeline/__pycache__/app.cpython-312.pyc",
            "src/multimodal_vision_pipeline/api/app.pyc",
            "src/multimodal_vision_pipeline/.env",
            "src/multimodal_vision_pipeline/.DS_Store",
            "src/secrets/token.txt",
            "tests/test_api.py",
            ".coverage",
            ".coverage.host.1234",
            "htmlcov/index.html",
            ".pytest_cache/v/cache/lastfailed",
            ".git/HEAD",
            ".git/config",
            ".githooks/pre-commit",
            ".env",
            ".env.production",
            ".envrc",
            ".envrc.local",
            "deploy/tls/server.key",
            # Inside the re-included tree, where only the denylist protects.
            "src/multimodal_vision_pipeline/__pycache__/notes.txt",
            "src/multimodal_vision_pipeline/dev-cert.pem",
            "src/multimodal_vision_pipeline/signing.key",
            "src/multimodal_vision_pipeline/client.p12",
            "src/multimodal_vision_pipeline/ca.crt",
            "src/multimodal_vision_pipeline/id_rsa",
            "src/multimodal_vision_pipeline/.env.local",
            "src/.venv/bin/python",
            "src/multimodal_vision_pipeline.egg-info/PKG-INFO",
            "src/multimodal_vision_pipeline/.mypy_cache/3.12/x.json",
            "deploy/tls/server.pem",
            ".claude/settings.json",
            "CLAUDE.md",
            "docker-compose.yml",
            "dist/multimodal_vision_pipeline-0.1.0.whl",
        ],
    )
    def test_sensitive_cache_and_test_paths_are_excluded(
        self, ignore: IgnoreRules, path: str
    ) -> None:
        assert ignore.excluded(path)

    def test_every_build_input_is_in_the_context(
        self, ignore: IgnoreRules, dockerfile: Dockerfile
    ) -> None:
        inputs: set[str] = set()
        for stage in dockerfile.stages:
            for source, paths, _ in _copies(stage):
                if source is None:
                    inputs.update(path.rstrip("/") for path in paths)
            for command in stage.run_commands():
                inputs.update(re.findall(r"type=bind,source=([^,\s]+)", command))
        assert inputs >= {"pyproject.toml", "uv.lock", "src"}
        for path in sorted(inputs):
            assert (ROOT / path).exists(), path
            probe = f"{path}/multimodal_vision_pipeline/__init__.py" if path == "src" else path
            assert not ignore.excluded(probe), f"{path} is copied but ignored"

    def test_actual_context_is_only_build_inputs(self, ignore: IgnoreRules) -> None:
        sent = _context_files(ignore)
        assert "src/multimodal_vision_pipeline/py.typed" in sent
        roots = {path.split("/", 1)[0] for path in sent}
        assert roots <= {"pyproject.toml", "uv.lock", "README.md", "LICENSE", "src", "docker"}
        assert not [path for path in sent if "__pycache__" in path or path.endswith(".pyc")]


# ---------------------------------------------------------------------------
# Health probe (python -m multimodal_vision_pipeline.healthcheck)
# ---------------------------------------------------------------------------


def _transport(statuses: dict[str, int], seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        status = statuses.get(request.url.path)
        if status is None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(status, json={})

    return httpx.MockTransport(handler)


class TestHealthProbe:
    def test_healthy_when_every_path_answers_200(self) -> None:
        seen: list[str] = []
        code = healthcheck.main(
            ["/healthz", "/readyz"],
            environ={},
            transport=_transport({"/healthz": 200, "/readyz": 200}, seen),
        )
        assert code == healthcheck.EXIT_HEALTHY
        assert seen == ["http://127.0.0.1:8000/healthz", "http://127.0.0.1:8000/readyz"]

    def test_uses_the_configured_port_and_default_path(self) -> None:
        seen: list[str] = []
        code = healthcheck.main(
            [], environ={"UVICORN_PORT": "9001"}, transport=_transport({"/healthz": 200}, seen)
        )
        assert code == healthcheck.EXIT_HEALTHY
        assert seen == ["http://127.0.0.1:9001/healthz"]

    def test_not_ready_is_unhealthy(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = healthcheck.main(
            ["/healthz", "/readyz"],
            environ={},
            transport=_transport({"/healthz": 200, "/readyz": 503}, []),
        )
        assert code == healthcheck.EXIT_UNHEALTHY
        assert capsys.readouterr().err == "unhealthy /readyz: HTTP 503\n"

    def test_connection_failure_is_unhealthy(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = healthcheck.main(["/healthz"], environ={}, transport=_transport({}, []))
        assert code == healthcheck.EXIT_UNHEALTHY
        assert capsys.readouterr().err == "unhealthy /healthz: ConnectError\n"

    @pytest.mark.parametrize(
        ("argv", "environ", "message"),
        [
            (["healthz"], {}, "must start with '/'"),
            (["/healthz"], {"UVICORN_PORT": "http"}, "not a port number"),
            (["/healthz"], {"UVICORN_PORT": "70000"}, "out of range"),
            (["/healthz"], {"UVICORN_PORT": "0"}, "out of range"),
        ],
    )
    def test_usage_errors(
        self,
        argv: list[str],
        environ: dict[str, str],
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: list[str] = []
        code = healthcheck.main(argv, environ=environ, transport=_transport({}, seen))
        assert code == healthcheck.EXIT_USAGE
        assert message in capsys.readouterr().err
        assert not seen

    def test_defaults_to_process_argv_and_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["healthcheck", "/readyz"])
        monkeypatch.setenv("UVICORN_PORT", "8123")
        seen: list[str] = []
        assert healthcheck.main(transport=_transport({"/readyz": 200}, seen)) == 0
        assert seen == ["http://127.0.0.1:8123/readyz"]

    def test_probe_client_ignores_proxy_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        # Constructing a client opens no socket; only its configuration is inspected.
        with healthcheck.local_client("http://127.0.0.1:8000") as client:
            assert client.trust_env is False
            assert client.timeout == httpx.Timeout(healthcheck.TIMEOUT_SECONDS)
