"""Phase 1 verification: repository configuration, strict tooling and attribution policy.

Pure file inspection: no subprocesses, no network, no git invocation.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

import multimodal_vision_pipeline

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_NAME = "mikeZ98"
ALLOWED_EMAIL = "145975972+mikeZ98@users.noreply.github.com"

CORE_DEPENDENCIES = {
    "fastapi",
    "uvicorn",
    "pydantic",
    "pdf2image",
    "pillow",
    "httpx",
    "python-multipart",
}
DEV_DEPENDENCIES = {
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "pytest-socket",
    "ruff",
    "mypy",
    "bandit",
}
EXECUTABLES = (
    "scripts/audit-attribution.sh",
    ".githooks/pre-commit",
    ".githooks/commit-msg",
    ".githooks/pre-push",
)


def _requirement_name(spec: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec)
    assert match is not None, spec
    return match.group(0).lower().replace("_", "-")


def _shell_constant(text: str, name: str) -> str:
    match = re.search(rf"^readonly {name}='(.*)'$", text, re.MULTILINE)
    assert match is not None, f"{name} not defined in _policy.sh"
    return match.group(1)


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def tool(pyproject: dict[str, Any]) -> dict[str, Any]:
    section: dict[str, Any] = pyproject["tool"]
    return section


@pytest.fixture(scope="module")
def policy() -> str:
    return (ROOT / ".githooks" / "_policy.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Required files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "relpath",
    [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        ".envrc",
        ".claude/settings.json",
        ".githooks/_policy.sh",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "README.md",
        "LICENSE",
        "src/multimodal_vision_pipeline/__init__.py",
        "src/multimodal_vision_pipeline/py.typed",
        *EXECUTABLES,
    ],
)
def test_required_file_present(relpath: str) -> None:
    assert (ROOT / relpath).is_file(), f"missing {relpath}"


@pytest.mark.parametrize("relpath", EXECUTABLES)
def test_scripts_are_executable_bash(relpath: str) -> None:
    path = ROOT / relpath
    assert os.access(path, os.X_OK), f"{relpath} is not executable"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


# ---------------------------------------------------------------------------
# Project metadata & dependencies
# ---------------------------------------------------------------------------
def test_python_312_plus(pyproject: dict[str, Any]) -> None:
    assert pyproject["project"]["requires-python"].startswith(">=3.12")
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_sole_author_metadata(pyproject: dict[str, Any]) -> None:
    assert pyproject["project"]["authors"] == [{"name": ALLOWED_NAME, "email": ALLOWED_EMAIL}]


def test_core_dependencies(pyproject: dict[str, Any]) -> None:
    declared = {_requirement_name(d) for d in pyproject["project"]["dependencies"]}
    assert declared >= CORE_DEPENDENCIES


def test_dev_dependencies(pyproject: dict[str, Any]) -> None:
    declared = {_requirement_name(d) for d in pyproject["dependency-groups"]["dev"]}
    assert declared >= DEV_DEPENDENCIES


def test_every_dependency_is_locked(pyproject: dict[str, Any]) -> None:
    with (ROOT / "uv.lock").open("rb") as fh:
        lock = tomllib.load(fh)
    locked = {pkg["name"] for pkg in lock["package"] if "version" in pkg}
    declared = {
        _requirement_name(d)
        for d in [*pyproject["project"]["dependencies"], *pyproject["dependency-groups"]["dev"]]
    }
    assert declared <= locked, f"unlocked: {sorted(declared - locked)}"


def test_package_version_matches_metadata(pyproject: dict[str, Any]) -> None:
    assert multimodal_vision_pipeline.__version__ == pyproject["project"]["version"]


# ---------------------------------------------------------------------------
# Tooling strictness
# ---------------------------------------------------------------------------
def test_mypy_strict(tool: dict[str, Any]) -> None:
    mypy = tool["mypy"]
    assert mypy["strict"] is True
    assert mypy["python_version"] == "3.12"
    assert "pydantic.mypy" in mypy["plugins"]
    assert mypy["warn_unreachable"] is True
    assert set(mypy["files"]) == {"src", "tests"}


def test_ruff_linter_and_formatter(tool: dict[str, Any]) -> None:
    ruff = tool["ruff"]
    assert ruff["target-version"] == "py312"
    assert {"E", "F", "I", "B", "S", "UP", "ASYNC", "RUF"} <= set(ruff["lint"]["select"])
    assert "format" in ruff


def test_pytest_hermetic_and_strict(tool: dict[str, Any]) -> None:
    ini = tool["pytest"]["ini_options"]
    addopts = set(ini["addopts"])
    assert {"--strict-markers", "--strict-config", "--disable-socket", "--cov-branch"} <= addopts
    assert "--allow-unix-socket" not in addopts
    assert ini["asyncio_mode"] == "auto"
    assert ini["filterwarnings"][0] == "error"


def test_coverage_threshold(tool: dict[str, Any]) -> None:
    assert tool["coverage"]["run"]["branch"] is True
    assert tool["coverage"]["report"]["fail_under"] >= 90


def test_bandit_scans_src(tool: dict[str, Any]) -> None:
    assert tool["bandit"]["targets"] == ["src"]


def test_caches_pinned_inside_repo(tool: dict[str, Any]) -> None:
    assert tool["ruff"]["cache-dir"].startswith(".cache/")
    assert tool["mypy"]["cache_dir"].startswith(".cache/")
    assert tool["pytest"]["ini_options"]["cache_dir"].startswith(".cache/")
    assert tool["coverage"]["run"]["data_file"].startswith(".cache/")


def test_gitignore_blocks_caches_and_secrets() -> None:
    lines = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".cache/", ".venv/", ".env", "__pycache__/"} <= lines


# ---------------------------------------------------------------------------
# Attribution policy
# ---------------------------------------------------------------------------
def test_policy_identity(policy: str) -> None:
    assert f'readonly POLICY_ALLOWED_NAME="{ALLOWED_NAME}"' in policy
    assert f'readonly POLICY_ALLOWED_EMAIL="{ALLOWED_EMAIL}"' in policy


def test_envrc_sets_identity_and_hooks() -> None:
    envrc = (ROOT / ".envrc").read_text(encoding="utf-8")
    assert "config core.hooksPath .githooks" in envrc
    assert f'config user.name "{ALLOWED_NAME}"' in envrc
    assert f'config user.email "{ALLOWED_EMAIL}"' in envrc
    assert "config user.useConfigOnly true" in envrc


def test_claude_settings_disable_attribution() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["includeCoAuthoredBy"] is False
    assert settings["attribution"] == {"commit": "", "pr": ""}


def test_audit_script_uses_shared_policy() -> None:
    text = (ROOT / "scripts" / "audit-attribution.sh").read_text(encoding="utf-8")
    assert 'source "$ROOT/.githooks/_policy.sh"' in text
    for check in ("policy_check_identity", "policy_check_message"):
        assert check in text


@pytest.mark.parametrize(
    "message",
    [
        "chore: x\n\nCo-authored-by: Someone <a@b.c>",
        "chore: x\n\nGenerated-by: tool",
        "feat: x\n\nGenerated with some assistant",
        "fix: x\n\nAssisted-by: bot",
        "fix: written by Claude",
        "docs: courtesy of ChatGPT",
        "docs: via GitHub Copilot",
        "chore: 🤖 automated",
    ],
)
def test_forbidden_message_rejected(policy: str, message: str) -> None:
    pattern = re.compile(_shell_constant(policy, "POLICY_FORBIDDEN_MSG_RE"), re.IGNORECASE)
    assert pattern.search(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        "chore(config): initialize pyproject.toml, tooling, and strict verification gates",
        "docs: document rules in CLAUDE.md",
    ],
)
def test_clean_message_accepted(policy: str, message: str) -> None:
    pattern = re.compile(_shell_constant(policy, "POLICY_FORBIDDEN_MSG_RE"), re.IGNORECASE)
    assert pattern.search(message.replace("CLAUDE.md", "")) is None


@pytest.mark.parametrize(
    ("subject", "valid"),
    [
        ("chore(config): initialize pyproject.toml, tooling, and strict verification gates", True),
        ("feat(schemas)!: break the contract", True),
        ("update stuff", False),
        ("Feat: capitalised type", False),
        ("feat: " + "x" * 120, False),
    ],
)
def test_conventional_subject(policy: str, subject: str, valid: bool) -> None:
    pattern = re.compile(_shell_constant(policy, "POLICY_CONVENTIONAL_RE"))
    assert (pattern.search(subject) is not None) is valid
